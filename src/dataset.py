"""Чтение YOLO-датасета (images/, labels/, data.yaml) и аугментации."""
from __future__ import annotations

import hashlib
import logging
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .boxes import letterbox, parse_label_line
from .config import Config

log = logging.getLogger("dataset")

SUPPORTED_EXT = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


class _LruImageCache:
    """LRU-кэш декодированных изображений с байтовым бюджетом."""

    def __init__(self, budget_bytes: int) -> None:
        self._budget = max(int(budget_bytes), 1)
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._bytes = 0

    def get(
        self,
        key: str,
        loader: Callable[[], np.ndarray],
    ) -> np.ndarray:
        cache = self._data

        img = cache.get(key)
        if img is not None:
            cache.move_to_end(key)
            return img

        img = loader()

        # Если одно изображение само по себе больше бюджета,
        # всё равно возвращаем его, но не пытаемся держать несколько
        # таких изображений в кеше.
        cache[key] = img
        self._bytes += img.nbytes

        while self._bytes > self._budget and len(cache) > 1:
            _, old = cache.popitem(last=False)
            self._bytes -= old.nbytes

        return img


def read_data_yaml(path: Path) -> dict:
    """Читает data.yaml."""
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Cannot read data.yaml: {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Invalid data.yaml structure: {path}")

    return data


def _resolve_dir(root: Path, value: str | Path) -> Path:
    """Разрешает относительный путь относительно root."""
    p = Path(value)

    if p.is_absolute():
        return p

    return (root / p).resolve()


def _collect_images_from_dir(directory: Path) -> list[Path]:
    """Рекурсивно собирает изображения из каталога."""
    if not directory.exists() or not directory.is_dir():
        return []

    result: list[Path] = []

    for ext in SUPPORTED_EXT:
        result.extend(directory.glob(f"**/*{ext}"))

    result.sort()
    return result


def _read_image_list(path: Path, root: Path) -> list[Path]:
    """
    Читает YOLO-style train.txt / val.txt.

    В строках допускаются:
      - абсолютные пути;
      - пути относительно data.yaml/root;
      - пути относительно самого txt-файла.
    """
    result: list[Path] = []

    if not path.exists() or not path.is_file():
        return result

    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
    except OSError as exc:
        log.warning("Cannot read image list %s: %s", path, exc)
        return result

    for raw in lines:
        value = raw.strip()

        if not value or value.startswith("#"):
            continue

        candidate = Path(value)

        candidates: list[Path] = []

        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            candidates.append(root / candidate)
            candidates.append(path.parent / candidate)

        resolved = None

        for item in candidates:
            item = item.resolve()
            if item.exists() and item.is_file():
                resolved = item
                break

        if resolved is None:
            # Сохраняем первый вариант, чтобы ошибка была видна
            # в общей диагностике датасета.
            resolved = candidates[0].resolve()

        if resolved.suffix.lower() in SUPPORTED_EXT:
            result.append(resolved)

    return sorted(set(result))


def _resolve_data_entry(
    root: Path,
    value: str | list | tuple,
) -> list[Path]:
    """
    Разрешает train/val из data.yaml.

    Поддерживает:
      train: images/train
      train: /absolute/path/images/train
      train: train.txt
      train:
        - images/a
        - images/b
    """
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]

    result: list[Path] = []

    for item in values:
        if not isinstance(item, str):
            continue

        path = _resolve_dir(root, item)

        if path.is_file():
            if path.suffix.lower() == ".txt":
                result.extend(_read_image_list(path, root))
            elif path.suffix.lower() in SUPPORTED_EXT:
                result.append(path)
            continue

        if path.is_dir():
            result.extend(_collect_images_from_dir(path))

    return sorted(set(result))


def find_image_sets(
    root: Path,
    data: dict,
    images_dir: str,
    labels_dir: str,
) -> tuple[list[Path], list[Path]]:
    """
    Определяет изображения train/val из data.yaml.

    Возвращает именно списки изображений, а не каталоги.
    Это важно для корректной обработки train.txt / val.txt
    и раздельных директорий.
    """
    del labels_dir  # оставлено в сигнатуре для совместимости API

    train_paths: list[Path] = []
    val_paths: list[Path] = []

    if "train" in data:
        train_paths = _resolve_data_entry(root, data["train"])

    if "val" in data:
        val_paths = _resolve_data_entry(root, data["val"])

    if not train_paths and not val_paths:
        fallback = _resolve_dir(root, images_dir)

        if fallback.exists() and fallback.is_dir():
            train_paths = _collect_images_from_dir(fallback)
            log.warning(
                "data.yaml has no usable train/val entries; "
                "using all images under %s as training data",
                fallback,
            )

    return train_paths, val_paths


def _labels_path_for(
    image_path: Path,
    root: Path,
    images_dir: str,
    labels_dir: str,
) -> Path:
    """
    Путь к label по стандартному YOLO-соглашению.

    Основной случай:
        root/images/train/a.jpg
        root/labels/train/a.txt

    Если image_path не находится внутри images_dir,
    сохраняется basename fallback.
    """
    img = image_path.resolve()

    image_root = (root / images_dir).resolve()
    labels_root = (root / labels_dir).resolve()

    try:
        rel = img.relative_to(image_root)
    except ValueError:
        # Для train.txt с нестандартным абсолютным путём
        # пробуем сохранить структуру относительно ближайшего
        # images-компонента пути.
        parts = img.parts

        image_index = -1
        for i, part in enumerate(parts):
            if part.lower() == "images":
                image_index = i
                break

        if image_index >= 0 and image_index + 1 < len(parts):
            rel = Path(*parts[image_index + 1 :])
        else:
            rel = Path(img.name)

    return labels_root / rel.with_suffix(".txt")


def _log_missing_labels(
    image_paths: list[Path],
    root: Path,
    images_dir: str,
    labels_dir: str,
    split: str,
) -> int:
    """Считает и логирует изображения без labels."""
    missing = 0

    for path in image_paths:
        label_path = _labels_path_for(
            path,
            root,
            images_dir,
            labels_dir,
        )

        if not label_path.exists():
            missing += 1

    log.warning(
        "%s: %d images total, %d without labels (treated as background)",
        split.capitalize(),
        len(image_paths),
        missing,
    )

    return missing


class YoloDataset(Dataset):
    """
    Датасет из YOLO-разметки.

    Возвращает:
        image: Tensor [3,H,W], float32, [0,1]
        target: ndarray [N,5]
            [x1,y1,x2,y2,class_id]
    """

    def __init__(
        self,
        cfg: Config,
        data: dict,
        image_dirs: list[Path],
        root: Path,
        split: str,
        input_size: int,
        augment: bool = False,
    ) -> None:
        self.cfg = cfg
        self.data = data
        self.root = root.resolve()
        self.images_dir = str(cfg.dataset["images_dir"])
        self.labels_dir = str(cfg.dataset["labels_dir"])
        self.split = split
        self.input_size = int(input_size)
        self.augment = bool(augment)
        self.num_classes = int(cfg.model["num_classes"])

        aug = cfg.augmentation

        self.flip_lr = float(aug.get("flip_lr", 0.0))
        self.flip_ud = float(aug.get("flip_ud", 0.0))
        self.bc = float(aug.get("brightness_contrast", 0.0))

        self.scale_crop = float(aug.get("scale_crop", 0.0))
        self.scale_min = float(aug.get("scale_min", 0.7))
        self.scale_max = float(aug.get("scale_max", 1.0))

        self.mosaic_prob = float(aug.get("mosaic_prob", 0.0))
        self.mixup_prob = float(aug.get("mixup_prob", 0.0))

        # Минимальный размер GT после Mosaic в пикселях.
        # Для твоих мелких FPV-целей специально оставляем низким.
        self.mosaic_min_box_px = float(
            aug.get("mosaic_min_box_px", 1.0)
        )

        self.image_paths: list[Path] = []

        # image_dirs теперь может содержать:
        #   - директории;
        #   - отдельные изображения;
        #   - txt-файлы.
        for item in image_dirs:
            path = Path(item)

            if path.is_file():
                if path.suffix.lower() == ".txt":
                    self.image_paths.extend(
                        _read_image_list(path, self.root)
                    )
                elif path.suffix.lower() in SUPPORTED_EXT:
                    self.image_paths.append(path.resolve())
                continue

            if not path.exists():
                log.warning("Image path does not exist: %s", path)
                continue

            if path.is_dir():
                self.image_paths.extend(
                    _collect_images_from_dir(path)
                )

        self.image_paths = sorted(set(self.image_paths))

        if not self.image_paths:
            raise RuntimeError(
                f"No images found for split '{split}' "
                f"in {[str(d) for d in image_dirs]}"
            )

        self.empty_bg_logged: set[str] = set()

        self.labels_cache: dict[str, np.ndarray] = {}

        disk_cache = (
            bool(cfg.dataset.get("cache_to_disk", False))
            and split == "train"
        )

        cache_enabled = bool(
            cfg.dataset.get("cache_images", True)
        )

        cache_mb = int(
            cfg.dataset.get("cache_images_mb", 512)
        )

        self._image_cache = (
            _LruImageCache(cache_mb * 1024 * 1024)
            if cache_enabled and not disk_cache
            else None
        )

        if disk_cache:
            cache_dir = (
                str(cfg.dataset.get("cache_dir", "")).strip()
                or f"{self.root}/.cache"
            )

            self._cache_root = (
                Path(cache_dir)
                .expanduser()
                .resolve()
            )

            self._cache_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._cache_side = int(
                cfg.dataset.get(
                    "cache_side",
                    0,
                )
                or self.input_size
            )
        else:
            self._cache_root = None
            self._cache_side = 0

        self._validate_dataset_config(data)

        log.info(
            "%s dataset: %d images, augment=%s",
            split,
            len(self.image_paths),
            self.augment,
        )

    def _validate_dataset_config(self, data: dict) -> None:
        """Проверяет согласованность classes / data.yaml."""
        names = data.get("names")

        if isinstance(names, dict) and names:
            try:
                ids = [int(k) for k in names.keys()]
                yaml_num_classes = max(ids) + 1
            except (TypeError, ValueError):
                log.warning(
                    "Invalid class IDs in data.yaml names: %r",
                    names,
                )
                return

            if yaml_num_classes != self.num_classes:
                log.warning(
                    "data.yaml declares %d classes but config sets %d; "
                    "using config value",
                    yaml_num_classes,
                    self.num_classes,
                )

            if self.num_classes == 1:
                if 0 not in ids:
                    log.warning(
                        "Single-class dataset has no class id 0: %r",
                        ids,
                    )

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_boxes(self, image_path: Path) -> np.ndarray:
        """
        Читает YOLO label.

        Возвращает:
            float32 ndarray [N,5]
            [class,cx,cy,w,h]

        Здесь намеренно выполняется строгая валидация:
        NaN/Inf, неизвестные классы и невалидные размеры
        не должны попасть в loss.
        """
        label_path = _labels_path_for(
            image_path,
            self.root,
            self.images_dir,
            self.labels_dir,
        )

        key = str(image_path.resolve())

        cached = self.labels_cache.get(key)
        if cached is not None:
            return cached.copy()

        boxes: list[
            tuple[int, float, float, float, float]
        ] = []

        if label_path.exists():
            try:
                with open(
                    label_path,
                    "r",
                    encoding="utf-8-sig",
                ) as f:
                    for line_no, line in enumerate(f, 1):
                        parsed = parse_label_line(
                            line,
                            self.num_classes,
                        )

                        if parsed is None:
                            continue

                        cls_id, cx, cy, bw, bh = parsed

                        values = np.asarray(
                            [cx, cy, bw, bh],
                            dtype=np.float32,
                        )

                        if not np.isfinite(values).all():
                            log.warning(
                                "Invalid NaN/Inf label %s:%d",
                                label_path,
                                line_no,
                            )
                            continue

                        if cls_id < 0 or cls_id >= self.num_classes:
                            log.warning(
                                "Invalid class %s:%d: %s",
                                label_path,
                                line_no,
                                cls_id,
                            )
                            continue

                        if bw <= 0.0 or bh <= 0.0:
                            log.warning(
                                "Invalid box size %s:%d: %s",
                                label_path,
                                line_no,
                                parsed,
                            )
                            continue

                        # YOLO coordinates должны быть нормализованы.
                        # Небольшой допуск позволяет пережить погрешности
                        # экспортёра, но не пропускает полностью битые строки.
                        if (
                            cx < -0.01
                            or cx > 1.01
                            or cy < -0.01
                            or cy > 1.01
                            or bw > 1.01
                            or bh > 1.01
                        ):
                            log.warning(
                                "Out-of-range YOLO label %s:%d: %s",
                                label_path,
                                line_no,
                                parsed,
                            )
                            continue

                        # Центр можно слегка зажать в диапазон.
                        # Размеры не зажимаем здесь: ниже bbox будет
                        # корректно пересечён с границами изображения.
                        cx = float(np.clip(cx, 0.0, 1.0))
                        cy = float(np.clip(cy, 0.0, 1.0))

                        boxes.append(
                            (
                                int(cls_id),
                                cx,
                                cy,
                                float(bw),
                                float(bh),
                            )
                        )

            except OSError as exc:
                log.warning(
                    "Failed to read label %s: %s",
                    label_path,
                    exc,
                )
        else:
            if self.split == "val":
                if key not in self.empty_bg_logged:
                    self.empty_bg_logged.add(key)

                    log.info(
                        "No label for %s, treated as empty background",
                        image_path,
                    )

        arr = np.asarray(
            boxes,
            dtype=np.float32,
        ).reshape(-1, 5)

        self.labels_cache[key] = arr

        return arr.copy()

    def _read_image(self, image_path: Path) -> np.ndarray:
        if self._cache_root is not None:
            return self._cached_image(image_path)

        return self._plain_image(image_path)

    def _plain_image(
        self,
        image_path: Path,
    ) -> np.ndarray:
        if self._image_cache is None:
            return self._decode_image(image_path)

        return self._image_cache.get(
            str(image_path.resolve()),
            lambda: self._decode_image(image_path),
        )

    def _cached_image(
        self,
        image_path: Path,
    ) -> np.ndarray:
        npy_path = self._npy_path(image_path)

        if npy_path.exists():
            try:
                return np.load(
                    npy_path,
                    allow_pickle=False,
                )
            except Exception as exc:
                log.warning(
                    "Invalid image cache %s: %s; rebuilding",
                    npy_path,
                    exc,
                )

        img = self._decode_image(image_path)
        img = self._downscale(img)

        self._write_npy(npy_path, img)

        return img

    def _npy_path(
        self,
        image_path: Path,
    ) -> Path:
        key = hashlib.md5(
            str(image_path.resolve()).encode("utf-8")
        ).hexdigest()[:20]

        return self._cache_root / f"{key}.npy"

    def _downscale(
        self,
        img: np.ndarray,
    ) -> np.ndarray:
        if self._cache_side <= 0:
            return img

        h, w = img.shape[:2]

        if max(h, w) <= self._cache_side:
            return img

        import cv2

        scale = self._cache_side / float(max(h, w))

        nw = max(
            1,
            int(round(w * scale)),
        )

        nh = max(
            1,
            int(round(h * scale)),
        )

        return cv2.resize(
            img,
            (nw, nh),
            interpolation=cv2.INTER_AREA,
        )

    def _write_npy(
        self,
        path: Path,
        img: np.ndarray,
    ) -> None:
        try:
            tmp = path.with_suffix(".tmp.npy")

            np.save(
                tmp,
                img,
                allow_pickle=False,
            )

            os.replace(tmp, path)

        except OSError as exc:
            log.warning(
                "Failed to write image cache %s: %s",
                path,
                exc,
            )

    def build_disk_cache(self, log_fn=None) -> int:
        if self._cache_root is None:
            return 0

        n = len(self.image_paths)
        built = 0

        for i, path in enumerate(
            self.image_paths,
            1,
        ):
            cache_path = self._npy_path(path)

            if not cache_path.exists():
                self._cached_image(path)
                built += 1

            if log_fn is not None and i % 500 == 0:
                log_fn(
                    f"    cache: {i}/{n} images"
                )

        return built

    def _decode_image(
        self,
        image_path: Path,
    ) -> np.ndarray:
        import cv2

        img = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if img is None:
            raise RuntimeError(
                f"Cannot read image: {image_path}"
            )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB,
        )

        return np.ascontiguousarray(img)

    def _augment_norm(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Аугментации в нормализованных YOLO-координатах."""

        boxes = boxes.copy()

        flip_lr = (
            self.flip_lr > 0.0
            and random.random() < self.flip_lr
        )

        flip_ud = (
            self.flip_ud > 0.0
            and random.random() < self.flip_ud
        )

        if flip_lr:
            img = np.ascontiguousarray(
                img[:, ::-1, :]
            )

            if boxes.shape[0]:
                boxes[:, 1] = (
                    1.0 - boxes[:, 1]
                )

        if flip_ud:
            img = np.ascontiguousarray(
                img[::-1, :, :]
            )

            if boxes.shape[0]:
                boxes[:, 2] = (
                    1.0 - boxes[:, 2]
                )

        if self.bc > 0.0:
            import cv2

            alpha = 1.0 + random.uniform(
                -self.bc,
                self.bc,
            )

            beta = random.randint(
                -int(30 * self.bc),
                int(30 * self.bc),
            )

            img = cv2.convertScaleAbs(
                img,
                alpha=alpha,
                beta=beta,
            )

        if (
            self.scale_crop > 0.0
            and random.random() < self.scale_crop
        ):
            img, boxes = self._random_scale_crop(
                img,
                boxes,
            )

        boxes = self._clip_normalized_boxes(boxes)

        return img, boxes

    @staticmethod
    def _clip_normalized_boxes(
        boxes: np.ndarray,
    ) -> np.ndarray:
        """
        Пересекает YOLO-боксы с границами [0,1].

        Нельзя просто clip-нуть cx/cy:
        сначала переводим bbox в xyxy.
        """
        if boxes.shape[0] == 0:
            return boxes

        result = boxes.copy()

        x1 = (
            result[:, 1]
            - result[:, 3] / 2.0
        )

        y1 = (
            result[:, 2]
            - result[:, 4] / 2.0
        )

        x2 = (
            result[:, 1]
            + result[:, 3] / 2.0
        )

        y2 = (
            result[:, 2]
            + result[:, 4] / 2.0
        )

        x1 = np.clip(x1, 0.0, 1.0)
        y1 = np.clip(y1, 0.0, 1.0)
        x2 = np.clip(x2, 0.0, 1.0)
        y2 = np.clip(y2, 0.0, 1.0)

        keep = (
            (x2 > x1)
            & (y2 > y1)
        )

        if not keep.any():
            return np.empty(
                (0, 5),
                dtype=np.float32,
            )

        result = result[keep]

        result[:, 1] = (
            x1[keep] + x2[keep]
        ) / 2.0

        result[:, 2] = (
            y1[keep] + y2[keep]
        ) / 2.0

        result[:, 3] = (
            x2[keep] - x1[keep]
        )

        result[:, 4] = (
            y2[keep] - y1[keep]
        )

        return result

    def _random_scale_crop(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Случайный crop с последующим масштабированием."""

        import cv2

        h, w = img.shape[:2]

        s = random.uniform(
            self.scale_min,
            self.scale_max,
        )

        if s >= 1.0 - 1e-6:
            return img, boxes

        new_w = max(
            1,
            int(round(w * s)),
        )

        new_h = max(
            1,
            int(round(h * s)),
        )

        if new_w >= w or new_h >= h:
            return img, boxes

        ox = random.randint(
            0,
            w - new_w,
        )

        oy = random.randint(
            0,
            h - new_h,
        )

        crop = img[
            oy : oy + new_h,
            ox : ox + new_w,
        ]

        img = cv2.resize(
            crop,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        boxes = boxes.copy()

        if boxes.shape[0]:
            boxes[:, 1] = (
                boxes[:, 1]
                - ox / float(w)
            ) / s

            boxes[:, 2] = (
                boxes[:, 2]
                - oy / float(h)
            ) / s

            boxes[:, 3] /= s
            boxes[:, 4] /= s

            boxes = self._clip_normalized_boxes(
                boxes
            )

        return img, boxes

    def load_original_sample(
        self,
        image_path: Path,
    ) -> tuple[
        np.ndarray,
        int,
        int,
        np.ndarray,
    ]:
        """
        Оригинальное RGB-изображение и GT.

        GT:
            [x1,y1,x2,y2,class]
        """
        img = self._read_image(image_path)

        h, w = img.shape[:2]

        boxes_yolo = self.load_boxes(
            image_path
        )

        gt = np.empty(
            (0, 5),
            dtype=np.float32,
        )

        if boxes_yolo.shape[0]:
            x1 = (
                boxes_yolo[:, 1]
                - boxes_yolo[:, 3] / 2.0
            ) * w

            y1 = (
                boxes_yolo[:, 2]
                - boxes_yolo[:, 4] / 2.0
            ) * h

            x2 = (
                boxes_yolo[:, 1]
                + boxes_yolo[:, 3] / 2.0
            ) * w

            y2 = (
                boxes_yolo[:, 2]
                + boxes_yolo[:, 4] / 2.0
            ) * h

            x1 = np.clip(x1, 0.0, float(w))
            y1 = np.clip(y1, 0.0, float(h))
            x2 = np.clip(x2, 0.0, float(w))
            y2 = np.clip(y2, 0.0, float(h))

            keep = (
                (x2 > x1)
                & (y2 > y1)
            )

            if keep.any():
                gt = np.stack(
                    [
                        x1,
                        y1,
                        x2,
                        y2,
                        boxes_yolo[:, 0],
                    ],
                    axis=-1,
                )[keep]

        return img, w, h, gt

    def _build_target(
        self,
        boxes_yolo: np.ndarray,
        eff_w: int,
        eff_h: int,
        pad_x: int,
        pad_y: int,
    ) -> np.ndarray:
        """
        YOLO normalized -> xyxy pixels.

        Формат:
            [x1,y1,x2,y2,class]
        """
        if boxes_yolo.shape[0] == 0:
            return np.empty(
                (0, 5),
                dtype=np.float32,
            )

        boxes = boxes_yolo.copy()

        cx = boxes[:, 1] * eff_w + pad_x
        cy = boxes[:, 2] * eff_h + pad_y

        bw = boxes[:, 3] * eff_w
        bh = boxes[:, 4] * eff_h

        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        # Защита от численных ошибок.
        x1 = np.clip(
            x1,
            0.0,
            float(self.input_size),
        )

        y1 = np.clip(
            y1,
            0.0,
            float(self.input_size),
        )

        x2 = np.clip(
            x2,
            0.0,
            float(self.input_size),
        )

        y2 = np.clip(
            y2,
            0.0,
            float(self.input_size),
        )

        keep = (
            (x2 > x1)
            & (y2 > y1)
        )

        if not keep.any():
            return np.empty(
                (0, 5),
                dtype=np.float32,
            )

        return np.stack(
            [
                x1,
                y1,
                x2,
                y2,
                boxes[:, 0],
            ],
            axis=-1,
        )[keep].astype(
            np.float32,
            copy=False,
        )

    def _load_plain(
        self,
        index: int,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Загружает изображение без аугментаций."""
        path = self.image_paths[index]

        img = self._read_image(path)

        boxes_yolo = self.load_boxes(path)

        img, _scale, pad_x, pad_y = letterbox(
            img,
            self.input_size,
        )

        img = np.ascontiguousarray(img)

        img_t = (
            torch.from_numpy(
                img.transpose(2, 0, 1)
            )
            .float()
            .div_(255.0)
        )

        eff_w = img.shape[1] - 2 * pad_x
        eff_h = img.shape[0] - 2 * pad_y

        target = self._build_target(
            boxes_yolo,
            eff_w,
            eff_h,
            pad_x,
            pad_y,
        )

        return img_t, target

    def _mosaic(
        self,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Mosaic 2x2.

        Важный момент для FPV:
        маленькие боксы не отбрасываются порогом 4px.
        """
        import cv2

        res = self.input_size
        cell = res // 2

        canvas = np.zeros(
            (res, res, 3),
            dtype=np.uint8,
        )

        parts: list[np.ndarray] = []

        safe = len(self.image_paths)

        for i in range(4):
            path = self.image_paths[
                random.randrange(safe)
            ]

            try:
                img, w, h, gt = (
                    self.load_original_sample(path)
                )
            except Exception as exc:
                log.warning(
                    "mosaic: failed %s: %s",
                    path,
                    exc,
                )
                continue

            if (
                img is None
                or w <= 0
                or h <= 0
            ):
                continue

            f = random.uniform(
                0.6,
                1.0,
            )

            if w >= h:
                nw = max(
                    8,
                    int(round(cell * f)),
                )

                nh = max(
                    8,
                    int(round(nw * h / w)),
                )
            else:
                nh = max(
                    8,
                    int(round(cell * f)),
                )

                nw = max(
                    8,
                    int(round(nh * w / h)),
                )

            nw = min(nw, cell)
            nh = min(nh, cell)

            resized = cv2.resize(
                img,
                (nw, nh),
                interpolation=cv2.INTER_LINEAR,
            )

            col = i % 2
            row = i // 2

            gx0 = (
                col * cell
                + random.randint(
                    0,
                    max(0, cell - nw),
                )
            )

            gy0 = (
                row * cell
                + random.randint(
                    0,
                    max(0, cell - nh),
                )
            )

            canvas[
                gy0 : gy0 + nh,
                gx0 : gx0 + nw,
            ] = resized

            if gt.shape[0]:
                bx = (
                    gt[:, 0] / w * nw
                    + gx0
                )

                by = (
                    gt[:, 1] / h * nh
                    + gy0
                )

                bx2 = (
                    gt[:, 2] / w * nw
                    + gx0
                )

                by2 = (
                    gt[:, 3] / h * nh
                    + gy0
                )

                cc = np.stack(
                    [
                        bx,
                        by,
                        bx2,
                        by2,
                        gt[:, 4],
                    ],
                    axis=-1,
                )

                cc[:, [0, 2]] = np.clip(
                    cc[:, [0, 2]],
                    gx0,
                    gx0 + nw,
                )

                cc[:, [1, 3]] = np.clip(
                    cc[:, [1, 3]],
                    gy0,
                    gy0 + nh,
                )

                widths = (
                    cc[:, 2] - cc[:, 0]
                )

                heights = (
                    cc[:, 3] - cc[:, 1]
                )

                keep = (
                    (widths >= self.mosaic_min_box_px)
                    & (heights >= self.mosaic_min_box_px)
                )

                if keep.any():
                    parts.append(
                        cc[keep]
                    )

        if self.bc > 0.0:
            alpha = 1.0 + random.uniform(
                -self.bc,
                self.bc,
            )

            beta = random.randint(
                -int(30 * self.bc),
                int(30 * self.bc),
            )

            canvas = cv2.convertScaleAbs(
                canvas,
                alpha=alpha,
                beta=beta,
            )

        if parts:
            target = np.concatenate(
                parts,
                axis=0,
            ).astype(
                np.float32,
                copy=False,
            )
        else:
            target = np.empty(
                (0, 5),
                dtype=np.float32,
            )

        img_t = (
            torch.from_numpy(
                np.ascontiguousarray(
                    canvas.transpose(2, 0, 1)
                )
            )
            .float()
            .div_(255.0)
        )

        return img_t, target

    def _mixup(
        self,
        img_t: torch.Tensor,
        target: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """MixUp с другим случайным изображением."""
        if len(self.image_paths) <= 1:
            return img_t, target

        other = random.randrange(
            len(self.image_paths)
        )

        img2, tgt2 = self._load_plain(
            other
        )

        lam = float(
            np.random.beta(
                8.0,
                8.0,
            )
        )

        mixed = (
            lam * img_t
            + (1.0 - lam) * img2
        )

        if target.shape[0] and tgt2.shape[0]:
            target = np.concatenate(
                [
                    target,
                    tgt2,
                ],
                axis=0,
            )
        elif tgt2.shape[0]:
            target = tgt2

        return mixed, target

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, np.ndarray]:

        # Mosaic.
        if (
            self.augment
            and self.mosaic_prob > 0.0
            and random.random() < self.mosaic_prob
        ):
            img_t, target = self._mosaic()

            if (
                self.mixup_prob > 0.0
                and random.random() < self.mixup_prob
            ):
                img_t, target = self._mixup(
                    img_t,
                    target,
                )

            return img_t, target

        # Обычный sample.
        image_path = self.image_paths[index]

        try:
            img = self._read_image(
                image_path
            )
        except Exception as exc:
            log.warning(
                "Unreadable image %s: %s",
                image_path,
                exc,
            )

            good_index = self._next_good_index(
                index
            )

            img = self._read_image(
                self.image_paths[good_index]
            )

            image_path = self.image_paths[
                good_index
            ]

        boxes_yolo = self.load_boxes(
            image_path
        )

        if self.augment:
            img, boxes_yolo = (
                self._augment_norm(
                    img,
                    boxes_yolo,
                )
            )

        img, _scale, pad_x, pad_y = letterbox(
            img,
            self.input_size,
        )

        img = np.ascontiguousarray(img)

        img_t = (
            torch.from_numpy(
                img.transpose(2, 0, 1)
            )
            .float()
            .div_(255.0)
        )

        eff_w = (
            img.shape[1]
            - 2 * pad_x
        )

        eff_h = (
            img.shape[0]
            - 2 * pad_y
        )

        target = self._build_target(
            boxes_yolo,
            eff_w,
            eff_h,
            pad_x,
            pad_y,
        )

        if (
            self.augment
            and self.mixup_prob > 0.0
            and random.random() < self.mixup_prob
        ):
            img_t, target = self._mixup(
                img_t,
                target,
            )

        return img_t, target

    def _next_good_index(
        self,
        index: int,
    ) -> int:
        """Ищет читаемое изображение без рекурсии."""
        import cv2

        n = len(self.image_paths)

        for step in range(1, n + 1):
            candidate = (
                index + step
            ) % n

            path = self.image_paths[
                candidate
            ]

            try:
                img = cv2.imread(
                    str(path),
                    cv2.IMREAD_COLOR,
                )

                if img is not None:
                    return candidate

            except Exception:
                continue

        raise RuntimeError(
            "No readable images in dataset"
        )


def dataloader_kwargs(
    workers: int,
) -> dict:
    """
    Возвращает безопасные kwargs для DataLoader.

    workers <= 0: префетча нет, воркеры не нужны.

    workers > 0: воркеры запускаются через spawn-контекст.

    Почему не fork (дефолт на Linux):
    DataLoader форкается, а в train.py к моменту его создания CUDA
    уже инициализирован (model.to(device)). Форк после CUDA-init —
    классический дедлок PyTorch (CPU/GPU висят на 0%). spawn-воркеры
    стартуют «чистыми» и не наследуют CUDA-состояние родителя.
    """
    if int(workers) <= 0:
        return {}

    import multiprocessing as mp

    return {
        "persistent_workers": True,
        "prefetch_factor": 2,
        "multiprocessing_context": mp.get_context(
            "spawn"
        ),
    }


class DetectionCollate:
    """
    Собирает батч из:
        (image, target)

    Возвращает:
        images
        batch_boxes
        batch_classes
        batch_mask
    """

    def __call__(
        self,
        items: list[
            tuple[torch.Tensor, np.ndarray]
        ],
    ):
        if not items:
            raise ValueError(
                "DetectionCollate received empty batch"
            )

        images = torch.stack(
            [it[0] for it in items]
        )

        max_boxes = max(
            len(it[1])
            for it in items
        )

        batch_boxes = torch.full(
            (
                len(items),
                max_boxes,
                4,
            ),
            -1.0,
            dtype=torch.float32,
        )

        batch_classes = torch.full(
            (
                len(items),
                max_boxes,
            ),
            -1,
            dtype=torch.long,
        )

        batch_mask = torch.zeros(
            (
                len(items),
                max_boxes,
            ),
            dtype=torch.float32,
        )

        for i, (_, target) in enumerate(
            items
        ):
            if target.shape[0] == 0:
                continue

            n = target.shape[0]

            batch_boxes[
                i,
                :n,
            ] = torch.from_numpy(
                target[:, :4]
            )

            batch_classes[
                i,
                :n,
            ] = torch.from_numpy(
                target[:, 4].astype(
                    np.int64
                )
            )

            batch_mask[
                i,
                :n,
            ] = 1.0

        return (
            images,
            batch_boxes,
            batch_classes,
            batch_mask,
        )


def build_datasets(
    cfg: Config,
    augment: bool = True,
):
    """Собирает train/val датасеты."""
    root = (
        Path(cfg.dataset["root"])
        .expanduser()
        .resolve()
    )

    data_yaml = _resolve_dir(
        root,
        cfg.dataset["data_yaml"],
    )

    data = read_data_yaml(
        data_yaml
    )

    if (
        not data
        and not cfg.dataset["classes"]
    ):
        raise ValueError(
            "No data.yaml found and no classes in config"
        )

    train_paths, val_paths = find_image_sets(
        root,
        data,
        cfg.dataset["images_dir"],
        cfg.dataset["labels_dir"],
    )

    if not train_paths:
        raise FileNotFoundError(
            f"No train images under {root}"
        )

    input_size = int(
        cfg.model["input_size"]
    )

    # Если val существует явно — используем его.
    if val_paths:
        train_ds = YoloDataset(
            cfg,
            data,
            train_paths,
            root,
            "train",
            input_size,
            augment=augment,
        )

        val_ds = YoloDataset(
            cfg,
            data,
            val_paths,
            root,
            "val",
            input_size,
            augment=False,
        )

    else:
        # Делим уже собранный список изображений.
        log.info(
            "No val split found, sampling 10%% "
            "of training images for validation"
        )

        rng = random.Random(
            int(cfg.train["seed"])
        )

        all_paths = sorted(
            set(train_paths)
        )

        n_val = max(
            1,
            int(len(all_paths) * 0.1),
        )

        n_val = min(
            n_val,
            max(1, len(all_paths) - 1),
        )

        val_paths_set = set(
            rng.sample(
                all_paths,
                n_val,
            )
        )

        real_train_paths = [
            p
            for p in all_paths
            if p not in val_paths_set
        ]

        if not real_train_paths:
            raise RuntimeError(
                "Training split became empty "
                "after validation split"
            )

        train_ds = YoloDataset(
            cfg,
            data,
            real_train_paths,
            root,
            "train",
            input_size,
            augment=augment,
        )

        val_ds = YoloDataset(
            cfg,
            data,
            sorted(val_paths_set),
            root,
            "val",
            input_size,
            augment=False,
        )

    train_abs = {
        str(p.resolve())
        for p in train_ds.image_paths
    }

    val_abs = {
        str(p.resolve())
        for p in val_ds.image_paths
    }

    overlap = train_abs & val_abs

    if overlap:
        sample = list(sorted(overlap))[:5]

        raise RuntimeError(
            "train/val overlap: "
            f"{len(overlap)} images present in both splits. "
            f"Examples: {sample}"
        )

    images_dir = str(
        cfg.dataset["images_dir"]
    )

    labels_dir = str(
        cfg.dataset["labels_dir"]
    )

    _log_missing_labels(
        train_ds.image_paths,
        root,
        images_dir,
        labels_dir,
        "train",
    )

    _log_missing_labels(
        val_ds.image_paths,
        root,
        images_dir,
        labels_dir,
        "val",
    )

    log.info(
        "Dataset ready: train=%d, val=%d",
        len(train_ds),
        len(val_ds),
    )

    return train_ds, val_ds