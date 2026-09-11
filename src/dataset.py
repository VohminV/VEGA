"""Чтение YOLO-датасета (images/, labels/, data.yaml) и аугментации."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .boxes import letterbox, parse_label_line
from .config import Config

log = logging.getLogger("dataset")

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_data_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_dir(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def find_image_sets(root: Path, data: dict, images_dir: str, labels_dir: str):
    """Определяет списки изображений для train/val из data.yaml.

    Поддерживает стандартную YOLO-схему (train/val в data.yaml, абсолютные или
    относительные пути) и простую схему root/images + root/labels с одинаковым
    набором файлов (тогда val отделяется случайным сэмплом).
    """
    train_img_dirs: list[Path] = []
    val_img_dirs: list[Path] = []

    for key in ("train", "val"):
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            dirs = [value]
        elif isinstance(value, list):
            dirs = [str(d) for d in value]
        else:
            continue
        target = train_img_dirs if key == "train" else val_img_dirs
        for d in dirs:
            target.append(_resolve_dir(root, d))

    if not train_img_dirs and not val_img_dirs:
        full = root / images_dir
        if full.exists():
            train_img_dirs.append(full)
            log.info("data.yaml has no train/val entries, using all images in %s", full)

    return train_img_dirs, val_img_dirs


def _labels_path_for(image_path: Path, root: Path, images_dir: str, labels_dir: str) -> Path:
    """Путь к лейблу по стандартному YOLO-соглашению."""
    img = image_path.resolve()
    base = (root / images_dir).resolve()
    if base == img.parent:
        rel = img.name
    else:
        try:
            rel = img.relative_to(base).as_posix()
        except ValueError:
            rel = img.name
    stem = Path(rel).with_suffix(".txt")
    return root / labels_dir / stem


def _log_missing_labels(image_paths: list[Path], root: Path, images_dir: str, labels_dir: str, split: str) -> int:
    """Считает и логирует изображения без labels (трактуются как фон)."""
    missing = sum(
        1 for p in image_paths
        if not _labels_path_for(p, root, images_dir, labels_dir).exists()
    )
    log.warning(
        "%s: %d images total, %d without labels (treated as background)",
        split.capitalize(), len(image_paths), missing,
    )
    return missing


class YoloDataset(Dataset):
    """Датасет из YOLO-разметки. Возвращает (image, boxes_xyxy_px, class_ids)."""

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
        self.root = root
        self.images_dir = cfg.dataset["images_dir"]
        self.labels_dir = cfg.dataset["labels_dir"]
        self.split = split
        self.input_size = input_size
        self.augment = augment
        self.num_classes = int(cfg.model["num_classes"])
        aug = cfg.augmentation
        self.flip_lr = float(aug.get("flip_lr", 0.0))
        self.flip_ud = float(aug.get("flip_ud", 0.0))
        self.bc = float(aug.get("brightness_contrast", 0.0))
        self.scale_crop = float(aug.get("scale_crop", 0.0))
        self.scale_min = float(aug.get("scale_min", 0.7))
        self.scale_max = float(aug.get("scale_max", 1.0))

        self.image_paths: list[Path] = []
        for d in image_dirs:
            if not d.exists():
                log.warning("Image dir does not exist: %s", d)
                continue
            for ext in SUPPORTED_EXT:
                self.image_paths.extend(sorted(d.glob(f"**/*{ext}")))
        self.image_paths.sort()
        if not self.image_paths:
            raise RuntimeError(f"No images found for split '{split}' in {[str(d) for d in image_dirs]}")

        self.empty_bg_logged: set[str] = set()
        self.labels_cache: dict[str, np.ndarray] = {}

        # Проверка соответствия data.yaml и config
        names = data.get("names")
        if isinstance(names, dict) and len(names) > 0:
            n_cls = max(int(max(names)) + 1, len(names))
            if self.num_classes != n_cls:
                log.warning(
                    "data.yaml declares %d classes but config sets %d; using config value",
                    n_cls, self.num_classes,
                )
            if self.num_classes == 1 and list(names)[0] != 0:
                log.warning("Single class in data.yaml has id %s, expected 0", list(names)[0])

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_boxes(self, image_path: Path) -> np.ndarray:
        """Читает .txt лейбл, фильтрует битые строки. Возвращает (N,5): cls,cx,cy,w,h."""
        label_path = _labels_path_for(image_path, self.root, self.images_dir, self.labels_dir)
        key = str(image_path)
        if key in self.labels_cache:
            return self.labels_cache[key]
        boxes: list[tuple[int, float, float, float, float]] = []
        if label_path.exists():
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parsed = parse_label_line(line, self.num_classes)
                        if parsed is None:
                            continue
                        boxes.append(parsed)
            except OSError as e:
                log.warning("Failed to read label %s: %s", label_path, e)
        else:
            if self.split == "val":
                if key not in self.empty_bg_logged:
                    self.empty_bg_logged.add(key)
                    log.info("No label for %s, treated as empty background", image_path)
        arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
        self.labels_cache[key] = arr
        return arr

    def _read_image(self, image_path: Path) -> np.ndarray:
        import cv2

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _augment_norm(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Аугментации в координатах исходного изображения.

        boxes: (N,5) cls,cx,cy,w,h нормализованные. Возвращает обновлённые.
        """
        h, w = img.shape[:2]
        flip_lr = random.random() < self.flip_lr
        flip_ud = random.random() < self.flip_ud
        if flip_lr:
            img = img[:, ::-1, :]
            if boxes.shape[0]:
                boxes[:, 1] = 1.0 - boxes[:, 1]
        if flip_ud:
            img = img[::-1, :, :]
            if boxes.shape[0]:
                boxes[:, 2] = 1.0 - boxes[:, 2]

        if self.bc > 0.0 and boxes.shape[0] >= 0:
            import cv2

            alpha = 1.0 + random.uniform(-self.bc, self.bc)
            beta = random.randint(-int(30 * self.bc), int(30 * self.bc))
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        if self.scale_crop > 0.0 and random.random() < self.scale_crop:
            img, boxes = self._random_scale_crop(img, boxes)

        if boxes.shape[0]:
            x1 = np.clip(boxes[:, 1] - boxes[:, 3] / 2.0, 0.0, 1.0)
            y1 = np.clip(boxes[:, 2] - boxes[:, 4] / 2.0, 0.0, 1.0)
            x2 = np.clip(boxes[:, 1] + boxes[:, 3] / 2.0, 0.0, 1.0)
            y2 = np.clip(boxes[:, 2] + boxes[:, 4] / 2.0, 0.0, 1.0)
            keep = (x2 - x1 > 0) & (y2 - y1 > 0)
            x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
            boxes = boxes[keep]
            if boxes.shape[0]:
                boxes[:, 1] = (x1 + x2) / 2.0
                boxes[:, 2] = (y1 + y2) / 2.0
                boxes[:, 3] = x2 - x1
                boxes[:, 4] = y2 - y1
        return img, boxes

    def _random_scale_crop(self, img: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Случайный crop+масштаб: вырезает область и растягивает обратно.

        Боксы заданы в нормализованных координатах (cx,cy,w,h). После crop
        центр сдвигается на (ox,oy)/размер_картинки и все величины делятся на s.
        """
        import cv2

        h, w = img.shape[:2]
        s = random.uniform(self.scale_min, self.scale_max)
        if s >= 1.0 - 1e-6:  # масштаб ~1 => пропускаем
            return img, boxes
        new_w = max(1, int(round(w * s)))
        new_h = max(1, int(round(h * s)))
        if new_w >= w or new_h >= h:
            return img, boxes
        ox = random.randint(0, w - new_w)
        oy = random.randint(0, h - new_h)
        crop = img[oy : oy + new_h, ox : ox + new_w]
        img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        if boxes.shape[0]:
            boxes[:, 1] = (boxes[:, 1] - ox / w) / s
            boxes[:, 2] = (boxes[:, 2] - oy / h) / s
            boxes[:, 3] = boxes[:, 3] / s
            boxes[:, 4] = boxes[:, 4] / s
        return img, boxes

    def load_original_sample(
        self, image_path: Path
    ) -> tuple[np.ndarray, int, int, np.ndarray]:
        """Оригинальное RGB-изображение и GT (N,5) [x1,y1,x2,y2,cls] в его пикселях."""
        img = self._read_image(image_path)
        h, w = img.shape[:2]
        boxes_yolo = self.load_boxes(image_path)
        gt = np.zeros((0, 5), dtype=np.float32)
        if boxes_yolo.shape[0]:
            x1 = (boxes_yolo[:, 1] - boxes_yolo[:, 3] / 2.0) * w
            y1 = (boxes_yolo[:, 2] - boxes_yolo[:, 4] / 2.0) * h
            x2 = (boxes_yolo[:, 1] + boxes_yolo[:, 3] / 2.0) * w
            y2 = (boxes_yolo[:, 2] + boxes_yolo[:, 4] / 2.0) * h
            keep = (x2 > x1) & (y2 > y1)
            if keep.any():
                gt = np.stack([x1, y1, x2, y2, boxes_yolo[:, 0]], axis=-1)[keep]
        return img, w, h, gt

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        image_path = self.image_paths[index]
        try:
            img = self._read_image(image_path)
        except Exception as e:  # битая картинка не роняет датасет
            log.warning("Skipping unreadable image %s: %s", image_path, e)
            neighbor = self._next_good_index(index)
            return self.__getitem__(neighbor)

        boxes_yolo = self.load_boxes(image_path).copy()

        if self.augment:
            img, boxes_yolo = self._augment_norm(img, boxes_yolo)

        img, scale, pad_x, pad_y = letterbox(img, self.input_size)
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().div_(255.0)

        target = np.empty((0, 5), dtype=np.float32)
        if boxes_yolo.shape[0]:
            cx = boxes_yolo[:, 1] * (img.shape[1] - 2 * pad_x) + pad_x
            cy = boxes_yolo[:, 2] * (img.shape[0] - 2 * pad_y) + pad_y
            w = boxes_yolo[:, 3] * (img.shape[1] - 2 * pad_x)
            h = boxes_yolo[:, 4] * (img.shape[0] - 2 * pad_y)
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            keep = (x2 > x1) & (y2 > y1)
            if keep.any():
                target = np.stack([x1, y1, x2, y2, boxes_yolo[:, 0]], axis=-1)[keep]

        return img_t, target, np.asarray([], dtype=np.int64)

    def _next_good_index(self, index: int) -> int:
        n = len(self.image_paths)
        for step in range(1, n + 1):
            candidate = (index + step) % n
            try:
                import cv2

                if cv2.imread(str(self.image_paths[candidate]), cv2.IMREAD_COLOR) is not None:
                    return candidate
            except Exception:
                continue
        raise RuntimeError("No readable images in dataset")


class DetectionCollate:
    """Собирает батч: паддит боксы до max в батче, маска указывает валидные."""

    def __call__(self, items: list[tuple[torch.Tensor, np.ndarray, np.ndarray]]):
        images = torch.stack([it[0] for it in items])
        max_boxes = max(len(it[1]) for it in items)
        batch_boxes = torch.full((len(items), max_boxes, 4), -1.0)
        batch_classes = torch.full((len(items), max_boxes), -1, dtype=torch.long)
        batch_mask = torch.zeros((len(items), max_boxes))
        for i, (_, target, _) in enumerate(items):
            if target.shape[0] > 0:
                batch_boxes[i, : target.shape[0]] = torch.from_numpy(target[:, :4])
                batch_classes[i, : target.shape[0]] = torch.from_numpy(target[:, 4].astype(np.int64))
                batch_mask[i, : target.shape[0]] = 1.0
        return images, batch_boxes, batch_classes, batch_mask


def build_datasets(cfg: Config, augment: bool = True):
    """Собирает train/val датасеты из cfg.dataset и data.yaml."""
    root = Path(cfg.dataset["root"]).expanduser()
    data_yaml = _resolve_dir(root, cfg.dataset["data_yaml"])
    data = read_data_yaml(data_yaml)
    if not data and not cfg.dataset["classes"]:
        raise ValueError("No data.yaml found and no classes in config")

    train_dirs, val_dirs = find_image_sets(root, data, cfg.dataset["images_dir"], cfg.dataset["labels_dir"])
    if not train_dirs:
        raise FileNotFoundError(f"No train images under {root}")

    input_size = int(cfg.model["input_size"])
    train_ds = YoloDataset(
        cfg, data, train_dirs, root, "train", input_size, augment=augment,
    )
    if val_dirs:
        val_ds = YoloDataset(cfg, data, val_dirs, root, "val", input_size, augment=False)
    else:
        log.info("No val split found, sampling 10%% of training images for validation")
        n_val = max(1, int(len(train_ds) * 0.1))
        rng = random.Random(int(cfg.train["seed"]))
        idx = set(rng.sample(range(len(train_ds)), n_val))
        val_paths = [p for i, p in enumerate(train_ds.image_paths) if i in idx]
        val_ds = YoloDataset(
            cfg, data, [Path(p).parent for p in val_paths] if val_paths else [],
            root, "val", input_size, augment=False,
        )
        if val_ds.image_paths:
            val_ds.image_paths = val_paths

    images_dir = cfg.dataset["images_dir"]
    labels_dir = cfg.dataset["labels_dir"]
    _log_missing_labels(train_ds.image_paths, root, images_dir, labels_dir, "train")
    _log_missing_labels(val_ds.image_paths, root, images_dir, labels_dir, "val")
    return train_ds, val_ds