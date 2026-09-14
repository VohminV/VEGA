"""Инференс детектора: обычный и tiled режим.

Поддерживает:
    - обычный inference целого изображения;
    - tiled inference с перекрытием;
    - преобразование координат обратно в исходное изображение;
    - финальный NMS;
    - batch inference для тайлов.

Формат предсказания:
    [x1, y1, x2, y2, confidence, class_id]

Все координаты после infer_image() находятся
в координатах исходного изображения.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .boxes import letterbox, nms


# ============================================================
# Constants
# ============================================================

MAX_NMS_CANDIDATES = 8192

MIN_TILE_SIZE = 64
MAX_TILE_BATCH = 128

EPS = 1e-9


# ============================================================
# Validation / utilities
# ============================================================


def _empty_predictions() -> np.ndarray:
    """Пустой массив предсказаний формата [N,6]."""
    return np.zeros(
        (0, 6),
        dtype=np.float32,
    )


def _validate_image(
    image: np.ndarray,
) -> tuple[int, int]:
    """Проверяет входное изображение."""
    if not isinstance(
        image,
        np.ndarray,
    ):
        raise TypeError(
            "image must be numpy.ndarray"
        )

    if image.ndim != 3:
        raise ValueError(
            "image must have shape [H,W,C], "
            f"got {image.shape}"
        )

    if image.shape[2] != 3:
        raise ValueError(
            "image must have 3 channels, "
            f"got {image.shape[2]}"
        )

    H, W = image.shape[:2]

    if H <= 0 or W <= 0:
        raise ValueError(
            f"invalid image size: {W}x{H}"
        )

    return H, W


def _sanitize_predictions(
    preds: np.ndarray,
) -> np.ndarray:
    """
    Удаляет NaN/Inf и явно некорректные боксы.

    Формат:
        [x1,y1,x2,y2,conf,cls]
    """
    if preds.size == 0:
        return _empty_predictions()

    preds = np.asarray(
        preds,
        dtype=np.float32,
    )

    if preds.ndim != 2 or preds.shape[1] != 6:
        raise ValueError(
            "predictions must have shape [N,6], "
            f"got {preds.shape}"
        )

    finite = np.isfinite(
        preds
    ).all(axis=1)

    preds = preds[finite]

    if preds.shape[0] == 0:
        return _empty_predictions()

    valid = (
        (preds[:, 2] > preds[:, 0])
        & (preds[:, 3] > preds[:, 1])
        & (preds[:, 4] >= 0.0)
        & (preds[:, 4] <= 1.0 + 1e-6)
    )

    preds = preds[valid]

    if preds.shape[0] == 0:
        return _empty_predictions()

    return np.ascontiguousarray(
        preds,
        dtype=np.float32,
    )


# ============================================================
# Model decode
# ============================================================


def _decode_to_pixels(
    model: Any,
    img_t: torch.Tensor,
    device: torch.device,
    conf_threshold: float,
) -> np.ndarray:
    """
    Прогоняет батч [1,3,S,S] через модель.

    Возвращает:
        [N,6]
        [x1,y1,x2,y2,conf,cls]

    Координаты находятся в пространстве input_size.
    """
    if img_t.ndim != 4:
        raise ValueError(
            "img_t must be [B,C,H,W]"
        )

    if img_t.shape[0] != 1:
        raise ValueError(
            "_decode_to_pixels expects B=1"
        )

    with torch.inference_mode():
        outputs = model(
            img_t.to(
                device,
                non_blocking=True,
            )
        )

        try:
            preds = model.decode(
                outputs,
                conf_threshold=conf_threshold,
            )[0]
        except TypeError:
            # Совместимость с старым decoder,
            # если он ещё не принимает threshold.
            preds = model.decode(
                outputs
            )[0]

            if preds.shape[0]:
                preds = preds[
                    preds[:, 4]
                    >= conf_threshold
                ]

    if preds.numel() == 0:
        return _empty_predictions()

    return _sanitize_predictions(
        preds.detach()
        .cpu()
        .numpy()
    )


# ============================================================
# Coordinate mapping
# ============================================================


def _map_to_orig(
    preds: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    offset_x: int = 0,
    offset_y: int = 0,
    orig_w: int | None = None,
    orig_h: int | None = None,
) -> np.ndarray:
    """
    Переносит предсказания из letterbox/input пространства
    обратно в исходное изображение.

    Для tiled режима offset_x/offset_y — положение tile
    в исходном изображении.
    """
    preds = _sanitize_predictions(
        preds
    )

    if preds.shape[0] == 0:
        return _empty_predictions()

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"scale must be positive finite, got {scale}"
        )

    out = preds.copy()

    out[:, 0] = (
        out[:, 0] - float(pad_x)
    ) / float(scale) + float(
        offset_x
    )

    out[:, 2] = (
        out[:, 2] - float(pad_x)
    ) / float(scale) + float(
        offset_x
    )

    out[:, 1] = (
        out[:, 1] - float(pad_y)
    ) / float(scale) + float(
        offset_y
    )

    out[:, 3] = (
        out[:, 3] - float(pad_y)
    ) / float(scale) + float(
        offset_y
    )

    if orig_w is not None:
        out[:, [0, 2]] = np.clip(
            out[:, [0, 2]],
            0.0,
            float(orig_w),
        )

    if orig_h is not None:
        out[:, [1, 3]] = np.clip(
            out[:, [1, 3]],
            0.0,
            float(orig_h),
        )

    return _sanitize_predictions(
        out
    )


# ============================================================
# Public inference API
# ============================================================


def infer_image(
    model: Any,
    image: np.ndarray,
    device: torch.device,
    input_size: int,
    conf_threshold: float,
    *,
    use_tiled: bool = False,
    tile_size: int = 1024,
    overlap_ratio: float = 0.25,
    tile_batch_size: int = 4,
) -> np.ndarray:
    """
    Инференс одного RGB-изображения.

    Возвращает:
        [N,6]
        x1,y1,x2,y2,conf,cls

    Координаты — в исходном изображении.

    NMS здесь НЕ выполняется.
    Для этого используется final_nms().
    """
    H, W = _validate_image(
        image
    )

    if input_size <= 0:
        raise ValueError(
            f"input_size must be > 0, got {input_size}"
        )

    if not (
        0.0
        <= float(conf_threshold)
        <= 1.0
    ):
        raise ValueError(
            "conf_threshold must be in [0,1], "
            f"got {conf_threshold}"
        )

    if use_tiled and (
        H > tile_size
        or W > tile_size
    ):
        return _tiled_predict(
            model=model,
            image=image,
            device=device,
            input_size=input_size,
            conf_threshold=conf_threshold,
            tile_size=tile_size,
            overlap_ratio=overlap_ratio,
            tile_batch_size=tile_batch_size,
        )

    return _single_predict(
        model=model,
        image=image,
        device=device,
        input_size=input_size,
        conf_threshold=conf_threshold,
    )


# ============================================================
# Single image
# ============================================================


def _single_predict(
    model: Any,
    image: np.ndarray,
    device: torch.device,
    input_size: int,
    conf_threshold: float,
) -> np.ndarray:
    """Инференс целого изображения."""
    H, W = image.shape[:2]

    img, scale, pad_x, pad_y = letterbox(
        image,
        input_size,
    )

    # Важно:
    # letterbox должен вернуть HWC RGB/BGR numpy array.
    img = np.ascontiguousarray(
        img
    )

    img_t = (
        torch.from_numpy(
            img.transpose(
                2,
                0,
                1,
            )
        )
        .float()
        .div_(255.0)
        .unsqueeze(0)
    )

    preds = _decode_to_pixels(
        model,
        img_t,
        device,
        conf_threshold,
    )

    if preds.shape[0] == 0:
        return _empty_predictions()

    return _map_to_orig(
        preds,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        offset_x=0,
        offset_y=0,
        orig_w=W,
        orig_h=H,
    )


# ============================================================
# Tile positions
# ============================================================


def _tile_positions(
    size: int,
    tile: int,
    overlap: float,
) -> list[int]:
    """
    Возвращает координаты начала tile по одной оси.

    Гарантирует:
        - первый tile начинается с 0;
        - последний tile заканчивается ровно на size;
        - нет отрицательных координат.
    """
    if size <= 0:
        raise ValueError(
            f"size must be > 0, got {size}"
        )

    if tile <= 0:
        raise ValueError(
            f"tile must be > 0, got {tile}"
        )

    if not (
        0.0
        <= overlap
        < 1.0
    ):
        raise ValueError(
            "overlap must be in [0,1), "
            f"got {overlap}"
        )

    if size <= tile:
        return [0]

    step = max(
        1,
        int(
            round(
                tile
                * (
                    1.0
                    - overlap
                )
            )
        ),
    )

    positions = [0]

    while True:
        current = positions[-1]

        if current + tile >= size:
            break

        next_pos = current + step

        if next_pos + tile >= size:
            next_pos = size - tile

        if next_pos <= current:
            break

        positions.append(
            next_pos
        )

        if next_pos + tile >= size:
            break

    return sorted(
        set(
            positions
        )
    )


# ============================================================
# Tiled inference
# ============================================================


def _tiled_predict(
    model: Any,
    image: np.ndarray,
    device: torch.device,
    input_size: int,
    conf_threshold: float,
    tile_size: int,
    overlap_ratio: float,
    tile_batch_size: int = 4,
) -> np.ndarray:
    """
    Tiled inference.

    Важный момент:

        tile_size != input_size

    означает, что каждый tile всё равно приводится
    к input_size через letterbox.

    Поэтому tile inference улучшает small-object detection
    только если tile содержит меньшую сцену относительно
    той же входной сетки.

    Например:

        исходное 1920x1080
        обычный inference -> вся сцена -> 640x640

        tile 640x640
        -> каждый tile -> 640x640

    Здесь объект получает гораздо больше пикселей
    относительно модели.
    """
    H, W = _validate_image(
        image
    )

    if tile_size < MIN_TILE_SIZE:
        raise ValueError(
            f"tile_size must be >= {MIN_TILE_SIZE}, "
            f"got {tile_size}"
        )

    if input_size <= 0:
        raise ValueError(
            f"input_size must be > 0, got {input_size}"
        )

    if not (
        0.0
        <= float(overlap_ratio)
        < 1.0
    ):
        raise ValueError(
            "overlap_ratio must be in [0,1), "
            f"got {overlap_ratio}"
        )

    batch_cap = max(
        1,
        min(
            int(tile_batch_size),
            MAX_TILE_BATCH,
        ),
    )

    xs = _tile_positions(
        W,
        tile_size,
        overlap_ratio,
    )

    ys = _tile_positions(
        H,
        tile_size,
        overlap_ratio,
    )

    # --------------------------------------------------------
    # Подготовка tiles.
    # --------------------------------------------------------

    tiles: list[
        tuple[
            torch.Tensor,
            float,
            int,
            int,
            int,
            int,
        ]
    ] = []

    for oy in ys:
        for ox in xs:
            tile = image[
                oy : oy + tile_size,
                ox : ox + tile_size,
            ]

            tile_h, tile_w = (
                tile.shape[:2]
            )

            if (
                tile_h <= 0
                or tile_w <= 0
            ):
                continue

            img, scale, pad_x, pad_y = (
                letterbox(
                    tile,
                    input_size,
                )
            )

            img = np.ascontiguousarray(
                img
            )

            img_t = (
                torch.from_numpy(
                    img.transpose(
                        2,
                        0,
                        1,
                    )
                )
                .float()
                .div_(255.0)
            )

            tiles.append(
                (
                    img_t,
                    float(scale),
                    int(pad_x),
                    int(pad_y),
                    int(ox),
                    int(oy),
                )
            )

    if not tiles:
        return _empty_predictions()

    # --------------------------------------------------------
    # Batched inference.
    # --------------------------------------------------------

    all_preds: list[
        np.ndarray
    ] = []

    for start in range(
        0,
        len(tiles),
        batch_cap,
    ):
        chunk = tiles[
            start : start + batch_cap
        ]

        batch = torch.stack(
            [
                item[0]
                for item in chunk
            ],
            dim=0,
        )

        batch = batch.to(
            device,
            non_blocking=True,
        )

        with torch.inference_mode():
            outputs = model(
                batch
            )

            try:
                per_image = model.decode(
                    outputs,
                    conf_threshold=conf_threshold,
                )
            except TypeError:
                per_image = model.decode(
                    outputs
                )

        for k, (
            _,
            scale,
            pad_x,
            pad_y,
            ox,
            oy,
        ) in enumerate(chunk):
            preds = (
                per_image[k]
                .detach()
                .cpu()
                .numpy()
            )

            preds = _sanitize_predictions(
                preds
            )

            if preds.shape[0] == 0:
                continue

            # Совместимость со старым decoder.
            preds = preds[
                preds[:, 4]
                >= conf_threshold
            ]

            if preds.shape[0] == 0:
                continue

            mapped = _map_to_orig(
                preds,
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
                offset_x=ox,
                offset_y=oy,
                orig_w=W,
                orig_h=H,
            )

            if mapped.shape[0]:
                all_preds.append(
                    mapped
                )

    if not all_preds:
        return _empty_predictions()

    return np.concatenate(
        all_preds,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Final NMS
# ============================================================


def final_nms(
    preds: np.ndarray,
    iou_threshold: float,
    class_aware: bool = False,
) -> np.ndarray:
    """
    Итоговый NMS по объединённым предсказаниям.

    Вход:
        [N,6]
        x1,y1,x2,y2,conf,cls

    Выход:
        [M,6]

    class_aware=False:
        один общий NMS.

    class_aware=True:
        отдельный NMS для каждого класса.
    """
    preds = _sanitize_predictions(
        preds
    )

    if preds.shape[0] == 0:
        return _empty_predictions()

    if not (
        0.0
        <= float(iou_threshold)
        <= 1.0
    ):
        raise ValueError(
            "iou_threshold must be in [0,1], "
            f"got {iou_threshold}"
        )

    # --------------------------------------------------------
    # Ограничение количества кандидатов.
    #
    # ВАЖНО:
    # делаем top-k только ПОСЛЕ conf filtering,
    # который уже должен быть сделан вызывающим кодом
    # или infer_image().
    # --------------------------------------------------------

    if (
        preds.shape[0]
        > MAX_NMS_CANDIDATES
    ):
        order = np.argsort(
            -preds[:, 4],
            kind="stable",
        )

        preds = preds[
            order[
                :MAX_NMS_CANDIDATES
            ]
        ]

    classes = None

    if class_aware:
        classes = (
            preds[:, 5]
            .astype(
                np.int64,
                copy=False,
            )
        )

    keep = nms(
        preds[:, :4],
        preds[:, 4],
        float(iou_threshold),
        classes=classes,
    )

    if keep is None:
        return _empty_predictions()

    keep = np.asarray(
        keep,
        dtype=np.int64,
    )

    if keep.size == 0:
        return _empty_predictions()

    return np.ascontiguousarray(
        preds[keep],
        dtype=np.float32,
    )