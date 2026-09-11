"""Инференс: обычный и тиловый (сам, без SAHI)."""
from __future__ import annotations

import numpy as np
import torch

from .boxes import letterbox, nms


def _decode_to_pixels(model, img_t: torch.Tensor, device: torch.device) -> np.ndarray:
    """Прогоняет батч (1,3,S,S) через модель, возвращает (N,6) [x1,y1,x2,y2,conf,cls]."""
    with torch.no_grad():
        outputs = model(img_t.to(device))
        preds = model.decode(outputs)[0]
    return preds.detach().cpu().numpy()


def _map_to_orig(
    preds: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> np.ndarray:
    """Переносит пиксельные предсказания из input_size-пространства back to исходное."""
    out = preds[:, :4].astype(np.float64)
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / scale + offset_x
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / scale + offset_y
    return np.concatenate([out, preds[:, 4:6]], axis=1)


def infer_image(
    model,
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
    """Инференс одного изображения (RGB, любого размера).

    Возвращает (N,6): x1,y1,x2,y2,conf,cls в координатах исходного изображения,
    уже пропущенные через conf_threshold (без NMS здесь — NMS делает вызывающий).
    """
    H, W = image.shape[:2]
    if use_tiled and (H > tile_size or W > tile_size):
        return _tiled_predict(model, image, device, input_size, conf_threshold, tile_size, overlap_ratio, tile_batch_size)
    return _single_predict(model, image, device, input_size, conf_threshold)


def _single_predict(
    model,
    image: np.ndarray,
    device: torch.device,
    input_size: int,
    conf_threshold: float,
) -> np.ndarray:
    import cv2

    img, scale, pad_x, pad_y = letterbox(image, input_size)
    img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0)
    preds = _decode_to_pixels(model, img_t, device)
    if preds.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
    keep = preds[:, 4] >= conf_threshold
    preds = preds[keep]
    return _map_to_orig(preds, scale, pad_x, pad_y)


def _tile_positions(size: int, tile: int, overlap: float) -> list[int]:
    if size <= tile:
        return [0]
    step = max(1, int(tile * (1.0 - overlap)))
    pos = [0]
    while pos[-1] + tile < size:
        pos.append(min(pos[-1] + step, size - tile))
        if pos[-1] + step >= size - tile:
            break
    if pos[-1] < size - tile:
        pos.append(size - tile)
    return sorted(set(pos))


def _tiled_predict(
    model,
    image: np.ndarray,
    device: torch.device,
    input_size: int,
    conf_threshold: float,
    tile_size: int,
    overlap_ratio: float,
    tile_batch_size: int = 4,
) -> np.ndarray:
    """Разрезает изображение на тайлы с перекрытием и собирает предсказания.

    Тайлы прогоняются батчами по tile_batch_size — один forward на весь батч.
    """
    H, W = image.shape[:2]
    xs = _tile_positions(W, tile_size, overlap_ratio)
    ys = _tile_positions(H, tile_size, overlap_ratio)
    tiles: list[tuple[torch.Tensor, float, int, int, int, int]] = []
    for oy in ys:
        for ox in xs:
            tile = image[oy : oy + tile_size, ox : ox + tile_size]
            img, scale, pad_x, pad_y = letterbox(tile, input_size)
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().div_(255.0)
            tiles.append((img_t, scale, pad_x, pad_y, ox, oy))

    all_preds: list[np.ndarray] = []
    batch_cap = max(1, min(int(tile_batch_size), 128))
    for start in range(0, len(tiles), batch_cap):
        chunk = tiles[start : start + batch_cap]
        batch = torch.stack([t[0] for t in chunk]).to(device)
        with torch.no_grad():
            outputs = model(batch)
            per_image = model.decode(outputs)
        for k, (_, scale, pad_x, pad_y, ox, oy) in enumerate(chunk):
            preds = per_image[k].detach().cpu().numpy()
            if preds.shape[0] == 0:
                continue
            preds = preds[preds[:, 4] >= conf_threshold]
            if preds.shape[0]:
                all_preds.append(_map_to_orig(preds, scale, pad_x, pad_y, ox, oy))
    if not all_preds:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(all_preds, axis=0)


MAX_NMS_CANDIDATES = 4096


def final_nms(preds: np.ndarray, iou_threshold: float, class_aware: bool = False) -> np.ndarray:
    """Итоговый NMS по объединённым предсказаниям.

    class_aware=True — NMS выполняется отдельно для каждого класса.
    Ограничивает число кандидатов сверху по confidence — важно в ранние эпохи,
    когда модель выстреливает почти со всех ячеек, иначе NMS раздувается.
    """
    if preds.shape[0] == 0:
        return preds
    if preds.shape[0] > MAX_NMS_CANDIDATES:
        cut = np.argsort(-preds[:, 4])[:MAX_NMS_CANDIDATES]
        preds = preds[cut]
    keep = nms(
        preds[:, :4], preds[:, 4], iou_threshold,
        classes=(preds[:, 5].astype(int) if class_aware else None),
    )
    return preds[keep]