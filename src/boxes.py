"""Работа с боксами: парсинг YOLO-лейблов, конвертации, IoU, NMS."""
from __future__ import annotations

import math

import numpy as np


def parse_label_line(
    line: str,
    num_classes: int,
) -> tuple[int, float, float, float, float] | None:
    """Парсит строку 'class_id cx cy w h'. Возвращает None для битых строк."""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(p) for p in parts[1:])
    except ValueError:
        return None
    if not math.isfinite(cx) or not math.isfinite(cy) or not math.isfinite(w) or not math.isfinite(h):
        return None
    if cls < 0 or cls >= num_classes:
        return None
    if w <= 0 or h <= 0:
        return None
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    x1 = max(0.0, min(x1, 1.0))
    y1 = max(0.0, min(y1, 1.0))
    x2 = max(0.0, min(x2, 1.0))
    y2 = max(0.0, min(y2, 1.0))
    if x2 <= x1 or y2 <= y1:
        return None
    return cls, (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1


def xywh_to_xyxy(boxes: np.ndarray, img_size: int) -> np.ndarray:
    """YOLO (cx,cy,w,h) в пикселях (x1,y1,x2,y2). boxes: (N,4)."""
    boxes = np.asarray(boxes, dtype=np.float32)
    x1 = (boxes[:, 0] - boxes[:, 2] / 2.0) * img_size
    y1 = (boxes[:, 1] - boxes[:, 3] / 2.0) * img_size
    x2 = (boxes[:, 0] + boxes[:, 2] / 2.0) * img_size
    y2 = (boxes[:, 1] + boxes[:, 3] / 2.0) * img_size
    return np.stack([x1, y1, x2, y2], axis=-1)


def xyxy_to_yolo(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Пиксельные (x1,y1,x2,y2) в YOLO (cx,cy,w,h) нормализованные. boxes: (N,4)."""
    boxes = np.asarray(boxes, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return np.stack([cx, cy, w, h], axis=-1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU между боксами a (N,4) и b (M,4) в формате xyxy. Возвращает (N,M)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter_area = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = a_area[:, None] + b_area[None, :] - inter_area
    union = np.maximum(union, 1e-9)
    return inter_area / union


def _nms_greedy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Жадный NMS для одного подмножества. Возвращает локальные индексы."""
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[i : i + 1], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return keep


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    classes: np.ndarray | None = None,
) -> list[int]:
    """Жадный NMS. boxes: (N,4) xyxy, scores: (N,). Возвращает индексы.

    При classes=None — class-agnostic NMS; при передаче classes: (N,) классы
    учитываются — боксы разных классов не подавляют друг друга.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.shape[0] == 0:
        return []
    if classes is None:
        return _nms_greedy(boxes, scores, iou_threshold)
    classes = np.asarray(classes)
    keep: list[int] = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        local = _nms_greedy(boxes[idx], scores[idx], iou_threshold)
        keep.extend(idx[local])
    return sorted(keep)


def letterbox(
    image: np.ndarray,
    target_size: int,
) -> tuple[np.ndarray, float, int, int]:
    """Rescale изображения с сохранением пропорций; pad серым цветом.

    Возвращает (изображение, scale, pad_x, pad_y) где scale — коэффициент
    масштабирования, pad_x/pad_y — левый/верхний отступ в пикселях.
    """
    import cv2

    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def yolo_to_abs(
    yolo_boxes: np.ndarray,
    img_size: int,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    """YOLO-боксы (нормализованные к input_size) -> пиксельные xyxy в оригинале.

    Вычитает padding и делит на scale, чтобы вернуться к исходному разрешению.
    """
    px = xywh_to_xyxy(yolo_boxes, img_size)
    px = (px - np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float32)) / scale
    return px


def area_of(boxes: np.ndarray) -> np.ndarray:
    """Площади боксов xyxy, (N,)."""
    boxes = np.asarray(boxes, dtype=np.float32)
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def box_area_fraction(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Доля площади изображения, занимаемая каждым боксом."""
    return area_of(boxes) / float(img_w * img_h)