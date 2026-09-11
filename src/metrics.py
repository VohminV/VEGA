"""Метрики: precision, recall, mAP@0.5 и отдельные метрики для мелких объектов."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .boxes import iou_matrix


def is_small_box(
    box_xyxy: np.ndarray,
    img_w: int,
    img_h: int,
    small_min_side_px: float,
    small_area_fraction: float,
) -> np.ndarray:
    """True для боксов: min сторона <= small_min_side_px ИЛИ площадь <= доли."""
    w = box_xyxy[:, 2] - box_xyxy[:, 0]
    h = box_xyxy[:, 3] - box_xyxy[:, 1]
    min_side = np.minimum(w, h)
    frac = w * h / float(img_w * img_h)
    return (min_side <= small_min_side_px) | (frac <= small_area_fraction)


def _match(preds: np.ndarray, gts: np.ndarray, iou_threshold: float):
    """Жадное сопоставление предсказаний и GT. Возвращает (matched, gt_used)."""
    matched = np.zeros(preds.shape[0], dtype=bool)
    gt_used = np.zeros(gts.shape[0], dtype=bool)
    if preds.shape[0] and gts.shape[0]:
        ious = iou_matrix(preds[:, :4], gts[:, :4])
        order = np.argsort(-preds[:, 4])
        for pi in order:
            gi = int(np.argmax(ious[pi]))
            if ious[pi, gi] >= iou_threshold and not gt_used[gi] and int(preds[pi, 5]) == int(gts[gi, 4]):
                matched[pi] = True
                gt_used[gi] = True
    return matched, gt_used


def _pr_points(
    preds_list: list[np.ndarray],
    gts_list: list[np.ndarray],
    iou_threshold: float,
):
    """Собирает точки (score, tp) по одному классу без cross-image matching.

    Matching выполняется локально для каждого изображения через _match, поэтому
    предсказание одного кадра не может заматчиться с GT другого кадра. После этого
    (score, tp) по всем изображениям склеиваются в общий массив для PR-кривой.
    """
    scores: list[float] = []
    tps: list[bool] = []
    for img_preds, img_gts in zip(preds_list, gts_list):
        if img_preds.shape[0] == 0:
            continue
        matched, _ = _match(img_preds, img_gts, iou_threshold)
        order = np.argsort(-img_preds[:, 4])
        scores.extend(img_preds[order, 4].astype(float).tolist())
        tps.extend(matched[order].tolist())
    return np.asarray(scores, dtype=np.float32), np.asarray(tps, dtype=bool)


def _pr_curve_from_points(scores: np.ndarray, tps: np.ndarray, n_gt: int):
    """PR-кривая из массива точек (score, tp) при ранжировании по confidence."""
    if scores.size == 0:
        return np.array([1.0]), np.array([0.0])
    order = np.argsort(-scores)
    tps = tps[order]
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(~tps)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    recalls = cum_tp / n_gt if n_gt else np.zeros_like(cum_tp)
    return np.concatenate([[1.0], precisions]), np.concatenate([[0.0], recalls])


def _ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """AP со 101 точкой интерполяции."""
    if recalls.size == 0 or recalls[-1] == 0:
        return 0.0
    ps = precisions.copy()
    for i in range(len(ps) - 2, -1, -1):
        ps[i] = max(ps[i], ps[i + 1])
    grid = np.linspace(0, 1, 101)
    ap = 0.0
    for g in grid:
        idx = int(np.searchsorted(recalls, g, side="left")) - 1
        idx = max(idx, 0)
        ap += ps[idx]
    return ap / 101.0


def evaluate(
    preds_per_image: list[np.ndarray],
    gts_per_image: list[np.ndarray],
    img_shapes: list[tuple[int, int]],
    num_classes: int,
    iou_threshold: float,
    conf_threshold: float,
    small_min_side_px: float,
    small_area_fraction: float,
) -> dict:
    """preds_per_image[i] = (N,6) [x1,y1,x2,y2,conf,cls]. gts = (M,5) [x1,y1,x2,y2,cls]."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    small_tp = 0
    small_fp = 0
    small_fn = 0
    total_gt = 0

    ap_list = []
    classes_with_gt: list[bool] = []
    for c in range(num_classes):
        cat_preds: list[np.ndarray] = []
        cat_gts: list[np.ndarray] = []
        for preds, gts in zip(preds_per_image, gts_per_image):
            if preds.shape[0]:
                keep = (preds[:, 4] >= conf_threshold) & (preds[:, 5].astype(int) == c)
                cat_preds.append(preds[keep])
            else:
                cat_preds.append(np.zeros((0, 6), dtype=np.float32))
            if gts.shape[0]:
                cat_gts.append(gts[gts[:, 4].astype(int) == c])
            else:
                cat_gts.append(np.zeros((0, 5), dtype=np.float32))
        n_gt_class = sum(g.shape[0] for g in cat_gts)
        classes_with_gt.append(n_gt_class > 0)
        scores, tps = _pr_points(cat_preds, cat_gts, iou_threshold)
        pr, rc = _pr_curve_from_points(scores, tps, n_gt_class)
        ap_list.append(_ap(pr, rc))

    for preds, gts, (img_w, img_h) in zip(preds_per_image, gts_per_image, img_shapes):
        if preds.shape[0]:
            keep = (preds[:, 4] >= conf_threshold) & (preds[:, 5].astype(int) < num_classes)
            preds = preds[keep]
        if gts.shape[0]:
            valid = gts[:, 4].astype(int) < num_classes
            gts = gts[valid]

        n_gt = gts.shape[0]
        if n_gt == 0:
            total_fp += preds.shape[0]
            small_fp += int(np.sum(is_small_box(preds[:, :4], img_w, img_h, small_min_side_px, small_area_fraction)))
            continue
        total_gt += n_gt
        matched, gt_used = _match(preds, gts, iou_threshold)
        total_tp += int(matched.sum())
        total_fp += int((~matched).sum())
        total_fn += int((~gt_used).sum())

        gt_small = is_small_box(gts[:, :4], img_w, img_h, small_min_side_px, small_area_fraction)
        small_fn += int(np.sum(~gt_used & gt_small))
        pred_small = is_small_box(preds[:, :4], img_w, img_h, small_min_side_px, small_area_fraction)
        small_tp += int(np.sum(matched & pred_small))
        small_fp += int(np.sum(~matched & pred_small))

    small_denom = small_tp + small_fn
    valid_aps = [ap for ap, has_gt in zip(ap_list, classes_with_gt) if has_gt]
    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0
    return {
        "precision": float(total_tp / (total_tp + total_fp + 1e-9)),
        "recall": float(total_tp / (total_tp + total_fn + 1e-9)),
        "mAP@0.5": mAP,
        "small_precision": float(small_tp / (small_tp + small_fp + 1e-9)),
        "small_recall": float(small_tp / (small_denom + 1e-9)),
        "small_true_positives": int(small_tp),
        "small_gt_objects": int(small_denom),
        "gt_objects": total_gt,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def save_metrics(metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    lines = [
        f"precision: {metrics['precision']:.4f}",
        f"recall: {metrics['recall']:.4f}",
        f"mAP@0.5: {metrics['mAP@0.5']:.4f}",
        f"small_precision: {metrics['small_precision']:.4f}",
        f"small_recall: {metrics['small_recall']:.4f}",
        f"small_tp: {metrics['small_true_positives']}",
        f"small_gt_objects: {metrics['small_gt_objects']}",
        f"gt_objects: {metrics['gt_objects']}",
    ]
    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")