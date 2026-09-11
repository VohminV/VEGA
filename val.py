"""Валидация обученной модели: метрики и примеры пропущенных мелких объектов."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from src.boxes import iou_matrix
from src.config import load_config
from src.dataset import build_datasets
from src.metrics import evaluate, is_small_box, save_metrics
from src.model import build_model
from src.tiled_inference import final_nms, infer_image
from train import resolve_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("val")


def load_weights(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def run(cfg, weights_path: str) -> dict:
    device = resolve_device(str(cfg.train["device"]))
    model = build_model(cfg)
    state = load_weights(weights_path)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    _, val_ds = build_datasets(cfg, augment=False)
    input_size = int(cfg.model["input_size"])
    v = cfg.val
    conf = float(v["conf_threshold"])
    iou = float(v["iou_threshold"])
    tile_size = int(v["tile_size"])
    overlap = float(v["overlap_ratio"])
    so = cfg.small_object
    small_min_side_px = float(so["small_min_side_px"])
    small_area_fraction = float(so["small_area_fraction"])
    use_tiled = bool(v["use_tiled_inference"])
    tile_batch_size = int(v.get("tile_batch_size", 4))
    class_aware = bool(v.get("class_aware_nms", False))

    preds_list: list[np.ndarray] = []
    gts_list: list[np.ndarray] = []
    shapes: list[tuple[int, int]] = []
    missed: list[dict] = []

    for path in val_ds.image_paths:
        image, w, h, gt = val_ds.load_original_sample(path)
        preds = infer_image(
            model, image, device, input_size, conf,
            use_tiled=use_tiled, tile_size=tile_size, overlap_ratio=overlap,
            tile_batch_size=tile_batch_size,
        )
        if preds.shape[0]:
            preds = final_nms(preds, iou, class_aware=class_aware)
        preds_list.append(preds)
        gts_list.append(gt)
        shapes.append((w, h))
        if gt.shape[0]:
            gt_small = is_small_box(gt[:, :4], w, h, small_min_side_px, small_area_fraction)
            for gi in np.where(gt_small)[0]:
                hit = (
                    preds.shape[0] > 0
                    and np.any(
                        (iou_helper(preds[:, :4], gt[gi][None, :4]) >= iou)
                        & (preds[:, 5].astype(int) == int(gt[gi, 4]))
                    )
                )
                if not hit:
                    missed.append({"image": str(path), "box_xyxy": gt[gi][:4].tolist()})

    metrics = evaluate(
        preds_list, gts_list, shapes,
        num_classes=model.num_classes,
        iou_threshold=iou,
        conf_threshold=conf,
        small_min_side_px=small_min_side_px,
        small_area_fraction=small_area_fraction,
    )
    out_dir = Path("runs/val")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, out_dir)

    if missed:
        with open(out_dir / "missed_small.txt", "w", encoding="utf-8") as f:
            for m in missed:
                x1, y1, x2, y2 = m["box_xyxy"]
                f.write(f"{m['image']} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
        log.info("missed small objects: %d (saved to %s)", len(missed), out_dir / "missed_small.txt")
        _save_missed_examples(missed, out_dir)
    return metrics


def iou_helper(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return iou_matrix(a, b)[:, 0]


def _save_missed_examples(missed: list[dict], out_dir: Path) -> None:
    import cv2

    save = out_dir / "missed_examples"
    save.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(missed[:10]):
        img = cv2.imread(m["image"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in m["box_xyxy"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        h, w = img.shape[:2]
        cx, cy, r = (x1 + x2) // 2, (y1 + y2) // 2, max(6, (x2 - x1 + y2 - y1) // 4)
        cv2.circle(img, (max(r, min(w - r, cx)), max(r, min(h - r, cy))), r, (0, 165, 255), 2)
        cv2.imwrite(str(save / f"missed_{i:03d}.jpg"), img)
    log.info("saved up to 10 missed-small examples in %s", save)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate custom detector")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--weights", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    metrics = run(cfg, args.weights)
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()