"""Инференс: изображение, папка, видео + тиловый режим."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import load_config
from src.model import build_model
from src.tiled_inference import final_nms, infer_image
from train import resolve_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("predict")

SUPPORTED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def make_names(cfg) -> list[str]:
    names = cfg.dataset["classes"]
    if not names:
        root = Path(str(cfg.dataset["root"])).expanduser()
        py = root / str(cfg.dataset["data_yaml"])
        if py.exists():
            import yaml

            data = yaml.safe_load(py.read_text(encoding="utf-8")) or {}
            found = data.get("names")
            if isinstance(found, dict):
                names = [found[str(i)] for i in sorted(int(k) for k in found)]
            elif isinstance(found, list):
                names = [str(n) for n in found]
    num_classes = int(cfg.model["num_classes"])
    while len(names) < num_classes:
        names.append(str(len(names)))
    return names[:num_classes]


def build_net(cfg, weights: str):
    device = resolve_device(str(cfg.train["device"]))
    model = build_model(cfg)
    try:
        state = torch.load(weights, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model, device


def override_predict_cfg(cfg, conf: float | None, iou: float | None) -> None:
    """Точечно переопределяет predict-настройки из CLI (не трогая yaml)."""
    if conf is not None:
        cfg.predict["conf_threshold"] = float(conf)
        log.info("config override: predict.conf_threshold = %s", cfg.predict["conf_threshold"])
    if iou is not None:
        cfg.predict["iou_threshold"] = float(iou)
        log.info("config override: predict.iou_threshold = %s", cfg.predict["iou_threshold"])


def draw(image_bgr: np.ndarray, preds: np.ndarray, names: list[str], save_conf: bool) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2, conf, cls_id in preds:
        x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        cv2.rectangle(out, (max(0, x1i), max(0, y1i)), (min(w, x2i), min(h, y2i)), (0, 255, 0), 2)
        label = names[int(cls_id)] if int(cls_id) < len(names) else str(int(cls_id))
        if save_conf:
            label = f"{label} {conf:.2f}"
        cv2.putText(out, label, (max(0, x1i), max(12, y1i - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def save_txt(output_txt: Path, preds: np.ndarray, img_w: int, img_h: int, save_conf: bool) -> None:
    lines = []
    for x1, y1, x2, y2, conf, cls_id in preds:
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        tail = f" {conf:.6f}" if save_conf else ""
        lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}{tail}")
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(lines), encoding="utf-8")


def process_one(
    model,
    device: torch.device,
    image_bgr: np.ndarray,
    cfg,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    p = cfg.predict
    conf = float(p["conf_threshold"])
    iou = float(p["iou_threshold"])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    preds = infer_image(
        model, image_rgb, device, int(cfg.model["input_size"]), conf,
        use_tiled=bool(p["use_tiled_inference"]),
        tile_size=int(p["tile_size"]),
        overlap_ratio=float(p["overlap_ratio"]),
        tile_batch_size=int(p.get("tile_batch_size", 4)),
    )
    if preds.shape[0]:
        preds = final_nms(preds, iou, class_aware=bool(p.get("class_aware_nms", False)))
    return image_bgr, preds


def handle_image(
    model, device, cfg, names, img_path: Path, out_images: Path, out_labels: Path,
) -> None:
    image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        log.warning("skipping unreadable image %s", img_path)
        return
    _, preds = process_one(model, device, image_bgr, cfg, names)
    h, w = image_bgr.shape[:2]
    if bool(cfg.predict["save_txt"]):
        label = out_labels / f"{img_path.stem}.txt"
        save_txt(label, preds, w, h, bool(cfg.predict["save_conf"]))
    if bool(cfg.predict["save_img"]):
        annotated = draw(image_bgr, preds, names, bool(cfg.predict["save_conf"]))
        cv2.imwrite(str(out_images / f"{img_path.stem}.jpg"), annotated)
    log.info("%s: %d objects", img_path.name, len(preds))


def predict_image(cfg, weights: str, source: str) -> None:
    model, device = build_net(cfg, weights)
    names = make_names(cfg)
    out_dir = Path(str(cfg.predict["output_dir"]))
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    handle_image(model, device, cfg, names, Path(source), out_images, out_labels)

def predict_folder(cfg, weights: str, source: str) -> None:
    model, device = build_net(cfg, weights)
    names = make_names(cfg)
    out_dir = Path(str(cfg.predict["output_dir"]))
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    src = Path(source)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in SUPPORTED_IMG_EXT)
    if not files:
        log.warning("no images found in %s", src)
        return
    for img_path in files:
        handle_image(model, device, cfg, names, img_path, out_images, out_labels)
    log.info("processed %d images", len(files))


def predict_video(cfg, weights: str, source: str, log_dets: bool = False) -> None:
    model, device = build_net(cfg, weights)
    names = make_names(cfg)
    out_dir = Path(str(cfg.predict["output_dir"]))
    name = Path(source).stem
    writer = None
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("cannot open video %s", source)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, preds = process_one(model, device, frame, cfg, names)
        if log_dets:
            log.info("frame %06d: %d detections", frame_i, len(preds))
        h, w = frame.shape[:2]
        if len(preds) and bool(cfg.predict["save_txt"]):
            label = out_dir / "video_labels" / f"{name}_{frame_i:06d}.txt"
            save_txt(label, preds, w, h, bool(cfg.predict["save_conf"]))
        if bool(cfg.predict["save_img"]):
            if writer is None:
                out_video = out_dir / f"{name}_annotated.avi"
                writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
            annotated = draw(frame, preds, names, bool(cfg.predict["save_conf"]))
            writer.write(annotated)
        frame_i += 1
    cap.release()
    if writer is not None:
        writer.release()
        log.info("video saved")
    log.info("processed %d frames", frame_i)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with custom detector")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--source_type", type=str, default=None, choices=["image", "folder", "video"])
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold override (по умолчанию из config)")
    parser.add_argument("--iou", type=float, default=None,
                        help="NMS IoU threshold override (по умолчанию из config)")
    parser.add_argument("--log-detections", action="store_true",
                        help="Писать число детекций на каждый кадр видео")
    args = parser.parse_args()
    cfg = load_config(args.config)
    override_predict_cfg(cfg, args.conf, args.iou)
    s_type = args.source_type or str(cfg.predict["source_type"])
    if s_type == "image":
        predict_image(cfg, args.weights, args.source)
    elif s_type == "folder":
        predict_folder(cfg, args.weights, args.source)
    elif s_type == "video":
        predict_video(cfg, args.weights, args.source, log_dets=args.log_detections)
    else:
        raise ValueError(f"unknown source_type: {s_type}")


if __name__ == "__main__":
    main()