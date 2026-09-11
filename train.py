"""Обучение собственного детектора на YOLO-датасете."""
from __future__ import annotations

import argparse
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import DetectionCollate, build_datasets
from src.losses import detector_loss
from src.metrics import evaluate
from src.model import build_model
from src.tiled_inference import final_nms, infer_image


def lr_schedule(epoch_idx: int, warmup_epochs: int, epochs: int) -> float:
    """Множитель learning rate: линейный warmup, затем cosine до 0.01."""
    if epoch_idx < warmup_epochs:
        return (epoch_idx + 1) / warmup_epochs
    progress = (epoch_idx - warmup_epochs) / max(epochs - warmup_epochs, 1)
    return 0.01 + 0.5 * (1 - 0.01) * (1 + math.cos(math.pi * progress))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


class DynLogger:
    """Пишет и в stdout, и в runs/train/log.txt."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message)
        self.file.write(message + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cuda_arch_supported() -> bool:
    """True, если в билде PyTorch есть ядро под текущую GPU (sm_XX)."""
    try:
        cap = torch.cuda.get_device_capability(0)
        arch = f"sm_{cap[0]}{cap[1]}"
        archs = getattr(torch.cuda, "get_arch_list", lambda: [])()
        return (not archs) or (arch in archs)
    except Exception:
        return True


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA недоступен, использую CPU")
        return torch.device("cpu")
    if name.startswith("cuda"):
        if not _cuda_arch_supported():
            cap = torch.cuda.get_device_capability(0)
            archs = torch.cuda.get_arch_list()
            log.warning(
                "GPU sm_%d%d не поддерживается этим билдом PyTorch (доступны: %s). "
                "Использую CPU. Для GPU установи совместимый билд (напр. pip install "
                "torch==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118)",
                cap[0], cap[1], ",".join(archs),
            )
            return torch.device("cpu")
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_stats(ds, num_classes: int) -> dict[str, int]:
    n_imgs = len(ds)
    n_boxes = 0
    per_class = [0] * num_classes
    for path in ds.image_paths:
        boxes = ds.load_boxes(path)
        n_boxes += len(boxes)
        for cls in boxes[:, 0].astype(int):
            per_class[cls] += 1
    return {"images": n_imgs, "boxes": n_boxes, "per_class": per_class}


def validate(
    model,
    val_ds,
    cfg,
    device: torch.device,
    use_tiled: bool,
) -> dict:
    """Валидация: обычный или тиловый инференс + метрики (в т.ч. по мелким объектам)."""
    input_size = int(cfg.model["input_size"])
    v = cfg.val
    conf = float(v["conf_threshold"])
    iou = float(v["iou_threshold"])
    tile_size = int(v["tile_size"])
    overlap = float(v["overlap_ratio"])
    tile_batch_size = int(v.get("tile_batch_size", 4))
    class_aware = bool(v.get("class_aware_nms", False))
    so = cfg.small_object
    model.eval()
    preds_list: list[np.ndarray] = []
    gts_list: list[np.ndarray] = []
    shapes: list[tuple[int, int]] = []
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
    return evaluate(
        preds_list, gts_list, shapes,
        num_classes=model.num_classes,
        iou_threshold=iou,
        conf_threshold=conf,
        small_min_side_px=float(so["small_min_side_px"]),
        small_area_fraction=float(so["small_area_fraction"]),
    )


def train(args) -> None:
    t0 = time.time()
    cfg = load_config(args.config)
    set_seed(int(cfg.train["seed"]))
    device = resolve_device(str(cfg.train["device"]))

    tl = DynLogger(Path(str(cfg.train["save_dir"])) / "log.txt")
    tl.log(f"config: {args.config}")
    tl.log(f"device: {device}")

    train_ds, val_ds = build_datasets(cfg, augment=True)
    train_stats = count_stats(train_ds, train_ds.num_classes)
    tl.log(
        f"train: {train_stats['images']} images, {train_stats['boxes']} boxes, "
        f"per class: {train_stats['per_class']}"
    )
    val_stats = count_stats(val_ds, val_ds.num_classes)
    tl.log(
        f"val:   {val_stats['images']} images, {val_stats['boxes']} boxes, "
        f"per class: {val_stats['per_class']}"
    )
    tl.log(f"classes: {cfg.dataset['classes'] or 'from data.yaml'}")

    model = build_model(cfg)
    tl.log(
        f"model: backbone={cfg.model['backbone']}, strides={cfg.model['strides']}, "
        f"params={model.box_param_count() / 1e6:.2f}M"
    )
    model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.train["workers"]),
        drop_last=False,
        collate_fn=DetectionCollate(),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.train["lr"]), weight_decay=float(cfg.train["weight_decay"])
    )
    use_focal = bool(cfg.train.get("use_focal", True))
    use_amp = bool(cfg.train["amp"]) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (TypeError, AttributeError):  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epochs = int(cfg.train["epochs"])
    use_scheduler = bool(cfg.train.get("use_scheduler", True))
    warmup_epochs = max(1, int(cfg.train.get("warmup_epochs", 5)))
    grad_clip = float(cfg.train.get("grad_clip", 10.0))

    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: lr_schedule(e, warmup_epochs, epochs))
        if use_scheduler
        else None
    )

    save_dir = Path(str(cfg.train["save_dir"]))
    save_dir.mkdir(parents=True, exist_ok=True)
    patience = int(cfg.train["patience"])
    best_map = 0.0
    best_epoch = 0

    tl.log(f"training {epochs} epochs, amp={use_amp}")
    for epoch in range(1, epochs + 1):
        model.train()
        ep_losses = {"total": 0.0, "objectness": 0.0, "box": 0.0, "class": 0.0}
        n_batches = 0
        for images, boxes, class_ids, mask in train_loader:
            images = images.to(device)
            boxes = boxes.to(device)
            class_ids = class_ids.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            try:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    losses = detector_loss(model, images, boxes, class_ids, mask, use_focal=use_focal)
            except (TypeError, AttributeError):  # torch < 2.3
                with torch.cuda.amp.autocast(enabled=use_amp):
                    losses = detector_loss(model, images, boxes, class_ids, mask, use_focal=use_focal)
            scaler.scale(losses["total"]).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            for k in ep_losses:
                ep_losses[k] += float(losses[k].detach())
            n_batches += 1

        for k in ep_losses:
            ep_losses[k] /= max(n_batches, 1)

        use_tiled = bool(cfg.val["use_tiled_inference"])
        metrics = validate(model, val_ds, cfg, device, use_tiled)
        cur_epoch = f"epoch {epoch:4d}/{epochs}"
        msg = (
            f"  {cur_epoch} loss={ep_losses['total']:.4f} obj={ep_losses['objectness']:.4f} "
            f"box={ep_losses['box']:.4f} cls={ep_losses['class']:.4f}"
        )
        if metrics["recall"] > 0 or metrics["precision"] > 0 or epoch == 1:
            msg += (
                f" | P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                f"mAP={metrics['mAP@0.5']:.3f} smallP={metrics['small_precision']:.3f} "
                f"smallR={metrics['small_recall']:.3f}"
            )
        tl.log(msg)

        state = {
            "model": model.state_dict(),
            "cfg": cfg.raw,
            "epoch": epoch,
            "num_classes": model.num_classes,
            "names": cfg.dataset["classes"],
        }
        torch.save(state, save_dir / "last.pt")
        mAP = metrics["mAP@0.5"]
        if epoch == 1 or mAP > best_map + 1e-9:
            best_map = mAP
            best_epoch = epoch
            torch.save(state, save_dir / "best.pt")
            tl.log(f"    saved best.pt (mAP={mAP:.4f})")

        stale = epoch - best_epoch
        if scheduler is not None:
            scheduler.step()
        if patience > 0 and stale >= patience:
            tl.log(f"early stopping after {patience} epochs without improvement (best epoch {best_epoch})")
            break

    tl.log(f"done in {time.time() - t0:.1f}s, best mAP={best_map:.4f} at epoch {best_epoch}")
    tl.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train custom detector")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()