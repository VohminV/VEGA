from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.config import Config, load_config
from src.dataset import (
    DetectionCollate,
    build_datasets,
    dataloader_kwargs,
)
from src.losses import detector_loss
from src.metrics import evaluate
from src.model import build_model
from src.tiled_inference import infer_image, final_nms


def lr_schedule(
    epoch_idx: int,
    warmup_epochs: int,
    epochs: int,
) -> float:
    """
    Warmup + cosine decay.

    Возвращает множитель относительно базового LR.
    """
    if warmup_epochs > 0 and epoch_idx < warmup_epochs:
        return (epoch_idx + 1) / float(
            warmup_epochs
        )

    progress = (
        (epoch_idx - warmup_epochs)
        / float(
            max(
                epochs - warmup_epochs,
                1,
            )
        )
    )

    progress = min(
        max(
            progress,
            0.0,
        ),
        1.0,
    )

    return (
        0.01
        + 0.5
        * (1.0 - 0.01)
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )


def _cuda_sync(
    device: torch.device,
) -> None:
    """
    Синхронизация CUDA только для корректного timing.

    Не вызывается внутри каждой итерации обучения.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _format_seconds(
    seconds: float,
) -> str:
    seconds = float(seconds)

    if seconds < 60.0:
        return f"{seconds:.1f}s"

    minutes = int(
        seconds // 60
    )

    remain = (
        seconds
        - minutes * 60
    )

    return (
        f"{minutes}m {remain:.1f}s"
    )


def validate(
    model,
    val_ds,
    cfg: Config,
    device: torch.device,
    use_tiled: bool,
):
    """
    Валидация.

    Текущий inference API принимает
    одно изображение, поэтому здесь намеренно
    нет искусственного batch inference.

    Возвращает metrics.
    """
    model.eval()

    input_size = int(
        cfg.model["input_size"]
    )

    val_cfg = cfg.val

    conf_threshold = float(
        val_cfg["conf_threshold"]
    )

    iou_threshold = float(
        val_cfg["iou_threshold"]
    )

    so = cfg.small_object

    small_min_side_px = float(
        so["small_min_side_px"]
    )

    small_area_fraction = float(
        so["small_area_fraction"]
    )

    tile_size = int(
        val_cfg.get(
            "tile_size",
            input_size,
        )
    )

    overlap_ratio = float(
        val_cfg.get(
            "overlap_ratio",
            0.25,
        )
    )

    tile_batch_size = int(
        val_cfg.get(
            "tile_batch_size",
            4,
        )
    )

    class_aware_nms = bool(
        val_cfg.get(
            "class_aware_nms",
            False,
        )
    )

    preds_list = []
    gts_list = []
    shapes = []

    start = time.perf_counter()

    with torch.inference_mode():
        for index, path in enumerate(
            val_ds.image_paths,
            1,
        ):
            image, width, height, gt = (
                val_ds.load_original_sample(
                    path
                )
            )

            preds = infer_image(
                model=model,
                image=image,
                device=device,
                input_size=input_size,
                conf_threshold=conf_threshold,
                use_tiled=use_tiled,
                tile_size=tile_size,
                overlap_ratio=overlap_ratio,
                tile_batch_size=tile_batch_size,
            )

            if preds.shape[0]:
                preds = final_nms(
                    preds,
                    iou_threshold,
                    class_aware=class_aware_nms,
                )

            preds_list.append(
                preds
            )

            gts_list.append(
                gt
            )

            shapes.append(
                (
                    width,
                    height,
                )
            )

            # Редкий progress log.
            if index % 100 == 0:
                elapsed = (
                    time.perf_counter()
                    - start
                )

                rate = index / max(
                    elapsed,
                    1e-6,
                )

                print(
                    f"  val: "
                    f"{index}/{len(val_ds)} "
                    f"({rate:.1f} img/s)",
                    flush=True,
                )

    elapsed = (
        time.perf_counter()
        - start
    )

    metrics = evaluate(
        preds_list,
        gts_list,
        shapes,
        num_classes=int(
            model.num_classes
        ),
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        small_min_side_px=small_min_side_px,
        small_area_fraction=small_area_fraction,
    )

    metrics["_val_seconds"] = elapsed

    metrics["_val_images"] = len(
        val_ds.image_paths
    )

    return metrics


def _extract_metric(
    metrics: dict,
) -> float:
    """
    Извлекает основную validation metric.

    Приоритет:
        map50
        mAP50
        map
        mAP
    """
    for key in (
        "map50",
        "mAP50",
        "map",
        "mAP",
    ):
        if key not in metrics:
            continue

        try:
            value = float(
                metrics[key]
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if math.isfinite(value):
            return value

    return -float("inf")


def _format_validation_metrics(
    metrics: dict,
) -> str:
    """
    Формирует короткий validation log.
    """
    parts = []

    if "precision" in metrics:
        try:
            parts.append(
                f"P={float(metrics['precision']):.4f}"
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    if "recall" in metrics:
        try:
            parts.append(
                f"R={float(metrics['recall']):.4f}"
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    if "map50" in metrics:
        try:
            parts.append(
                f"mAP50={float(metrics['map50']):.4f}"
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    elif "mAP50" in metrics:
        try:
            parts.append(
                f"mAP50={float(metrics['mAP50']):.4f}"
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    if not parts:
        return "metrics=N/A"

    return " ".join(parts)


def train(
    config_path: str = "configs/default.yaml",
    resume: str | None = None,
    epochs_override: int | None = None,
):
    """
    Основной training loop.

    Оптимизации:
      - workers=0 сохраняется;
      - non_blocking CUDA transfers;
      - validation выполняется по interval;
      - train/val timing;
      - CUDA synchronization только вокруг timing;
      - checkpoint сохраняется каждый epoch.
    """
    cfg = load_config(
        config_path
    )

    if epochs_override is not None:
        if epochs_override <= 0:
            raise ValueError(
                "--epochs must be > 0"
            )

        cfg.train["epochs"] = int(
            epochs_override
        )

    epochs = int(
        cfg.train["epochs"]
    )

    seed = int(
        cfg.train["seed"]
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    device_name = str(
        cfg.train["device"]
    ).strip().lower()

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but is not available"
            )

        device = torch.device(
            "cuda"
        )

    else:
        device = torch.device(
            device_name
        )

    print(
        f"device: {device}",
        flush=True,
    )

    # ---------------------------------------------------------
    # DATASET
    # ---------------------------------------------------------

    train_ds, val_ds = build_datasets(
        cfg,
        augment=True,
    )

    print(
        f"train: {len(train_ds)} images",
        flush=True,
    )

    print(
        f"val:   {len(val_ds)} images",
        flush=True,
    )

    # ---------------------------------------------------------
    # TRAIN DISK CACHE
    # ---------------------------------------------------------

    if (
        train_ds._cache_root is not None
        and bool(
            cfg.dataset.get(
                "cache_prebuild",
                True,
            )
        )
    ):
        print(
            "building disk image cache...",
            flush=True,
        )

        built = train_ds.build_disk_cache(
            lambda message: print(
                message,
                flush=True,
            )
        )

        print(
            f"cache ready: "
            f"{built} new images",
            flush=True,
        )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    model = build_model(
        cfg
    )

    model.to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "model: "
        f"backbone={cfg.model['backbone']}, "
        f"strides={cfg.model['strides']}, "
        f"params={params / 1e6:.2f}M",
        flush=True,
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---------------------------------------------------------
    # DATALOADER
    # ---------------------------------------------------------

    workers = int(
        cfg.train["workers"]
    )

    if workers < 0:
        raise ValueError(
            "train.workers must be >= 0, "
            f"got {workers}"
        )

    loader_kwargs = dataloader_kwargs(
        workers
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(
            cfg.train["batch_size"]
        ),
        shuffle=True,
        num_workers=workers,
        drop_last=False,
        collate_fn=DetectionCollate(),
        pin_memory=(
            device.type == "cuda"
        ),
        **loader_kwargs,
    )

    # ---------------------------------------------------------
    # OPTIMIZER
    # ---------------------------------------------------------

    lr = float(
        cfg.train["lr"]
    )

    weight_decay = float(
        cfg.train["weight_decay"]
    )

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ---------------------------------------------------------
    # LOSS
    # ---------------------------------------------------------

    # ВАЖНО:
    #
    # detector_loss() принимает:
    #
    #   obj_iou
    #   box_loss
    #   weights
    #
    # YAML использует:
    #
    #   obj_iou_targets
    #   box_loss
    #   loss_weights
    #
    # Поэтому здесь выполняется явное
    # сопоставление YAML -> API losses.py.
    loss_kwargs = {
        "use_focal": bool(
            cfg.train.get(
                "use_focal",
                True,
            )
        ),
        "focal_alpha": float(
            cfg.train.get(
                "focal_alpha",
                0.75,
            )
        ),
        "focal_gamma": float(
            cfg.train.get(
                "focal_gamma",
                1.5,
            )
        ),
        "obj_pos_weight": float(
            cfg.train.get(
                "obj_pos_weight",
                0.75,
            )
        ),
        "obj_iou": bool(
            cfg.train.get(
                "obj_iou_targets",
                False,
            )
        ),
        "box_loss": str(
            cfg.train.get(
                "box_loss",
                "eiou",
            )
        ).strip().lower(),
        "weights": dict(
            cfg.train.get(
                "loss_weights",
                {
                    "obj": 1.0,
                    "box": 1.0,
                    "cls": 1.0,
                },
            )
        ),
    }

    print(
        "loss: "
        f"focal={loss_kwargs['use_focal']} "
        f"alpha={loss_kwargs['focal_alpha']:.3f} "
        f"gamma={loss_kwargs['focal_gamma']:.3f} "
        f"obj_pos_weight={loss_kwargs['obj_pos_weight']:.3f} "
        f"obj_iou={loss_kwargs['obj_iou']} "
        f"box={loss_kwargs['box_loss']} "
        f"weights={loss_kwargs['weights']}",
        flush=True,
    )

    # ---------------------------------------------------------
    # AMP
    # ---------------------------------------------------------

    use_amp = bool(
        cfg.train.get(
            "amp",
            True,
        )
    ) and device.type == "cuda"

    scaler = GradScaler(
        enabled=use_amp
    )

    # ---------------------------------------------------------
    # SCHEDULER
    # ---------------------------------------------------------

    warmup_epochs = int(
        cfg.train.get(
            "warmup_epochs",
            5,
        )
    )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch_idx: lr_schedule(
            epoch_idx,
            warmup_epochs,
            epochs,
        ),
    )

    # ---------------------------------------------------------
    # SETTINGS
    # ---------------------------------------------------------

    grad_clip = float(
        cfg.train.get(
            "grad_clip",
            0.0,
        )
    )

    patience = int(
        cfg.train.get(
            "patience",
            20,
        )
    )

    val_cfg = cfg.val

    val_interval = max(
        1,
        int(
            val_cfg.get(
                "interval",
                5,
            )
        ),
    )

    use_tiled = bool(
        val_cfg.get(
            "use_tiled_inference",
            False,
        )
    )

    save_dir = Path(
        cfg.train["save_dir"]
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # RESUME
    # ---------------------------------------------------------

    start_epoch = 0

    best_metric = -float(
        "inf"
    )

    bad_epochs = 0

    # ---------------------------------------------------------
    # NaN / Inf GUARD
    #
    # Счётчик подряд идущих батчей с non-finite loss.
    #
    # Аварийная остановка — единственный способ не затереть
    # last.pt NaN-весами и не потерять данные для recovery.
    # ---------------------------------------------------------

    non_finite_batches = 0

    max_non_finite_batches = int(
        cfg.train.get(
            "max_non_finite_batches",
            5,
        )
    )

    training_aborted = False

    if resume:
        resume_path = Path(
            resume
        )

        if not resume_path.exists():
            raise FileNotFoundError(
                "Resume checkpoint not found: "
                f"{resume_path}"
            )

        checkpoint = torch.load(
            resume_path,
            map_location=device,
        )

        if (
            isinstance(checkpoint, dict)
            and "model" in checkpoint
        ):
            model.load_state_dict(
                checkpoint["model"]
            )

        else:
            model.load_state_dict(
                checkpoint
            )

        # Чекпоинт может оказаться сломанным: прошлый взрыв
        # обучения мог записать NaN/Inf в веса. Продолжать
        # обучение с такого состояния бессмысленно.
        if not all(
            bool(
                torch.isfinite(
                    p.data
                ).all().item()
            )
            for p in model.parameters()
        ):
            raise ValueError(
                f"Resume checkpoint {resume_path} has "
                "non-finite weights (NaN/Inf). "
                "Не продолжай обучение с повреждённого "
                "чекпоинта — возьми более ранний "
                "last.pt / best.pt."
            )

        if (
            isinstance(checkpoint, dict)
            and "optimizer" in checkpoint
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        if (
            isinstance(checkpoint, dict)
            and "scheduler" in checkpoint
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler"]
            )

        if (
            isinstance(checkpoint, dict)
            and "scaler" in checkpoint
        ):
            scaler.load_state_dict(
                checkpoint["scaler"]
            )

        if (
            isinstance(checkpoint, dict)
            and "epoch" in checkpoint
        ):
            start_epoch = (
                int(
                    checkpoint["epoch"]
                )
                + 1
            )

        if (
            isinstance(checkpoint, dict)
            and "best_metric" in checkpoint
        ):
            best_metric = float(
                checkpoint["best_metric"]
            )

        if (
            isinstance(checkpoint, dict)
            and "bad_epochs" in checkpoint
        ):
            bad_epochs = int(
                checkpoint["bad_epochs"]
            )

        print(
            f"resumed from {resume_path}, "
            f"start_epoch={start_epoch + 1}",
            flush=True,
        )

    if start_epoch >= epochs:
        print(
            "resume checkpoint is already "
            f"at epoch {start_epoch}, "
            f"target epochs={epochs}. "
            "Nothing to train.",
            flush=True,
        )

        return model

    print(
        f"training {epochs} epochs, "
        f"amp={use_amp}, "
        f"workers={workers}, "
        f"val_interval={val_interval}",
        flush=True,
    )

    # ---------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------

    for epoch_idx in range(
        start_epoch,
        epochs,
    ):
        epoch_number = (
            epoch_idx + 1
        )

        model.train()

        train_start = (
            time.perf_counter()
        )

        running_total = 0.0
        running_obj = 0.0
        running_box = 0.0
        running_cls = 0.0

        num_batches = 0

        for (
            batch_idx,
            batch,
        ) in enumerate(
            train_loader,
            1,
        ):
            (
                images,
                boxes,
                class_ids,
                mask,
            ) = batch

            # -------------------------------------------------
            # HOST -> GPU
            # -------------------------------------------------

            images = images.to(
                device,
                non_blocking=True,
            )

            boxes = boxes.to(
                device,
                non_blocking=True,
            )

            class_ids = class_ids.to(
                device,
                non_blocking=True,
            )

            mask = mask.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # -------------------------------------------------
            # FORWARD + LOSS
            # -------------------------------------------------

            with autocast(
                enabled=use_amp
            ):
                losses = detector_loss(
                    model,
                    images,
                    boxes,
                    class_ids,
                    mask,
                    **loss_kwargs,
                )

            total_loss = losses[
                "total"
            ]

            # -------------------------------------------------
            # NON-FINITE GUARD
            #
            # NaN/Inf loss (AMP-переполнение, взрыв градиента,
            # битая геометрия батча) — пропускаем обновление
            # весов и статистику, чтобы NaN не протёк ни в
            # веса, ни в средние по эпохе.
            #
            # N подряд таких батчей = обучение сломано:
            # останавливаемся, не трогая last.pt на диске.
            # -------------------------------------------------

            if not bool(
                torch.isfinite(
                    total_loss
                ).item()
            ):
                non_finite_batches += 1

                print(
                    f"  WARN [epoch {epoch_number} "
                    f"batch {batch_idx}]: "
                    f"non-finite loss="
                    f"{total_loss.item()!r}; "
                    "skipping optimizer update",
                    flush=True,
                )

                if (
                    max_non_finite_batches > 0
                    and non_finite_batches
                    >= max_non_finite_batches
                ):
                    print(
                        "  FATAL: "
                        f"{non_finite_batches} consecutive "
                        "non-finite losses; aborting "
                        "training. last.pt on disk is "
                        "unaffected (valid weights)",
                        flush=True,
                    )

                    training_aborted = True

                    break

                continue

            non_finite_batches = 0

            # -------------------------------------------------
            # BACKWARD
            # -------------------------------------------------

            scaler.scale(
                total_loss
            ).backward()

            # -------------------------------------------------
            # GRADIENT CLIPPING
            # -------------------------------------------------

            if grad_clip > 0.0:
                scaler.unscale_(
                    optimizer
                )

                clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            # -------------------------------------------------
            # OPTIMIZER STEP
            # -------------------------------------------------

            scaler.step(
                optimizer
            )

            scaler.update()

            # -------------------------------------------------
            # STATISTICS
            # -------------------------------------------------

            running_total += float(
                losses["total"].detach()
            )

            running_obj += float(
                losses["objectness"].detach()
            )

            running_box += float(
                losses["box"].detach()
            )

            running_cls += float(
                losses["class"].detach()
            )

            num_batches += 1

        # -----------------------------------------------------
        # ABORT
        #
        # Выход без валидации, без scheduler.step() и — главное —
        # без перезаписи last.pt: на диске остаётся последний
        # валидный чекпоинт.
        # -----------------------------------------------------

        if training_aborted:
            print(
                "training aborted at epoch "
                f"{epoch_number} after "
                f"{non_finite_batches} consecutive "
                "non-finite losses",
                flush=True,
            )

            break

        # -----------------------------------------------------
        # TRAIN TIMING
        # -----------------------------------------------------

        _cuda_sync(
            device
        )

        train_seconds = (
            time.perf_counter()
            - train_start
        )

        if num_batches > 0:
            train_total = (
                running_total
                / num_batches
            )

            train_obj = (
                running_obj
                / num_batches
            )

            train_box = (
                running_box
                / num_batches
            )

            train_cls = (
                running_cls
                / num_batches
            )

        else:
            train_total = 0.0
            train_obj = 0.0
            train_box = 0.0
            train_cls = 0.0

        # -----------------------------------------------------
        # VALIDATION
        # -----------------------------------------------------

        do_validation = (
            epoch_number == 1
            or epoch_number % val_interval == 0
            or epoch_number == epochs
        )

        val_seconds = 0.0
        metrics = None
        improved = False

        if do_validation:
            val_start = (
                time.perf_counter()
            )

            metrics = validate(
                model,
                val_ds,
                cfg,
                device,
                use_tiled,
            )

            _cuda_sync(
                device
            )

            val_seconds = (
                time.perf_counter()
                - val_start
            )

            metric_value = (
                _extract_metric(
                    metrics
                )
            )

            improved = (
                metric_value
                > best_metric
            )

            if improved:
                best_metric = (
                    metric_value
                )

                bad_epochs = 0

            else:
                bad_epochs += 1

        # -----------------------------------------------------
        # SCHEDULER
        # -----------------------------------------------------

        scheduler.step()

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        # -----------------------------------------------------
        # LOG
        # -----------------------------------------------------

        total_seconds = (
            train_seconds
            + val_seconds
        )

        if metrics is not None:
            metric_text = (
                f" | val={val_seconds:.1f}s "
                f"{_format_validation_metrics(metrics)}"
            )

        else:
            metric_text = (
                " | val=SKIPPED"
            )

        print(
            f"epoch "
            f"{epoch_number}/{epochs} | "
            f"loss={train_total:.5f} "
            f"obj={train_obj:.5f} "
            f"box={train_box:.5f} "
            f"cls={train_cls:.5f} | "
            f"lr={current_lr:.7f} | "
            f"train="
            f"{_format_seconds(train_seconds)}"
            f"{metric_text} | "
            f"total="
            f"{_format_seconds(total_seconds)}",
            flush=True,
        )

        if improved:
            print(
                f"  NEW BEST: "
                f"metric={best_metric:.6f}",
                flush=True,
            )

        # -----------------------------------------------------
        # CHECKPOINT
        # -----------------------------------------------------

        checkpoint = {
            "epoch": epoch_idx,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_metric": best_metric,
            "bad_epochs": bad_epochs,
            "config": config_path,
        }

        last_path = (
            save_dir
            / "last.pt"
        )

        torch.save(
            checkpoint,
            last_path,
        )

        if improved:
            best_path = (
                save_dir
                / "best.pt"
            )

            torch.save(
                checkpoint,
                best_path,
            )

        # -----------------------------------------------------
        # EARLY STOPPING
        # -----------------------------------------------------

        if do_validation:
            if (
                patience > 0
                and bad_epochs >= patience
            ):
                print(
                    "early stopping: "
                    f"{bad_epochs} validation checks "
                    "without improvement",
                    flush=True,
                )

                break

    print(
        "training finished",
        flush=True,
    )

    print(
        f"best metric: "
        f"{best_metric:.6f}",
        flush=True,
    )

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Train custom object detector"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=(
            "configs/default.yaml"
        ),
        help=(
            "Path to training config"
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to checkpoint "
            "for resume"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Override number of epochs"
        ),
    )

    args = parser.parse_args()

    train(
        config_path=args.config,
        resume=args.resume,
        epochs_override=args.epochs,
    )