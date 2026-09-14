"""Загрузка и валидация конфигурации детектора."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """
    Typed-ish wrapper над YAML-конфигурацией.

    Конфигурация намеренно хранится как обычный dict,
    чтобы остальные модули проекта могли получать доступ
    через cfg.dataset / cfg.model / cfg.train и т.д.
    """

    REQUIRED_SECTIONS = (
        "dataset",
        "model",
        "train",
        "augmentation",
        "val",
        "predict",
        "small_object",
    )

    def __init__(self, raw: dict[str, Any]):
        if not isinstance(raw, dict):
            raise TypeError(
                "Configuration root must be a mapping/dict."
            )

        self.raw = raw

        self.dataset = raw.get(
            "dataset",
            {},
        )

        self.model = raw.get(
            "model",
            {},
        )

        self.train = raw.get(
            "train",
            {},
        )

        self.augmentation = raw.get(
            "augmentation",
            {},
        )

        self.val = raw.get(
            "val",
            {},
        )

        self.predict = raw.get(
            "predict",
            {},
        )

        self.small_object = raw.get(
            "small_object",
            {},
        )

        self._validate()

    def _validate(self) -> None:
        self._validate_sections()
        self._validate_dataset()
        self._validate_model()
        self._validate_train()
        self._validate_augmentation()
        self._validate_val()
        self._validate_predict()
        self._validate_small_object()

    # ---------------------------------------------------------
    # SECTIONS
    # ---------------------------------------------------------

    def _validate_sections(self) -> None:
        missing = [
            name
            for name in self.REQUIRED_SECTIONS
            if name not in self.raw
        ]

        if missing:
            raise ValueError(
                "Missing configuration sections: "
                + ", ".join(missing)
            )

    # ---------------------------------------------------------
    # DATASET
    # ---------------------------------------------------------

    def _validate_dataset(self) -> None:
        cfg = self.dataset

        if not isinstance(cfg, dict):
            raise TypeError(
                "dataset must be a mapping."
            )

        root = cfg.get("root")

        if not isinstance(root, str) or not root.strip():
            raise ValueError(
                "dataset.root must be a non-empty string."
            )

        images_dir = cfg.get("images_dir")

        if not isinstance(images_dir, str) or not images_dir.strip():
            raise ValueError(
                "dataset.images_dir must be a non-empty string."
            )

        labels_dir = cfg.get("labels_dir")

        if not isinstance(labels_dir, str) or not labels_dir.strip():
            raise ValueError(
                "dataset.labels_dir must be a non-empty string."
            )

        classes = cfg.get("classes")

        if not isinstance(classes, list):
            raise ValueError(
                "dataset.classes must be a list."
            )

        if not classes:
            raise ValueError(
                "dataset.classes must contain at least one class."
            )

        if not all(
            isinstance(name, str) and name.strip()
            for name in classes
        ):
            raise ValueError(
                "Every dataset class name must be a non-empty string."
            )

        if len(set(classes)) != len(classes):
            raise ValueError(
                "dataset.classes must not contain duplicates."
            )

        # RAM cache.
        cache_images = bool(
            cfg.get(
                "cache_images",
                False,
            )
        )

        cache_images_mb = float(
            cfg.get(
                "cache_images_mb",
                512,
            )
        )

        if cache_images_mb <= 0:
            raise ValueError(
                "dataset.cache_images_mb must be > 0."
            )

        # Disk cache.
        cache_to_disk = bool(
            cfg.get(
                "cache_to_disk",
                False,
            )
        )

        cache_dir = cfg.get(
            "cache_dir",
            "dataset/.cache",
        )

        if not isinstance(cache_dir, str):
            raise ValueError(
                "dataset.cache_dir must be a string."
            )

        cache_side = int(
            cfg.get(
                "cache_side",
                640,
            )
        )

        if cache_side <= 0:
            raise ValueError(
                "dataset.cache_side must be > 0."
            )

        cache_prebuild = bool(
            cfg.get(
                "cache_prebuild",
                False,
            )
        )

        # Сохраняем значения обратно в нормализованном виде.
        cfg["cache_images"] = cache_images
        cfg["cache_images_mb"] = cache_images_mb
        cfg["cache_to_disk"] = cache_to_disk
        cfg["cache_dir"] = cache_dir
        cfg["cache_side"] = cache_side
        cfg["cache_prebuild"] = cache_prebuild

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    def _validate_model(self) -> None:
        cfg = self.model

        if not isinstance(cfg, dict):
            raise TypeError(
                "model must be a mapping."
            )

        backbone = cfg.get("backbone")

        if not isinstance(backbone, str) or not backbone.strip():
            raise ValueError(
                "model.backbone must be a non-empty string."
            )

        pretrained_backbone = bool(
            cfg.get(
                "pretrained_backbone",
                False,
            )
        )

        input_size = int(
            cfg.get(
                "input_size",
                640,
            )
        )

        if input_size <= 0:
            raise ValueError(
                "model.input_size must be > 0."
            )

        strides = cfg.get(
            "strides",
            [4, 8, 16],
        )

        if not isinstance(strides, list):
            raise ValueError(
                "model.strides must be a list."
            )

        if not strides:
            raise ValueError(
                "model.strides must contain at least one stride."
            )

        normalized_strides = []

        for stride in strides:
            stride = int(stride)

            if stride <= 0:
                raise ValueError(
                    f"model.strides contains invalid value: {stride}"
                )

            if input_size % stride != 0:
                raise ValueError(
                    f"model.input_size={input_size} "
                    f"must be divisible by stride={stride}."
                )

            normalized_strides.append(stride)

        if len(set(normalized_strides)) != len(
            normalized_strides
        ):
            raise ValueError(
                "model.strides must not contain duplicates."
            )

        fpn_channels = int(
            cfg.get(
                "fpn_channels",
                128,
            )
        )

        head_channels = int(
            cfg.get(
                "head_channels",
                128,
            )
        )

        num_classes = int(
            cfg.get(
                "num_classes",
                len(self.dataset.get("classes", [])),
            )
        )

        if fpn_channels <= 0:
            raise ValueError(
                "model.fpn_channels must be > 0."
            )

        if head_channels <= 0:
            raise ValueError(
                "model.head_channels must be > 0."
            )

        if num_classes <= 0:
            raise ValueError(
                "model.num_classes must be > 0."
            )

        dataset_classes = self.dataset.get(
            "classes",
            [],
        )

        if dataset_classes and num_classes != len(
            dataset_classes
        ):
            raise ValueError(
                "model.num_classes must match "
                "len(dataset.classes): "
                f"{num_classes} != {len(dataset_classes)}"
            )

        cfg["pretrained_backbone"] = pretrained_backbone
        cfg["input_size"] = input_size
        cfg["strides"] = normalized_strides
        cfg["fpn_channels"] = fpn_channels
        cfg["head_channels"] = head_channels
        cfg["num_classes"] = num_classes

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    def _validate_train(self) -> None:
        cfg = self.train

        if not isinstance(cfg, dict):
            raise TypeError(
                "train must be a mapping."
            )

        epochs = int(
            cfg.get(
                "epochs",
                100,
            )
        )

        batch_size = int(
            cfg.get(
                "batch_size",
                16,
            )
        )

        workers = int(
            cfg.get(
                "workers",
                0,
            )
        )

        lr = float(
            cfg.get(
                "lr",
                0.002,
            )
        )

        weight_decay = float(
            cfg.get(
                "weight_decay",
                0.0001,
            )
        )

        if epochs <= 0:
            raise ValueError(
                "train.epochs must be > 0."
            )

        if batch_size <= 0:
            raise ValueError(
                "train.batch_size must be > 0."
            )

        if workers < 0:
            raise ValueError(
                "train.workers must be >= 0."
            )

        if lr <= 0:
            raise ValueError(
                "train.lr must be > 0."
            )

        if weight_decay < 0:
            raise ValueError(
                "train.weight_decay must be >= 0."
            )

        device = str(
            cfg.get(
                "device",
                "cuda",
            )
        ).strip().lower()

        if not device:
            raise ValueError(
                "train.device must not be empty."
            )

        seed = int(
            cfg.get(
                "seed",
                42,
            )
        )

        amp = bool(
            cfg.get(
                "amp",
                True,
            )
        )

        patience = int(
            cfg.get(
                "patience",
                20,
            )
        )

        warmup_epochs = int(
            cfg.get(
                "warmup_epochs",
                5,
            )
        )

        grad_clip = float(
            cfg.get(
                "grad_clip",
                10.0,
            )
        )

        if patience < 0:
            raise ValueError(
                "train.patience must be >= 0."
            )

        if warmup_epochs < 0:
            raise ValueError(
                "train.warmup_epochs must be >= 0."
            )

        if warmup_epochs > epochs:
            raise ValueError(
                "train.warmup_epochs must not exceed "
                "train.epochs."
            )

        if grad_clip < 0:
            raise ValueError(
                "train.grad_clip must be >= 0."
            )

        # -----------------------------------------------------
        # LOSS SETTINGS
        # -----------------------------------------------------

        use_focal = bool(
            cfg.get(
                "use_focal",
                True,
            )
        )

        focal_alpha = float(
            cfg.get(
                "focal_alpha",
                0.75,
            )
        )

        focal_gamma = float(
            cfg.get(
                "focal_gamma",
                1.5,
            )
        )

        obj_pos_weight = float(
            cfg.get(
                "obj_pos_weight",
                0.75,
            )
        )

        obj_iou_targets = bool(
            cfg.get(
                "obj_iou_targets",
                False,
            )
        )

        box_loss = str(
            cfg.get(
                "box_loss",
                "eiou",
            )
        ).strip().lower()

        if box_loss not in {
            "giou",
            "eiou",
        }:
            raise ValueError(
                "train.box_loss must be 'giou' or 'eiou'."
            )

        if not 0.0 <= focal_alpha <= 1.0:
            raise ValueError(
                "train.focal_alpha must be in [0, 1]."
            )

        if focal_gamma < 0.0:
            raise ValueError(
                "train.focal_gamma must be >= 0."
            )

        if not 0.0 < obj_pos_weight < 1.0:
            raise ValueError(
                "train.obj_pos_weight must be strictly in (0, 1). "
                "obj_pos_weight = 1.0 отключает negative objectness "
                "loss: модель вырождается в 'объект везде' "
                "(obj -> 0, precision -> ~0, recall ~0.5). "
                "Рекомендуемое значение: 0.75."
            )

        loss_weights = cfg.get(
            "loss_weights",
            {
                "obj": 1.0,
                "box": 1.0,
                "cls": 1.0,
            },
        )

        if not isinstance(loss_weights, dict):
            raise ValueError(
                "train.loss_weights must be a mapping."
            )

        normalized_weights = {
            "obj": float(
                loss_weights.get(
                    "obj",
                    1.0,
                )
            ),
            "box": float(
                loss_weights.get(
                    "box",
                    1.0,
                )
            ),
            "cls": float(
                loss_weights.get(
                    "cls",
                    1.0,
                )
            ),
        }

        for name, value in normalized_weights.items():
            if value < 0.0:
                raise ValueError(
                    f"train.loss_weights.{name} "
                    "must be >= 0."
                )

        # -----------------------------------------------------
        # NORMALIZED VALUES
        # -----------------------------------------------------

        cfg["epochs"] = epochs
        cfg["batch_size"] = batch_size
        cfg["workers"] = workers
        cfg["lr"] = lr
        cfg["weight_decay"] = weight_decay
        cfg["device"] = device
        cfg["seed"] = seed
        cfg["amp"] = amp
        cfg["patience"] = patience
        cfg["warmup_epochs"] = warmup_epochs
        cfg["grad_clip"] = grad_clip

        cfg["use_focal"] = use_focal
        cfg["focal_alpha"] = focal_alpha
        cfg["focal_gamma"] = focal_gamma
        cfg["obj_pos_weight"] = obj_pos_weight
        cfg["obj_iou_targets"] = obj_iou_targets
        cfg["box_loss"] = box_loss
        cfg["loss_weights"] = normalized_weights

    # ---------------------------------------------------------
    # AUGMENTATION
    # ---------------------------------------------------------

    def _validate_augmentation(self) -> None:
        cfg = self.augmentation

        if not isinstance(cfg, dict):
            raise TypeError(
                "augmentation must be a mapping."
            )

        flip_lr = float(
            cfg.get(
                "flip_lr",
                0.5,
            )
        )

        flip_ud = float(
            cfg.get(
                "flip_ud",
                0.0,
            )
        )

        brightness_contrast = float(
            cfg.get(
                "brightness_contrast",
                0.2,
            )
        )

        scale_crop = float(
            cfg.get(
                "scale_crop",
                0.5,
            )
        )

        scale_min = float(
            cfg.get(
                "scale_min",
                0.7,
            )
        )

        scale_max = float(
            cfg.get(
                "scale_max",
                1.0,
            )
        )

        mosaic_prob = float(
            cfg.get(
                "mosaic_prob",
                0.0,
            )
        )

        mixup_prob = float(
            cfg.get(
                "mixup_prob",
                0.0,
            )
        )

        probabilities = {
            "flip_lr": flip_lr,
            "flip_ud": flip_ud,
            "scale_crop": scale_crop,
            "mosaic_prob": mosaic_prob,
            "mixup_prob": mixup_prob,
        }

        for name, value in probabilities.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"augmentation.{name} "
                    "must be in [0, 1]."
                )

        if brightness_contrast < 0.0:
            raise ValueError(
                "augmentation.brightness_contrast "
                "must be >= 0."
            )

        if scale_min <= 0.0:
            raise ValueError(
                "augmentation.scale_min must be > 0."
            )

        if scale_max < scale_min:
            raise ValueError(
                "augmentation.scale_max must be >= "
                "augmentation.scale_min."
            )

        cfg["flip_lr"] = flip_lr
        cfg["flip_ud"] = flip_ud
        cfg["brightness_contrast"] = brightness_contrast
        cfg["scale_crop"] = scale_crop
        cfg["scale_min"] = scale_min
        cfg["scale_max"] = scale_max
        cfg["mosaic_prob"] = mosaic_prob
        cfg["mixup_prob"] = mixup_prob

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    def _validate_val(self) -> None:
        cfg = self.val

        if not isinstance(cfg, dict):
            raise TypeError(
                "val must be a mapping."
            )

        conf_threshold = float(
            cfg.get(
                "conf_threshold",
                0.001,
            )
        )

        iou_threshold = float(
            cfg.get(
                "iou_threshold",
                0.5,
            )
        )

        batch_size = int(
            cfg.get(
                "batch_size",
                1,
            )
        )

        tile_size = int(
            cfg.get(
                "tile_size",
                self.model["input_size"],
            )
        )

        overlap_ratio = float(
            cfg.get(
                "overlap_ratio",
                0.25,
            )
        )

        tile_batch_size = int(
            cfg.get(
                "tile_batch_size",
                4,
            )
        )

        class_aware_nms = bool(
            cfg.get(
                "class_aware_nms",
                False,
            )
        )

        use_tiled_inference = bool(
            cfg.get(
                "use_tiled_inference",
                False,
            )
        )

        interval = int(
            cfg.get(
                "interval",
                5,
            )
        )

        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError(
                "val.conf_threshold must be in [0, 1]."
            )

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(
                "val.iou_threshold must be in [0, 1]."
            )

        if batch_size <= 0:
            raise ValueError(
                "val.batch_size must be > 0."
            )

        if tile_size <= 0:
            raise ValueError(
                "val.tile_size must be > 0."
            )

        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(
                "val.overlap_ratio must be in [0, 1)."
            )

        if tile_batch_size <= 0:
            raise ValueError(
                "val.tile_batch_size must be > 0."
            )

        if interval <= 0:
            raise ValueError(
                "val.interval must be > 0."
            )

        cfg["conf_threshold"] = conf_threshold
        cfg["iou_threshold"] = iou_threshold
        cfg["batch_size"] = batch_size
        cfg["tile_size"] = tile_size
        cfg["overlap_ratio"] = overlap_ratio
        cfg["tile_batch_size"] = tile_batch_size
        cfg["class_aware_nms"] = class_aware_nms
        cfg["use_tiled_inference"] = use_tiled_inference
        cfg["interval"] = interval

    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------

    def _validate_predict(self) -> None:
        cfg = self.predict

        if not isinstance(cfg, dict):
            raise TypeError(
                "predict must be a mapping."
            )

        source = cfg.get(
            "source",
            "",
        )

        source_type = str(
            cfg.get(
                "source_type",
                "image",
            )
        ).strip().lower()

        conf_threshold = float(
            cfg.get(
                "conf_threshold",
                0.20,
            )
        )

        iou_threshold = float(
            cfg.get(
                "iou_threshold",
                0.45,
            )
        )

        # Поддерживаем оба имени:
        #
        # use_tiled_inference
        # use_tiled
        #
        # Каноническое имя — use_tiled_inference.
        use_tiled_inference = bool(
            cfg.get(
                "use_tiled_inference",
                cfg.get(
                    "use_tiled",
                    False,
                ),
            )
        )

        tile_size = int(
            cfg.get(
                "tile_size",
                self.model["input_size"],
            )
        )

        overlap_ratio = float(
            cfg.get(
                "overlap_ratio",
                0.25,
            )
        )

        tile_batch_size = int(
            cfg.get(
                "tile_batch_size",
                4,
            )
        )

        class_aware_nms = bool(
            cfg.get(
                "class_aware_nms",
                False,
            )
        )

        save_txt = bool(
            cfg.get(
                "save_txt",
                True,
            )
        )

        save_img = bool(
            cfg.get(
                "save_img",
                True,
            )
        )

        save_conf = bool(
            cfg.get(
                "save_conf",
                True,
            )
        )

        output_dir = str(
            cfg.get(
                "output_dir",
                "runs/predict",
            )
        )

        if not isinstance(source, str):
            raise ValueError(
                "predict.source must be a string."
            )

        if source_type not in {
            "image",
            "images",
            "video",
            "camera",
            "webcam",
        }:
            raise ValueError(
                "predict.source_type must be one of: "
                "image, images, video, camera, webcam."
            )

        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError(
                "predict.conf_threshold must be in [0, 1]."
            )

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(
                "predict.iou_threshold must be in [0, 1]."
            )

        if tile_size <= 0:
            raise ValueError(
                "predict.tile_size must be > 0."
            )

        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(
                "predict.overlap_ratio must be in [0, 1)."
            )

        if tile_batch_size <= 0:
            raise ValueError(
                "predict.tile_batch_size must be > 0."
            )

        if not output_dir.strip():
            raise ValueError(
                "predict.output_dir must not be empty."
            )

        cfg["source"] = source
        cfg["source_type"] = source_type
        cfg["conf_threshold"] = conf_threshold
        cfg["iou_threshold"] = iou_threshold
        cfg["use_tiled_inference"] = use_tiled_inference
        cfg["tile_size"] = tile_size
        cfg["overlap_ratio"] = overlap_ratio
        cfg["tile_batch_size"] = tile_batch_size
        cfg["class_aware_nms"] = class_aware_nms
        cfg["save_txt"] = save_txt
        cfg["save_img"] = save_img
        cfg["save_conf"] = save_conf
        cfg["output_dir"] = output_dir

    # ---------------------------------------------------------
    # SMALL OBJECT
    # ---------------------------------------------------------

    def _validate_small_object(self) -> None:
        cfg = self.small_object

        if not isinstance(cfg, dict):
            raise TypeError(
                "small_object must be a mapping."
            )

        # Поддерживаем текущие имена YAML.
        small_min_side_px = float(
            cfg.get(
                "small_min_side_px",
                cfg.get(
                    "min_side_px",
                    24,
                ),
            )
        )

        small_area_fraction = float(
            cfg.get(
                "small_area_fraction",
                cfg.get(
                    "area_fraction",
                    0.001,
                ),
            )
        )

        keep_small_boxes = bool(
            cfg.get(
                "keep_small_boxes",
                True,
            )
        )

        # Старое имя также поддерживаем для совместимости.
        min_box_side = float(
            cfg.get(
                "min_box_side",
                small_min_side_px,
            )
        )

        if small_min_side_px <= 0:
            raise ValueError(
                "small_object.small_min_side_px "
                "must be > 0."
            )

        if small_area_fraction < 0.0:
            raise ValueError(
                "small_object.small_area_fraction "
                "must be >= 0."
            )

        if min_box_side <= 0:
            raise ValueError(
                "small_object.min_box_side "
                "must be > 0."
            )

        cfg["small_min_side_px"] = (
            small_min_side_px
        )

        cfg["small_area_fraction"] = (
            small_area_fraction
        )

        cfg["keep_small_boxes"] = (
            keep_small_boxes
        )

        cfg["min_side_px"] = (
            small_min_side_px
        )

        cfg["area_fraction"] = (
            small_area_fraction
        )

        cfg["min_box_side"] = (
            min_box_side
        )


def load_config(
    path: str | Path,
) -> Config:
    """
    Загружает YAML и возвращает Config.

    Пример:

        cfg = load_config(
            "configs/default.yaml"
        )
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Configuration path is not a file: {path}"
        )

    try:
        text = path.read_text(
            encoding="utf-8",
        )
    except UnicodeDecodeError as exc:
        raise UnicodeDecodeError(
            exc.encoding,
            exc.object,
            exc.start,
            exc.end,
            "Configuration file must be UTF-8.",
        ) from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML configuration: {path}"
        ) from exc

    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise ValueError(
            "Configuration YAML root must be a mapping."
        )

    return Config(raw)