"""Минимальный конфиг для unit-тестов, не трогающий реальный файл."""
import copy

from src.config import Config


DEFAULT_AUG = {
    "flip_lr": 0.5,
    "flip_ud": 0.0,
    "brightness_contrast": 0.2,
    "scale_crop": 0.0,
    "scale_min": 0.7,
    "scale_max": 1.0,
}


def make_cfg() -> Config:
    raw = {
        "dataset": {
            "root": "dataset",
            "images_dir": "images",
            "labels_dir": "labels",
            "data_yaml": "data.yaml",
            "classes": ["drone"],
        },
        "model": {
            "backbone": "custom_small",
            "pretrained_backbone": False,
            "input_size": 640,
            "strides": [4, 8, 16],
            "fpn_channels": 128,
            "head_channels": 128,
            "num_classes": 1,
        },
        "train": {
            "epochs": 100,
            "batch_size": 4,
            "device": "cpu",
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "workers": 0,
            "amp": False,
            "seed": 42,
            "save_dir": "runs/train",
            "patience": 20,
            "use_focal": True,
            "use_scheduler": True,
            "warmup_epochs": 5,
            "grad_clip": 10.0,
        },
        "augmentation": copy.deepcopy(DEFAULT_AUG),
        "val": {
            "batch_size": 2,
            "conf_threshold": 0.15,
            "iou_threshold": 0.5,
            "use_tiled_inference": True,
            "tile_size": 640,
            "overlap_ratio": 0.25,
            "tile_batch_size": 4,
            "class_aware_nms": False,
        },
        "predict": {},
        "small_object": {
            "small_min_side_px": 24,
            "small_area_fraction": 0.001,
            "keep_small_boxes": True,
        },
    }
    return Config(raw)