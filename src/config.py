"""Конфигурация проекта: чтение YAML-конфига с проверкой."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw: dict[str, Any] = raw
        self.dataset: dict[str, Any] = raw["dataset"]
        self.model: dict[str, Any] = raw["model"]
        self.train: dict[str, Any] = raw["train"]
        self.augmentation: dict[str, Any] = raw["augmentation"]
        self.val: dict[str, Any] = raw["val"]
        self.predict: dict[str, Any] = raw["predict"]
        self.small_object: dict[str, Any] = raw["small_object"]


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    _validate(raw)
    return Config(raw)


def _validate(raw: dict[str, Any]) -> None:
    required_sections = [
        "dataset", "model", "train", "augmentation", "val", "predict", "small_object",
    ]
    for section in required_sections:
        if section not in raw:
            raise ValueError(f"Config missing required section: {section}")

    model: dict[str, Any] = raw["model"]
    if "input_size" not in model:
        raise ValueError("Config missing model.input_size")
    if "strides" not in model or not isinstance(model["strides"], list):
        raise ValueError("Config missing model.strides (list)")
    if 4 not in model["strides"]:
        raise ValueError("model.strides must contain 4 for small objects")
    if "num_classes" not in model or model["num_classes"] < 1:
        raise ValueError("model.num_classes must be >= 1")