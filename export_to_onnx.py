"""Экспорт обученной модели в ONNX.

Использование:
    python export_to_onnx.py --weights runs/train/best.pt --config configs/default.yaml
    python export_to_onnx.py --weights runs/train/best.pt --config configs/default.yaml --batch 4
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from src.config import load_config
from src.model import CustomDetector, build_model


class DetectorExport(nn.Module):
    """Обёртка: forward возвращает tuple[torch.Tensor] вместо list[dict].

    Для каждого уровня пирамиды выдаётся тензор (B, 1+4+C, H, W):
      канал 0     — logit objectness
      каналы 1..4 — dx, dy, dw, dh
      каналы 5..  — logits классов
    """

    def __init__(self, model: CustomDetector) -> None:
        super().__init__()
        self.model = model
        self.strides: list[int] = model.strides

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outs = self.model(x)
        return tuple(
            torch.cat([o["obj"], o["box"], o["cls"]], dim=1) for o in outs
        )


def load_model_weights(model: CustomDetector, weights: str) -> None:
    ckpt = torch.load(weights, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export detector to ONNX")
    parser.add_argument("--weights", type=str, default="runs/train/best.pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="runs/export/detector.onnx")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    load_model_weights(model, args.weights)
    model.eval()

    wrapper = DetectorExport(model)
    wrapper.eval()
    input_size = int(cfg.model["input_size"])
    dummy = torch.zeros(args.batch, 3, input_size, input_size)

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        input_names=["images"],
        output_names=[f"out_s{s}" for s in model.strides],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={  # закомментируй, если нужен фиксированный вход
            "images": {0: "batch"},
            **{f"out_s{s}": {0: "batch"} for s in model.strides},
        },
    )
    print(f"OK -> {args.output} (strides={model.strides}, input={input_size})")


if __name__ == "__main__":
    main()