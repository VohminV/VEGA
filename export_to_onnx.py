"""Экспорт обученной модели в ONNX.

Два режима:
  raw  — "сырой" экспорт бэкбона: выход — 3 тензора obj/box/cls на каждый уровень.
         Декодирование и NMS выполняются снаружи (torch/numpy/OpenVINO NMS).
         Формы статичны; для динамического батча/размера добавьте --dynamic.
  e2e  — end-to-end: forward + decode + NMS внутри одной ONNX-модели.
         Выход фиксированной формы (1, max_dets, 6):
         [x1, y1, x2, y2, score, class_id] в пикселях исходного изображения.
         NMS выполняется в графе (opset >= 16), пригодно для OpenVINO/ONNX Runtime.

Использование:
    python export_to_onnx.py --weights runs/train/best.pt --config configs/default.yaml
    python export_to_onnx.py --weights runs/train/best.pt --mode raw --output detector_raw.onnx
    python export_to_onnx.py --weights runs/train/best.pt --conf 0.25 --iou 0.45 --max-dets 300
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torchvision.ops import nms

from src.config import load_config
from src.model import CustomDetector, build_model


def _shapes(s: int, size: int) -> tuple[int, int]:
    return size // s, size // s


class DetectorRaw(nn.Module):
    """Сырой экспорт: возвращает (obj, box, cls) на каждый уровень."""

    def __init__(self, model: CustomDetector) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outs = self.model(x)
        return tuple(torch.cat([out["obj"], out["box"], out["cls"]], dim=1) for out in outs)


class DetectorEndToEnd(nn.Module):
    """Forward + decode + NMS в одном графе, экспортируемом в ONNX.

    Выход: (1, max_dets, 6) — [x1, y1, x2, y2, score, class_id].
    Всегда фиксированного размера (подобран под batch=1).
    """

    def __init__(self, model: CustomDetector, conf: float, iou: float, max_dets: int = 300) -> None:
        super().__init__()
        self.model = model
        self.strides = model.strides
        self.input_size = model.input_size
        self.num_classes = model.num_classes
        self.conf = conf
        self.iou = iou
        self.max_dets = max_dets

        # Предодалленные сетки координат для каждого уровня
        grids = []
        for stride in self.strides:
            H, W = _shapes(stride, self.input_size)
            yv, xv = torch.meshgrid(
                torch.arange(H, dtype=torch.float32),
                torch.arange(W, dtype=torch.float32),
                indexing="ij",
            )
            grids.append(torch.stack([xv, yv], dim=-1).reshape(-1, 2))
        for i, g in enumerate(grids):
            self.register_buffer(f"grid_{i}", g)

    def _decode_level(self, obj, box, cls, stride: int, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        grid = self.get_buffer(f"grid_{idx}")
        B, C, H, W = obj.shape

        obj_sig = torch.sigmoid(obj[:, 0]).reshape(B, -1)  # (B, HW)
        box_t = box.permute(0, 2, 3, 1).reshape(B, -1, 4)
        cls_sig = torch.sigmoid(cls).permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)

        dx, dy = box_t[:, :, 0], box_t[:, :, 1]
        dw = torch.clamp(box_t[:, :, 2], -8.0, 8.0)
        dh = torch.clamp(box_t[:, :, 3], -8.0, 8.0)

        cx = (grid[:, 0] + 0.5 + dx) * stride
        cy = (grid[:, 1] + 0.5 + dy) * stride
        w = torch.exp(dw) * stride
        h = torch.exp(dh) * stride

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        x1 = torch.clamp(x1, 0.0, float(self.input_size))
        y1 = torch.clamp(y1, 0.0, float(self.input_size))
        x2 = torch.clamp(x2, 0.0, float(self.input_size))
        y2 = torch.clamp(y2, 0.0, float(self.input_size))

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, HW, 4)
        if self.num_classes <= 1:
            scores = obj_sig.unsqueeze(-1)              # (B, HW, 1), как в decode_level
        else:
            scores = cls_sig * obj_sig.unsqueeze(-1)    # (B, HW, C)
        return boxes, scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = self.model(x)

        all_boxes: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        for i, out in enumerate(outs):
            boxes, scores = self._decode_level(out["obj"], out["box"], out["cls"], self.strides[i], i)
            all_boxes.append(boxes)
            all_scores.append(scores)

        boxes = torch.cat(all_boxes, dim=1)   # (1, N, 4)
        scores = torch.cat(all_scores, dim=1)  # (1, N, C)

        scores_max, class_idx = scores[0].max(dim=1)  # (N,), (N,)

        # Только ячейки выше порога попадают в NMS (иначе слабые боксы
        # не подавляются и «протекают» в выход как (0,0,0,0,0,0)).
        conf_mask = scores_max >= self.conf            # (N,) bool
        boxes_f = boxes[0][conf_mask]
        scores_f = scores_max[conf_mask]
        cls_f = class_idx[conf_mask]

        # NMS квадратичен по числу кандидатов: жёсткий pre-прин в top-K,
        # иначе сотни тысяч ячеек на 1024² взрывают память и время инференса.
        topk = min(self.max_dets * 8, scores_f.numel())
        if int(topk) < scores_f.numel():
            vals, idx = torch.topk(scores_f, int(topk))
            boxes_f = boxes_f[idx]
            scores_f = vals
            cls_f = cls_f[idx]

        if scores_f.numel() == 0:
            return torch.zeros(1, self.max_dets, 6, device=x.device, dtype=torch.float32)

        keep = nms(boxes_f, scores_f, self.iou)        # (k,)
        picked_boxes = boxes_f[keep]
        picked_scores = scores_f[keep]
        picked_cls = cls_f[keep].float()

        # torchvision.nms не гарантирует порядок по уверенности —
        # сортируем и обрезаем до max_dets.
        order = torch.argsort(picked_scores, descending=True)[: self.max_dets]
        cat = torch.cat(
            [picked_boxes[order], picked_scores[order].unsqueeze(1), picked_cls[order].unsqueeze(1)],
            dim=1,
        )
        pad = torch.zeros(self.max_dets, 6, device=x.device, dtype=torch.float32)
        big = torch.cat([cat, pad], dim=0)  # (K + max_dets, 6)
        result = big[: self.max_dets].reshape(1, self.max_dets, 6)
        return result  # (1, max_dets, 6) — реальные детекции первыми, остальное нули


def load_model_weights(model: CustomDetector, weights: str) -> None:
    ckpt = torch.load(weights, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export detector to ONNX (raw or end-to-end)")
    parser.add_argument("--weights", type=str, default="runs/train/best.pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="runs/export/detector_e2e.onnx")
    parser.add_argument("--mode", type=str, choices=["raw", "e2e"], default="e2e",
                        help="raw: сырые тензоры per-level; e2e: decode+NMS в графе")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (в NMS)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--max-dets", type=int, default=300, help="Max detections per image")
    parser.add_argument("--dynamic", action="store_true", help="Динамический батч (raw: dynamic_axes)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    load_model_weights(model, args.weights)
    model.eval()

    input_size = int(cfg.model["input_size"])
    dummy = torch.zeros(1, 3, input_size, input_size)

    if args.mode == "raw":
        wrapper = DetectorRaw(model).eval()
        dynamic_axes = None
        if args.dynamic:
            dynamic_axes = {"images": {0: "batch"}}
        output_names = [f"output_{i}" for i in range(len(model.strides))]
        torch.onnx.export(
            wrapper,
            dummy,
            args.output,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        print(f"OK -> {args.output}")
        print(f"  mode=raw, strides={model.strides}, input={input_size}")
        print("  output: 3 x (1, 1+4+C, H, W) per level — [obj, box, cls]")
        return

    wrapper = DetectorEndToEnd(model, conf=args.conf, iou=args.iou, max_dets=args.max_dets)
    wrapper.eval()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        input_names=["images"],
        output_names=["detections"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"OK -> {args.output}")
    print(f"  mode=e2e, strides={model.strides}, input={input_size}")
    print(f"  conf={args.conf}, iou={args.iou}, max_dets={args.max_dets}")
    print(f"  output: (1, {args.max_dets}, 6) — [x1,y1,x2,y2,score,class_id]")


if __name__ == "__main__":
    main()