"""Собственная лёгкая детекторная сеть: backbone + FPN + голова.

Выход на уровнях strides [4, 8, 16]. Каждая ячейка предсказывает:
objectness, смещение центра (dx,dy), логарифм размера (dw,dh) и класс.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, k: int = 3) -> None:
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock(ch, ch, stride=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        out = self.bn2(out)
        return self.act(x + out)


class Stem(nn.Module):
    """Сжимает разрешение вдвое (stride 2 на входе)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, stride=2, k=3)
        self.res = ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.conv(x))


class CustomBackbone(nn.Module):
    """Собственный мини-backbone: выходы на stride [4, 8, 16]."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.stem = Stem(3, width // 2)
        self.stage1 = self._make_stage(width // 2, width)
        self.stage2 = self._make_stage(width, width * 2)
        self.stage3 = self._make_stage(width * 2, width * 4)

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            ConvBlock(in_ch, out_ch, stride=2),
            ResidualBlock(out_ch),
            ResidualBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)          # stride 2
        c4 = self.stage1(x)       # stride 4
        c8 = self.stage2(c4)      # stride 8
        c16 = self.stage3(c8)     # stride 16
        return c4, c8, c16


class ResnetBackbone(nn.Module):
    """Torchvision ResNet18/34 как источник фичей на strides [4,8,16]."""

    def __init__(self, name: str, pretrained: bool) -> None:
        super().__init__()
        import torchvision.models as tv_models

        weights = "DEFAULT" if pretrained else None
        if name == "resnet18":
            net = tv_models.resnet18(weights=weights)
        elif name == "resnet34":
            net = tv_models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unknown resnet: {name}")
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1  # stride 4
        self.layer2 = net.layer2  # stride 8
        self.layer3 = net.layer3  # stride 16

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # stride 4
        c4 = self.layer1(x)
        c8 = self.layer2(c4)
        c16 = self.layer3(c8)
        return c4, c8, c16


class FPN(nn.Module):
    """Простой top-down FPN поверх трёх уровней backbone."""

    def __init__(self, backbone_channels: list[int], fpn_channels: int) -> None:
        super().__init__()
        self.lat = nn.ModuleList(
            [nn.Conv2d(ch, fpn_channels, 1) for ch in backbone_channels]
        )
        self.outs = nn.ModuleList(
            [nn.Sequential(ConvBlock(fpn_channels, fpn_channels), nn.Conv2d(fpn_channels, fpn_channels, 1)) for _ in range(3)]
        )

    def forward(
        self, features: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        lat = [conv(f) for conv, f in zip(self.lat, features)]
        p3 = lat[2]
        p2 = lat[1] + F.interpolate(p3, scale_factor=2, mode="nearest")
        p1 = lat[0] + F.interpolate(p2, scale_factor=2, mode="nearest")
        return [out(f) for out, f in zip(self.outs, [p1, p2, p3])]


class DetectHead(nn.Module):
    """Общая для всех уровней голова: objectness, box, class."""

    def __init__(self, in_ch: int, head_ch: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_ch, head_ch),
            ConvBlock(head_ch, head_ch),
        )
        self.obj = nn.Conv2d(head_ch, 1, 1)
        self.box = nn.Conv2d(head_ch, 4, 1)
        self.cls = nn.Conv2d(head_ch, num_classes, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.net(x)
        return self.obj(f), self.box(f), self.cls(f)


class CustomDetector(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        m = cfg.model
        self.num_classes = int(m["num_classes"])
        self.input_size = int(m["input_size"])
        self.strides: list[int] = [int(s) for s in m["strides"]]
        self.fpn_channels = int(m["fpn_channels"])
        self.head_channels = int(m["head_channels"])

        backbone_name: str = m["backbone"]
        if backbone_name == "custom_small":
            self.backbone = CustomBackbone(width=64)
            backbone_channels = [64, 128, 256]
        elif backbone_name in ("resnet18", "resnet34"):
            self.backbone = ResnetBackbone(backbone_name, bool(m["pretrained_backbone"]))
            backbone_channels = [64, 128, 256]
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.fpn = FPN(backbone_channels, self.fpn_channels)
        self.head = DetectHead(self.fpn_channels, self.head_channels, self.num_classes)

    def forward(self, x: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        c4, c8, c16 = self.backbone(x)
        levels = self.fpn([c4, c8, c16])
        outputs = []
        for stride, feat in zip(self.strides, levels):
            obj, box, cls = self.head(feat)
            outputs.append({"stride": stride, "obj": obj, "box": box, "cls": cls})
        return outputs

    def decode_level(
        self,
        obj: torch.Tensor,
        box: torch.Tensor,
        cls: torch.Tensor,
        stride: int,
    ) -> torch.Tensor:
        """Декодирует один уровень в боксы xyxy + оценки.

        Возвращает (N, 7): x1 y1 x2 y2 objectness class_conf class_id.
        """
        B, _, H, W = obj.shape
        device = obj.device
        iy, ix = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij",
        )
        dx, dy, dw, dh = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        dw = torch.clamp(dw, -8.0, 8.0)
        dh = torch.clamp(dh, -8.0, 8.0)
        cx = (ix.float() + 0.5 + dx) * stride
        cy = (iy.float() + 0.5 + dy) * stride
        w = torch.exp(dw) * stride
        h = torch.exp(dh) * stride
        x1 = torch.clamp(cx - w / 2.0, 0, self.input_size)
        y1 = torch.clamp(cy - h / 2.0, 0, self.input_size)
        x2 = torch.clamp(cx + w / 2.0, 0, self.input_size)
        y2 = torch.clamp(cy + h / 2.0, 0, self.input_size)

        obj_p = torch.sigmoid(obj)
        if self.num_classes > 1:
            cls_p = torch.sigmoid(cls)
            class_conf, class_id = cls_p.max(dim=1)
            class_conf = class_conf * obj_p[:, 0]
        else:
            class_id = torch.zeros_like(obj_p[:, 0], dtype=torch.long)
            class_conf = obj_p[:, 0].clone()

        valid = (x2 > x1) & (y2 > y1) & (obj_p[:, 0] > 0)
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        scores = torch.stack([class_conf, class_id.float()], dim=-1)
        pred = torch.cat([boxes, scores], dim=-1)  # B,H,W,6
        pred = pred[valid]
        return pred

    def decode(self, outputs: list[dict[str, torch.Tensor]]) -> list[torch.Tensor]:
        """Декодирует все уровни. Возвращает (N,6) [x1,y1,x2,y2,conf,cls] на образ."""
        batch_size = outputs[0]["obj"].shape[0]
        per_image_preds: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        for out in outputs:
            stride = out["stride"]
            for i in range(batch_size):
                level_obj = out["obj"][i : i + 1]
                level_box = out["box"][i : i + 1]
                level_cls = out["cls"][i : i + 1]
                per = self.decode_level(level_obj, level_box, level_cls, stride)
                if per.shape[0] > 0:
                    per_image_preds[i].append(per)
        result: list[torch.Tensor] = []
        for preds in per_image_preds:
            if preds:
                result.append(torch.cat(preds, dim=0))
            else:
                result.append(torch.zeros((0, 6), device=outputs[0]["obj"].device))
        return result

    def box_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg) -> CustomDetector:
    return CustomDetector(cfg)