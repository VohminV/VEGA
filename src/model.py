"""Собственная лёгкая детекторная сеть: backbone + FPN + голова.

Выход на уровнях strides [4, 8, 16].

Каждая ячейка предсказывает:
    objectness
    dx, dy
    dw, dh
    class logits

Регрессия bbox:
    dx = (cx / stride) - (grid_x + 0.5)
    dy = (cy / stride) - (grid_y + 0.5)
    dw = log(width / stride)
    dh = log(height / stride)

Декодирование:
    cx = (grid_x + 0.5 + dx) * stride
    cy = (grid_y + 0.5 + dy) * stride
    w  = exp(dw) * stride
    h  = exp(dh) * stride
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic blocks
# ============================================================


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + SiLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        k: int = 3,
    ) -> None:
        super().__init__()

        if in_ch <= 0:
            raise ValueError(
                f"in_ch must be > 0, got {in_ch}"
            )

        if out_ch <= 0:
            raise ValueError(
                f"out_ch must be > 0, got {out_ch}"
            )

        if stride <= 0:
            raise ValueError(
                f"stride must be > 0, got {stride}"
            )

        if k <= 0 or k % 2 == 0:
            raise ValueError(
                f"k must be positive odd number, got {k}"
            )

        pad = k // 2

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=stride,
            padding=pad,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            out_ch
        )

        self.act = nn.SiLU(
            inplace=True
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.act(
            self.bn(
                self.conv(x)
            )
        )


class ResidualBlock(nn.Module):
    """Два conv-слоя с residual connection."""

    def __init__(
        self,
        ch: int,
    ) -> None:
        super().__init__()

        if ch <= 0:
            raise ValueError(
                f"ch must be > 0, got {ch}"
            )

        self.conv1 = ConvBlock(
            ch,
            ch,
            stride=1,
            k=3,
        )

        self.conv2 = nn.Conv2d(
            ch,
            ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            ch
        )

        self.act = nn.SiLU(
            inplace=True
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)

        return self.act(
            x + out
        )


# ============================================================
# Custom backbone
# ============================================================


class Stem(nn.Module):
    """Сжимает разрешение с stride 1 до stride 2."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
    ) -> None:
        super().__init__()

        self.conv = ConvBlock(
            in_ch,
            out_ch,
            stride=2,
            k=3,
        )

        self.res = ResidualBlock(
            out_ch
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.res(
            self.conv(x)
        )


class CustomBackbone(nn.Module):
    """
    Собственный mini-backbone.

    Выходы:
        c4  -> stride 4
        c8  -> stride 8
        c16 -> stride 16
    """

    def __init__(
        self,
        width: int = 64,
    ) -> None:
        super().__init__()

        if width <= 0:
            raise ValueError(
                f"width must be > 0, got {width}"
            )

        self.stem = Stem(
            3,
            width // 2,
        )

        self.stage1 = self._make_stage(
            width // 2,
            width,
        )

        self.stage2 = self._make_stage(
            width,
            width * 2,
        )

        self.stage3 = self._make_stage(
            width * 2,
            width * 4,
        )

    @staticmethod
    def _make_stage(
        in_ch: int,
        out_ch: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            ConvBlock(
                in_ch,
                out_ch,
                stride=2,
            ),
            ResidualBlock(
                out_ch
            ),
            ResidualBlock(
                out_ch
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x = self.stem(x)

        # input / 2
        # ->
        # input / 4
        c4 = self.stage1(x)

        # input / 8
        c8 = self.stage2(c4)

        # input / 16
        c16 = self.stage3(c8)

        return c4, c8, c16


# ============================================================
# ResNet backbone
# ============================================================


class ResnetBackbone(nn.Module):
    """
    Torchvision ResNet18/34 как источник feature maps.

    Выходы:
        c4  -> stride 4
        c8  -> stride 8
        c16 -> stride 16

    layer4 намеренно не используется.
    """

    def __init__(
        self,
        name: str,
        pretrained: bool,
    ) -> None:
        super().__init__()

        import torchvision.models as tv_models

        if name not in (
            "resnet18",
            "resnet34",
        ):
            raise ValueError(
                f"Unknown resnet: {name}"
            )

        # Поддерживаем современные torchvision.
        if pretrained:
            weights: Any = "DEFAULT"
        else:
            weights = None

        if name == "resnet18":
            net = tv_models.resnet18(
                weights=weights
            )
        else:
            net = tv_models.resnet34(
                weights=weights
            )

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        # После conv1 + maxpool:
        # stride = 4
        self.layer1 = net.layer1

        # stride = 8
        self.layer2 = net.layer2

        # stride = 16
        self.layer3 = net.layer3

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c4 = self.layer1(x)
        c8 = self.layer2(c4)
        c16 = self.layer3(c8)

        return c4, c8, c16


# ============================================================
# FPN
# ============================================================


class FPN(nn.Module):
    """
    Top-down FPN.

    Input:
        c4, c8, c16

    Output:
        p4, p8, p16

    В списке:
        [stride 4, stride 8, stride 16]
    """

    def __init__(
        self,
        backbone_channels: list[int],
        fpn_channels: int,
    ) -> None:
        super().__init__()

        if len(backbone_channels) != 3:
            raise ValueError(
                "FPN expects exactly 3 backbone levels "
                "[stride4, stride8, stride16]"
            )

        if any(
            ch <= 0
            for ch in backbone_channels
        ):
            raise ValueError(
                "All backbone channel counts must be > 0"
            )

        if fpn_channels <= 0:
            raise ValueError(
                "fpn_channels must be > 0"
            )

        self.lat = nn.ModuleList(
            [
                nn.Conv2d(
                    ch,
                    fpn_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for ch in backbone_channels
            ]
        )

        self.outs = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(
                        fpn_channels,
                        fpn_channels,
                    ),
                    nn.Conv2d(
                        fpn_channels,
                        fpn_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                )
                for _ in range(3)
            ]
        )

    def forward(
        self,
        features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(features) != 3:
            raise ValueError(
                "FPN expects exactly 3 feature maps"
            )

        c4, c8, c16 = features

        lat4 = self.lat[0](c4)
        lat8 = self.lat[1](c8)
        lat16 = self.lat[2](c16)

        # stride 16 -> stride 8
        p8 = (
            lat8
            + F.interpolate(
                lat16,
                size=lat8.shape[-2:],
                mode="nearest",
            )
        )

        # stride 8 -> stride 4
        p4 = (
            lat4
            + F.interpolate(
                p8,
                size=lat4.shape[-2:],
                mode="nearest",
            )
        )

        p4 = self.outs[0](p4)
        p8 = self.outs[1](p8)
        p16 = self.outs[2](lat16)

        return [
            p4,
            p8,
            p16,
        ]


# ============================================================
# Detection head
# ============================================================


class DetectHead(nn.Module):
    """
    Общая detection head для всех FPN levels.

    Выходы:
        obj -> [B, 1, H, W]
        box -> [B, 4, H, W]
        cls -> [B, C, H, W]
    """

    def __init__(
        self,
        in_ch: int,
        head_ch: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        if in_ch <= 0:
            raise ValueError(
                f"in_ch must be > 0, got {in_ch}"
            )

        if head_ch <= 0:
            raise ValueError(
                f"head_ch must be > 0, got {head_ch}"
            )

        if num_classes <= 0:
            raise ValueError(
                f"num_classes must be > 0, got {num_classes}"
            )

        self.net = nn.Sequential(
            ConvBlock(
                in_ch,
                head_ch,
            ),
            ConvBlock(
                head_ch,
                head_ch,
            ),
        )

        self.obj = nn.Conv2d(
            head_ch,
            1,
            kernel_size=1,
        )

        self.box = nn.Conv2d(
            head_ch,
            4,
            kernel_size=1,
        )

        self.cls = nn.Conv2d(
            head_ch,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        f = self.net(x)

        return (
            self.obj(f),
            self.box(f),
            self.cls(f),
        )


# ============================================================
# Detector
# ============================================================


class CustomDetector(nn.Module):
    """
    Полный detector:

        Backbone
            ↓
        FPN
            ↓
        shared DetectHead
            ↓
        stride 4 / 8 / 16
    """

    # Эти параметры должны быть одинаковыми
    # в train loss и inference decoder.
    REG_MIN = -4.0
    REG_MAX = 4.0

    def __init__(
        self,
        cfg,
    ) -> None:
        super().__init__()

        m = cfg.model

        self.num_classes = int(
            m["num_classes"]
        )

        self.input_size = int(
            m["input_size"]
        )

        self.fpn_channels = int(
            m["fpn_channels"]
        )

        self.head_channels = int(
            m["head_channels"]
        )

        configured_strides = [
            int(s)
            for s in m["strides"]
        ]

        if sorted(
            configured_strides
        ) != [4, 8, 16]:
            raise ValueError(
                "model.strides must contain "
                "[4, 8, 16], "
                f"got {configured_strides}"
            )

        # Архитектура жёстко построена под эти levels.
        self.strides: list[int] = [
            4,
            8,
            16,
        ]

        if self.input_size <= 0:
            raise ValueError(
                "model.input_size must be > 0"
            )

        if self.input_size % 16 != 0:
            raise ValueError(
                "model.input_size must be divisible by 16; "
                f"got {self.input_size}"
            )

        backbone_name = str(
            m["backbone"]
        )

        if backbone_name == "custom_small":
            self.backbone = CustomBackbone(
                width=64
            )

            backbone_channels = [
                64,
                128,
                256,
            ]

        elif backbone_name in (
            "resnet18",
            "resnet34",
        ):
            self.backbone = ResnetBackbone(
                backbone_name,
                bool(
                    m["pretrained_backbone"]
                ),
            )

            backbone_channels = [
                64,
                128,
                256,
            ]

        else:
            raise ValueError(
                f"Unknown backbone: "
                f"{backbone_name}"
            )

        self.fpn = FPN(
            backbone_channels,
            self.fpn_channels,
        )

        self.head = DetectHead(
            self.fpn_channels,
            self.head_channels,
            self.num_classes,
        )

        self._initialize_detection_heads()

    def _initialize_detection_heads(
        self,
    ) -> None:
        """
        Инициализация detection logits.

        Objectness bias делаем отрицательным, потому что
        положительных cells намного меньше отрицательных.

        Это особенно важно для dense detector:
        иначе в начале обучения практически вся карта
        начинает иметь objectness около 0.5.
        """
        nn.init.normal_(
            self.head.obj.weight,
            mean=0.0,
            std=0.01,
        )

        # Начальная вероятность objectness.
        prior_prob = 0.01

        prior_logit = (
            torch.log(
                torch.tensor(
                    prior_prob
                )
            )
            - torch.log(
                torch.tensor(
                    1.0 - prior_prob
                )
            )
        )

        nn.init.constant_(
            self.head.obj.bias,
            float(prior_logit),
        )

        nn.init.normal_(
            self.head.box.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.constant_(
            self.head.box.bias,
            0.0,
        )

        nn.init.normal_(
            self.head.cls.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.constant_(
            self.head.cls.bias,
            0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> list[
        dict[str, torch.Tensor | int]
    ]:
        if x.ndim != 4:
            raise ValueError(
                "Detector input must have shape "
                "[B, C, H, W], "
                f"got {tuple(x.shape)}"
            )

        if x.shape[1] != 3:
            raise ValueError(
                "Detector expects 3 input channels, "
                f"got {x.shape[1]}"
            )

        c4, c8, c16 = self.backbone(x)

        levels = self.fpn(
            [
                c4,
                c8,
                c16,
            ]
        )

        expected_sizes = [
            self.input_size // 4,
            self.input_size // 8,
            self.input_size // 16,
        ]

        for idx, (
            feat,
            expected,
        ) in enumerate(
            zip(
                levels,
                expected_sizes,
            )
        ):
            actual_h = feat.shape[-2]
            actual_w = feat.shape[-1]

            if (
                actual_h != expected
                or actual_w != expected
            ):
                raise RuntimeError(
                    "FPN spatial size mismatch at "
                    f"level {idx}: "
                    f"expected {expected}x{expected}, "
                    f"got {actual_h}x{actual_w}"
                )

        outputs: list[
            dict[str, torch.Tensor | int]
        ] = []

        for stride, feat in zip(
            self.strides,
            levels,
        ):
            obj, box, cls = self.head(
                feat
            )

            outputs.append(
                {
                    "stride": stride,
                    "obj": obj,
                    "box": box,
                    "cls": cls,
                }
            )

        return outputs

    @torch.no_grad()
    def decode_level(
        self,
        obj: torch.Tensor,
        box: torch.Tensor,
        cls: torch.Tensor,
        stride: int,
        conf_threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        Декодирует один detection level.

        Вход:
            obj: [1, 1, H, W]
            box: [1, 4, H, W]
            cls: [1, C, H, W]

        Возвращает:
            [N, 6]

        Формат:
            x1, y1, x2, y2, confidence, class_id
        """
        if obj.ndim != 4:
            raise ValueError(
                "obj must be [B,1,H,W], "
                f"got {tuple(obj.shape)}"
            )

        if box.ndim != 4:
            raise ValueError(
                "box must be [B,4,H,W], "
                f"got {tuple(box.shape)}"
            )

        if cls.ndim != 4:
            raise ValueError(
                "cls must be [B,C,H,W], "
                f"got {tuple(cls.shape)}"
            )

        B, obj_channels, H, W = (
            obj.shape
        )

        if obj_channels != 1:
            raise ValueError(
                "Objectness output must have "
                f"1 channel, got {obj_channels}"
            )

        if box.shape[1] != 4:
            raise ValueError(
                "Box output must have "
                f"4 channels, got {box.shape[1]}"
            )

        if cls.shape[1] != self.num_classes:
            raise ValueError(
                "Class output channel mismatch: "
                f"expected {self.num_classes}, "
                f"got {cls.shape[1]}"
            )

        if (
            box.shape[-2:] != (H, W)
            or cls.shape[-2:] != (H, W)
        ):
            raise ValueError(
                "obj/box/cls spatial dimensions "
                "must match"
            )

        if B != 1:
            raise ValueError(
                "decode_level expects a single image "
                "[B=1]. Use decode() for a batch."
            )

        if stride not in self.strides:
            raise ValueError(
                f"Unknown stride {stride}; "
                f"expected one of {self.strides}"
            )

        device = obj.device

        # ----------------------------------------------------
        # Grid.
        # ----------------------------------------------------
        iy, ix = torch.meshgrid(
            torch.arange(
                H,
                device=device,
            ),
            torch.arange(
                W,
                device=device,
            ),
            indexing="ij",
        )

        ix = ix.float()
        iy = iy.float()

        # ----------------------------------------------------
        # Regression.
        # ----------------------------------------------------
        box = box.float()

        dx = box[0, 0]
        dy = box[0, 1]

        dw = torch.clamp(
            box[0, 2],
            self.REG_MIN,
            self.REG_MAX,
        )

        dh = torch.clamp(
            box[0, 3],
            self.REG_MIN,
            self.REG_MAX,
        )

        S = float(stride)

        cx = (
            ix
            + 0.5
            + dx
        ) * S

        cy = (
            iy
            + 0.5
            + dy
        ) * S

        w = (
            torch.exp(dw)
            * S
        )

        h = (
            torch.exp(dh)
            * S
        )

        max_x = float(
            W * stride
        )

        max_y = float(
            H * stride
        )

        x1 = torch.clamp(
            cx - w / 2.0,
            0.0,
            max_x,
        )

        y1 = torch.clamp(
            cy - h / 2.0,
            0.0,
            max_y,
        )

        x2 = torch.clamp(
            cx + w / 2.0,
            0.0,
            max_x,
        )

        y2 = torch.clamp(
            cy + h / 2.0,
            0.0,
            max_y,
        )

        # ----------------------------------------------------
        # Objectness.
        # ----------------------------------------------------
        obj_p = torch.sigmoid(
            obj[0, 0]
        )

        # ----------------------------------------------------
        # Classification.
        #
        # BCEWithLogits используется в loss.py,
        # поэтому здесь sigmoid, а не softmax.
        # ----------------------------------------------------
        if self.num_classes > 1:
            cls_p = torch.sigmoid(
                cls[0]
            )

            class_conf, class_id = (
                cls_p.max(dim=0)
            )

            confidence = (
                obj_p
                * class_conf
            )

        else:
            class_id = torch.zeros(
                (H, W),
                dtype=torch.long,
                device=device,
            )

            confidence = obj_p

        # ----------------------------------------------------
        # Valid geometry + optional confidence threshold.
        # ----------------------------------------------------
        valid = (
            (x2 > x1)
            & (y2 > y1)
            & torch.isfinite(x1)
            & torch.isfinite(y1)
            & torch.isfinite(x2)
            & torch.isfinite(y2)
            & torch.isfinite(confidence)
            & (
                confidence
                >= float(conf_threshold)
            )
        )

        if not bool(valid.any()):
            return torch.zeros(
                (0, 6),
                dtype=torch.float32,
                device=device,
            )

        boxes = torch.stack(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            dim=-1,
        )

        scores = confidence.unsqueeze(
            -1
        )

        classes = class_id.float().unsqueeze(
            -1
        )

        pred = torch.cat(
            [
                boxes,
                scores,
                classes,
            ],
            dim=-1,
        )

        return pred[valid]

    @torch.no_grad()
    def decode(
        self,
        outputs: list[
            dict[str, torch.Tensor | int]
        ],
        conf_threshold: float = 0.0,
    ) -> list[torch.Tensor]:
        """
        Декодирует все levels.

        Возвращает список длины B.

        Каждый элемент:
            [N, 6]

        Формат:
            [x1, y1, x2, y2, conf, cls]
        """
        if not outputs:
            return []

        first_obj = outputs[0]["obj"]

        if not isinstance(
            first_obj,
            torch.Tensor,
        ):
            raise TypeError(
                "outputs[0]['obj'] must be Tensor"
            )

        batch_size = (
            first_obj.shape[0]
        )

        per_image_preds: list[
            list[torch.Tensor]
        ] = [
            []
            for _ in range(batch_size)
        ]

        for out in outputs:
            obj = out["obj"]
            box = out["box"]
            cls = out["cls"]
            stride = out["stride"]

            if not isinstance(
                obj,
                torch.Tensor,
            ):
                raise TypeError(
                    "output['obj'] must be Tensor"
                )

            if not isinstance(
                box,
                torch.Tensor,
            ):
                raise TypeError(
                    "output['box'] must be Tensor"
                )

            if not isinstance(
                cls,
                torch.Tensor,
            ):
                raise TypeError(
                    "output['cls'] must be Tensor"
                )

            if not isinstance(
                stride,
                int,
            ):
                stride = int(stride)

            for i in range(
                batch_size
            ):
                level_obj = obj[
                    i : i + 1
                ]

                level_box = box[
                    i : i + 1
                ]

                level_cls = cls[
                    i : i + 1
                ]

                per = self.decode_level(
                    level_obj,
                    level_box,
                    level_cls,
                    stride,
                    conf_threshold=conf_threshold,
                )

                if per.shape[0] > 0:
                    per_image_preds[
                        i
                    ].append(per)

        result: list[
            torch.Tensor
        ] = []

        for preds in per_image_preds:
            if preds:
                result.append(
                    torch.cat(
                        preds,
                        dim=0,
                    )
                )
            else:
                result.append(
                    torch.zeros(
                        (
                            0,
                            6,
                        ),
                        dtype=torch.float32,
                        device=first_obj.device,
                    )
                )

        return result

    def box_param_count(
        self,
    ) -> int:
        """Количество параметров модели."""
        return sum(
            p.numel()
            for p in self.parameters()
        )


def build_model(
    cfg,
) -> CustomDetector:
    """Создаёт CustomDetector из конфигурации."""
    return CustomDetector(cfg)