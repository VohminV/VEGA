"""Regression tests for regression heads decoding: _decode_reg vs encode."""

import math

import numpy as np
import pytest
import torch

from src.losses import _decode_reg, _giou_loss


def _encode(box_xyxy, stride, cell):
    """Обратная операция к декодированию: box -> (dx,dy,dw,dh)."""
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    iy, ix = cell
    dx = (cx - (ix + 0.5) * stride) / stride
    dy = (cy - (iy + 0.5) * stride) / stride
    dw = math.log(max((x2 - x1) / stride, 1e-4))
    dh = math.log(max((y2 - y1) / stride, 1e-4))
    return [dx, dy, dw, dh]


@pytest.mark.parametrize("stride", [4, 8, 16])
def test_decode_reg_roundtrip(stride):
    input_size = 640
    H = W = input_size // stride
    box_out = torch.zeros(1, 4, H, W)  # биадные нули: ячейки как есть
    # закодируем один бокс в ячейке (iy, ix)
    iy, ix = 10, 12
    box = [40.0, 40.0 + stride * iy, 60.0, 60.0 + stride * iy]  # xyxy в пикселях
    # но _decode_reg использует позиции (B,H,W), подготовим pos-маску
    vals = _encode(box, stride, (iy, ix))
    box_out[0, :, iy, ix] = torch.tensor(vals)
    pos = torch.zeros(1, 1, H, W, dtype=torch.bool)
    pos[0, 0, iy, ix] = True
    decoded = _decode_reg(box_out, pos, stride)
    assert decoded.shape[0] == 1
    x1, y1, x2, y2 = [float(v) for v in decoded[0]]
    assert x1 == pytest.approx(box[0], abs=0.6)
    assert y1 == pytest.approx(box[1], abs=0.6)
    assert x2 == pytest.approx(box[2], abs=0.6)
    assert y2 == pytest.approx(box[3], abs=0.6)


def test_decoded_boxes_clamped_to_image():
    input_size = 64
    stride = 8
    H = W = input_size // stride
    box_out = torch.zeros(1, 4, H, W)
    box_out[0, 0, 0, 0] = 50.0  # далеко за пределами
    box_out[0, 2, 0, 0] = 8.0
    pos = torch.zeros(1, 1, H, W, dtype=torch.bool)
    pos[0, 0, 0, 0] = True
    decoded = _decode_reg(box_out, pos, stride)[0]
    assert float(decoded[0]) >= 0 and float(decoded[2]) <= input_size
    assert float(decoded[1]) >= 0 and float(decoded[3]) <= input_size


def test_giou_zero_for_perfect_match():
    pred = torch.tensor([[0.0, 0.0, 50.0, 50.0]])
    tgt = torch.tensor([[0.0, 0.0, 50.0, 50.0]])
    assert float(_giou_loss(pred, tgt)) == pytest.approx(0.0, abs=1e-6)


def test_giou_one_for_disjoint():
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    tgt = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    val = float(_giou_loss(pred, tgt))
    assert val > 1.0  # GIoU уходит вниз для непересекающихся


def test_giou_empty_input_is_zero():
    pred = torch.zeros((0, 4))
    tgt = torch.zeros((0, 4))
    assert float(_giou_loss(pred, tgt)) == 0.0