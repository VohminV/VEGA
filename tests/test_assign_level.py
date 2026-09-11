"""Equivalence: vectorized _assign_level vs old sequential reference."""
import math

import numpy as np
import pytest
import torch

from src.losses import _assign_level as new_assign


def old_assign(boxes, class_ids, mask, stride, input_size, num_classes):
    B, N, _ = boxes.shape
    H = W = input_size // stride
    device = boxes.device
    obj_t = torch.zeros(B, 1, H, W, device=device)
    box_t = torch.zeros(B, 4, H, W, device=device)
    gt_box_xyxy = torch.zeros(B, 4, H, W, device=device)
    gt_class = torch.zeros(B, H, W, dtype=torch.long, device=device)
    best_area = torch.full((B, H, W), -1.0, device=device)

    for b in range(B):
        valid = torch.where(mask[b] > 0.5)[0]
        for j in valid.tolist():
            x1, y1, x2, y2 = (float(v) for v in boxes[b, j])
            if x2 <= x1 or y2 <= y1:
                continue
            w = x2 - x1
            h = y2 - y1
            cxs = (x1 + x2) / 2.0
            cys = (y1 + y2) / 2.0
            area = w * h

            ix1 = max(0, min(W - 1, int(math.floor(x1 / stride))))
            iy1 = max(0, min(H - 1, int(math.floor(y1 / stride))))
            ix2 = max(0, min(W - 1, int(math.floor((x2 - 1e-6) / stride))))
            iy2 = max(0, min(H - 1, int(math.floor((y2 - 1e-6) / stride))))

            small = w < stride or h < stride
            cells = set()
            if small:
                ixc = max(0, min(W - 1, int(math.floor((cxs - 1e-6) / stride))))
                iyc = max(0, min(H - 1, int(math.floor((cys - 1e-6) / stride))))
                cells.add((iyc, ixc))
                for iy, ix in ((iyc - 1, ixc), (iyc + 1, ixc), (iyc, ixc - 1), (iyc, ixc + 1)):
                    if 0 <= iy < H and 0 <= ix < W:
                        cells.add((iy, ix))
            cells.update({(iy1, ix1), (iy1, ix2), (iy2, ix1), (iy2, ix2)})
            if not small and (iy1 != iy2 or ix1 != ix2):
                for iy in range(iy1, iy2 + 1):
                    for ix in range(ix1, ix2 + 1):
                        cells.add((iy, ix))

            small_priority = stride == 4
            for iy, ix in cells:
                bx = (ix + 0.5) * stride
                by = (iy + 0.5) * stride
                cur = best_area[b, iy, ix]
                if small_priority:
                    if cur >= 0.0 and area >= cur:
                        continue
                else:
                    if cur >= 0.0 and area <= cur:
                        continue
                best_area[b, iy, ix] = area
                obj_t[b, 0, iy, ix] = 1.0
                dx = (cxs - bx) / stride
                dy = (cys - by) / stride
                dw = math.log(max(w / stride, 1e-4))
                dh = math.log(max(h / stride, 1e-4))
                box_t[b, 0, iy, ix] = dx
                box_t[b, 1, iy, ix] = dy
                box_t[b, 2, iy, ix] = dw
                box_t[b, 3, iy, ix] = dh
                gt_box_xyxy[b, 0, iy, ix] = x1
                gt_box_xyxy[b, 1, iy, ix] = y1
                gt_box_xyxy[b, 2, iy, ix] = x2
                gt_box_xyxy[b, 3, iy, ix] = y2
                gt_class[b, iy, ix] = class_ids[b, j]
    return {"obj": obj_t, "box": box_t, "gt": gt_box_xyxy, "cls": gt_class}


def _random_batch(input_size, num_classes):
    B, N = 3, 12
    boxes = torch.zeros(B, N, 4)
    mask = torch.zeros(B, N)
    classes = torch.zeros(B, N, dtype=torch.long)
    for b in range(B):
        for j in range(N):
            if np.random.rand() < 0.75:
                r = np.random.rand(2)
                sz = np.random.choice([2, 3, 5, 7, 9, 16, 40, 60])
                if np.random.rand() < 0.5:
                    x1 = np.random.uniform(0, input_size - 2)
                    y1 = np.random.uniform(0, input_size - 2)
                    x2 = min(input_size, x1 + np.random.uniform(1, sz))
                    y2 = min(input_size, y1 + np.random.uniform(1, sz))
                else:
                    x1 = np.random.uniform(0, input_size)
                    y1 = np.random.uniform(0, input_size)
                    x2 = max(0.0, min(input_size, x1 + np.random.uniform(1, sz)))
                    y2 = max(0.0, min(input_size, y1 + np.random.uniform(1, sz)))
                if x2 > x1 and y2 > y1:
                    boxes[b, j] = torch.tensor([x1, y1, x2, y2])
                    mask[b, j] = 1.0
                    classes[b, j] = np.random.randint(0, num_classes)
    return boxes, classes, mask


@pytest.mark.parametrize("trial", range(40))
def test_assign_level_equiv(trial):
    torch.manual_seed(trial)
    np.random.seed(trial)
    input_size = 64
    num_classes = 3
    strides = [4, 8, 16]
    boxes, classes, mask = _random_batch(input_size, num_classes)
    for stride in strides:
        o = old_assign(boxes, classes, mask, stride, input_size, num_classes)
        n = new_assign(boxes, classes, mask, stride, input_size, num_classes)
        assert torch.equal(n["obj_t"], o["obj"]), f"obj mismatch stride {stride}"
        assert torch.equal(n["box_t"], o["box"]), f"box mismatch stride {stride}"
        assert torch.equal(n["gt_box_xyxy"], o["gt"]), f"gt mismatch stride {stride}"
        assert torch.equal(n["gt_class"].float(), o["cls"].float()), f"cls mismatch stride {stride}"


def test_assign_level_empty():
    boxes = torch.zeros(2, 8, 4)
    mask = torch.zeros(2, 8)
    classes = torch.zeros(2, 8, dtype=torch.long)
    out = new_assign(boxes, classes, mask, 4, 64, 3)
    assert out["obj_t"].sum() == 0
    assert out["box_t"].sum() == 0
    assert out["cls_t"].sum() == 0