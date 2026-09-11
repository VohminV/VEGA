"""Потери детектора и назначение целей (target assignment)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import CustomDetector


def focal_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal-binary cross entropy (для несбалансированной objectness)."""
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = target * p + (1 - target) * (1 - p)
    w = (alpha * target + (1 - alpha) * (1 - target)) * torch.pow(1 - pt, gamma)
    return (w * bce).mean()


def _giou_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """GIoU loss. Оба (N,4) xyxy. Возвращает скаляр (0, если N==0)."""
    if pred.shape[0] == 0:
        return torch.zeros((), device=pred.device)
    inter_x1 = torch.maximum(pred[:, 0], tgt[:, 0])
    inter_y1 = torch.maximum(pred[:, 1], tgt[:, 1])
    inter_x2 = torch.minimum(pred[:, 2], tgt[:, 2])
    inter_y2 = torch.minimum(pred[:, 3], tgt[:, 3])
    inter = torch.clamp(inter_x2 - inter_x1, min=0.0) * torch.clamp(inter_y2 - inter_y1, min=0.0)
    p_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    t_area = (tgt[:, 2] - tgt[:, 0]) * (tgt[:, 3] - tgt[:, 1])
    union = p_area + t_area - inter + 1e-9
    iou = inter / union
    enclose_x1 = torch.minimum(pred[:, 0], tgt[:, 0])
    enclose_y1 = torch.minimum(pred[:, 1], tgt[:, 1])
    enclose_x2 = torch.maximum(pred[:, 2], tgt[:, 2])
    enclose_y2 = torch.maximum(pred[:, 3], tgt[:, 3])
    enclose = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + 1e-9
    giou = iou - (enclose - union) / enclose
    return (1.0 - giou).mean()


def _assign_level(
    boxes: torch.Tensor,
    class_ids: torch.Tensor,
    mask: torch.Tensor,
    stride: int,
    input_size: int,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    """Назначает цели для одного уровня stride (векторизованно).

    boxes: (B, N, 4) xyxy в пикселях input_size. class_ids/k: (B,N), mask: (B,N).
    Возвращает obj_t (B,1,H,W), box_t (B,4,H,W), cls_t (B,C,H,W),
    gt_box_xyxy (B,4,H,W) и gt_class (B,H,W) только в позитивных ячейках.

    Правила: мелкий объект (хоть одна сторона < stride) получает ячейку центра
    плюс 4 соседа; объекты крупнее — все ячейки своего прямоугольника. При
    конфликте ячейки на уровне stride=4 побеждает более мелкий объект, на
    остальных уровнях — более крупный. Все ячейки-кандидаты собираются векторно,
    победитель выбирается сортировкой по площади (детерминированно, при равенстве
    площади — по порядку следования бокса; эквивалентно последовательной версии).
    """
    B, N, _ = boxes.shape
    H = W = input_size // stride
    device = boxes.device
    S = float(stride)

    obj_t = torch.zeros(B, 1, H, W, device=device)
    box_t = torch.zeros(B, 4, H, W, device=device)
    gt_box_xyxy = torch.zeros(B, 4, H, W, device=device)
    gt_class = torch.zeros(B, H, W, dtype=torch.long, device=device)

    try:
        b_idx, j_idx = np.nonzero((mask > 0.5).cpu().numpy())
    except TypeError:  # старый torch без возврата tuple
        b_idx_np = (mask > 0.5).cpu().numpy()
        b_idx, j_idx = b_idx_np.nonzero()
    if b_idx.size == 0:
        cls_t = torch.zeros(B, num_classes, H, W, device=device)
        return {
            "obj_t": obj_t, "box_t": box_t, "cls_t": cls_t,
            "gt_box_xyxy": gt_box_xyxy, "gt_class": gt_class,
        }

    xb = boxes.cpu().numpy()
    x1 = xb[b_idx, j_idx, 0].astype(np.float64)
    y1 = xb[b_idx, j_idx, 1].astype(np.float64)
    x2 = xb[b_idx, j_idx, 2].astype(np.float64)
    y2 = xb[b_idx, j_idx, 3].astype(np.float64)
    cs = class_ids.cpu().numpy()[b_idx, j_idx].astype(np.int64)

    good = (x2 > x1) & (y2 > y1)
    x1, y1, x2, y2 = (a[good] for a in (x1, y1, x2, y2))
    b_idx, j_idx, cs = b_idx[good], j_idx[good], cs[good]
    if x1.size == 0:
        cls_t = torch.zeros(B, num_classes, H, W, device=device)
        return {
            "obj_t": obj_t, "box_t": box_t, "cls_t": cls_t,
            "gt_box_xyxy": gt_box_xyxy, "gt_class": gt_class,
        }

    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = w * h
    m = x1.size

    def _clamp_i(v: np.ndarray, n: int) -> np.ndarray:
        return np.minimum(np.maximum(v, 0.0), float(n - 1)).astype(np.int64)

    ix1 = _clamp_i(np.floor(x1 / S), W)
    iy1 = _clamp_i(np.floor(y1 / S), H)
    ix2 = _clamp_i(np.floor((x2 - 1e-6) / S), W)
    iy2 = _clamp_i(np.floor((y2 - 1e-6) / S), H)
    small = (w < S) | (h < S)

    # --- ячейки прямоугольника боксов, полностью векторно ---
    # Не мелкие боксы занимают весь прямоугольник; мелкие — только его углы.
    ny = iy2 - iy1 + 1
    nx = ix2 - ix1 + 1
    sizes = ny * nx
    big = ~small
    ids_b = np.nonzero(big)[0]

    if ids_b.size:
        ny_b, nx_b, sizes_b = ny[big], nx[big], sizes[big]
        sizes_b_total = int(sizes_b.sum())
        y_len_cum = np.concatenate([[0], np.cumsum(ny_b)])
        y_off = np.arange(int(y_len_cum[-1])) - np.repeat(y_len_cum[:-1], ny_b)
        row_reps = np.repeat(nx_b, ny_b)  # nx_k повторов для каждой строки бокса k
        cell_y = np.repeat(iy1[big], sizes_b) + np.repeat(y_off, row_reps)
        cell_x = np.repeat(ix1[big], sizes_b) + (
            np.arange(sizes_b_total) - np.repeat(y_len_cum[:-1], sizes_b)
        ) % np.repeat(nx_b, sizes_b)
        cell_b = np.repeat(b_idx[big], sizes_b)
        cell_g = np.repeat(ids_b, sizes_b)
    else:
        cell_y = np.zeros(0, dtype=np.int64)
        cell_x = np.zeros(0, dtype=np.int64)
        cell_b = np.zeros(0, dtype=np.int64)
        cell_g = np.zeros(0, dtype=np.int64)

    if bool(small.any()):
        s_id = np.nonzero(small)[0]
        # углы прямоугольника мелких боксов
        c_y = np.concatenate([iy1[small], iy1[small], iy2[small], iy2[small]])
        c_x = np.concatenate([ix1[small], ix2[small], ix1[small], ix2[small]])
        cell_y = np.concatenate([cell_y, c_y])
        cell_x = np.concatenate([cell_x, c_x])
        cell_b = np.concatenate([cell_b] + [b_idx[small]] * 4)
        cell_g = np.concatenate([cell_g] + [s_id] * 4)
        # центр + 4 соседа ячейки центра для мелких объектов
        ixc = _clamp_i(np.floor((cx - 1e-6) / S), W)[small]
        iyc = _clamp_i(np.floor((cy - 1e-6) / S), H)[small]
        sb_y = np.concatenate([iyc, iyc - 1, iyc + 1, iyc, iyc])
        sb_x = np.concatenate([ixc, ixc, ixc, ixc - 1, ixc + 1])
        sb_b = np.concatenate([b_idx[s_id]] * 5)
        sb_g = np.concatenate([s_id] * 5)
        inside = (sb_y >= 0) & (sb_y < H) & (sb_x >= 0) & (sb_x < W)
        cell_y = np.concatenate([cell_y, sb_y[inside]])
        cell_x = np.concatenate([cell_x, sb_x[inside]])
        cell_b = np.concatenate([cell_b, sb_b[inside]])
        cell_g = np.concatenate([cell_g, sb_g[inside]])

    # --- победитель ячейки: меньше площадь на stride=4, иначе больше ---
    key = area[cell_g] if stride == 4 else -area[cell_g]
    order = np.lexsort((cell_g, key))
    cells_sorted = np.stack([cell_b, cell_y, cell_x], axis=1)[order]
    _, first = np.unique(cells_sorted, axis=0, return_index=True)
    win = order[first]

    wb = cell_b[win]
    wy = cell_y[win]
    wx = cell_x[win]
    wg = cell_g[win]
    del cell_b, cell_y, cell_x, cell_g

    dx = (cx[wg] - (wx + 0.5) * S) / S
    dy = (cy[wg] - (wy + 0.5) * S) / S
    dw = np.log(np.maximum(w[wg] / S, 1e-4))
    dh = np.log(np.maximum(h[wg] / S, 1e-4))
    w_cls = cs[wg]
    gt_x1, gt_y1, gt_x2, gt_y2 = x1[wg], y1[wg], x2[wg], y2[wg]

    def _to_device(a: np.ndarray, dtype) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)

    t_b = _to_device(wb, torch.long)
    t_y = _to_device(wy, torch.long)
    t_x = _to_device(wx, torch.long)

    obj_t[t_b, 0, t_y, t_x] = 1.0
    box_t[t_b, 0, t_y, t_x] = _to_device(dx, torch.float32)
    box_t[t_b, 1, t_y, t_x] = _to_device(dy, torch.float32)
    box_t[t_b, 2, t_y, t_x] = _to_device(dw, torch.float32)
    box_t[t_b, 3, t_y, t_x] = _to_device(dh, torch.float32)
    gt_box_xyxy[t_b, 0, t_y, t_x] = _to_device(gt_x1, torch.float32)
    gt_box_xyxy[t_b, 1, t_y, t_x] = _to_device(gt_y1, torch.float32)
    gt_box_xyxy[t_b, 2, t_y, t_x] = _to_device(gt_x2, torch.float32)
    gt_box_xyxy[t_b, 3, t_y, t_x] = _to_device(gt_y2, torch.float32)
    gt_class[t_b, t_y, t_x] = _to_device(w_cls, torch.long)

    cls_t = torch.zeros(B, num_classes, H, W, device=device)
    if num_classes > 1:
        one_hot = F.one_hot(gt_class, num_classes).permute(0, 3, 1, 2).float()
        cls_t = one_hot * obj_t
    return {
        "obj_t": obj_t,
        "box_t": box_t,
        "cls_t": cls_t,
        "gt_box_xyxy": gt_box_xyxy,
        "gt_class": gt_class,
    }


def detector_loss(
    model: CustomDetector,
    images: torch.Tensor,
    boxes: torch.Tensor,
    class_ids: torch.Tensor,
    mask: torch.Tensor,
    use_focal: bool = True,
) -> dict[str, torch.Tensor]:
    """Считает total/objectness/box/class потери для батча."""
    outputs = model(images)
    strides = model.strides
    num_classes = model.num_classes

    totals = {"obj": torch.zeros((), device=images.device),
              "box": torch.zeros((), device=images.device),
              "cls": torch.zeros((), device=images.device)}

    for out, stride in zip(outputs, strides):
        targets = _assign_level(boxes, class_ids, mask, stride, model.input_size, num_classes)
        obj_logits = out["obj"]            # (B,1,H,W)
        box_out = out["box"]               # (B,4,H,W)

        if use_focal:
            obj_loss = focal_bce(obj_logits, targets["obj_t"])
        else:
            obj_loss = F.binary_cross_entropy_with_logits(obj_logits, targets["obj_t"])
        totals["obj"] = totals["obj"] + obj_loss

        tgt_xyxy, pos = targets["gt_box_xyxy"], targets["obj_t"] > 0.5
        n_pos = int(pos.sum().item())
        if n_pos > 0:
            pred_xyxy = _decode_reg(box_out, pos, stride)
            pred_xyxy = pred_xyxy.view(-1, 4)
            tgt_xyxy = tgt_xyxy.permute(0, 2, 3, 1)[pos.squeeze(1)].view(-1, 4)
            totals["box"] = totals["box"] + _giou_loss(pred_xyxy, tgt_xyxy)

        if num_classes > 1:
            cls_logits = out["cls"]  # (B,C,H,W)
            cls_t = targets["cls_t"]
            cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_t, reduction="mean")
            totals["cls"] = totals["cls"] + cls_loss
        else:
            totals["cls"] = totals["cls"] + torch.zeros((), device=obj_logits.device)

    n_levels = len(strides)
    obj_mean = totals["obj"] / max(float(n_levels), 1.0)
    box_mean = totals["box"] / max(float(n_levels), 1.0)
    cls_mean = totals["cls"] / max(float(n_levels), 1.0)
    weights = {"obj": 1.0, "box": 1.0, "cls": 1.0}
    total = (
        obj_mean * weights["obj"]
        + box_mean * weights["box"]
        + cls_mean * weights["cls"]
    )
    return {
        "total": total,
        "objectness": obj_mean,
        "box": box_mean,
        "class": cls_mean,
    }


def _decode_reg(
    box_out: torch.Tensor,
    pos: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """Декодирует (dx,dy,dw,dh) из box_out в xyxy только для позитивных ячеек.

    Возвращает (P,4) xyxy для позитивных ячеек.
    """
    B, _, H, W = box_out.shape
    device = box_out.device
    iy, ix = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    iy = iy.float().expand(B, -1, -1)
    ix = ix.float().expand(B, -1, -1)
    dx = box_out[:, 0]
    dy = box_out[:, 1]
    dw = torch.clamp(box_out[:, 2], -8.0, 8.0)
    dh = torch.clamp(box_out[:, 3], -8.0, 8.0)
    cx = (ix + 0.5 + dx) * stride
    cy = (iy + 0.5 + dy) * stride
    w = torch.exp(dw) * stride
    h = torch.exp(dh) * stride
    x1 = torch.clamp(cx - w / 2.0, 0, box_out.shape[2] * stride)
    y1 = torch.clamp(cy - h / 2.0, 0, box_out.shape[2] * stride)
    x2 = torch.clamp(cx + w / 2.0, 0, box_out.shape[2] * stride)
    y2 = torch.clamp(cy + h / 2.0, 0, box_out.shape[2] * stride)
    xyxy = torch.stack([x1, y1, x2, y2], dim=1)  # (B,4,H,W)
    xyxy = xyxy.permute(0, 2, 3, 1)  # (B,H,W,4)
    return xyxy[pos.squeeze(1)]