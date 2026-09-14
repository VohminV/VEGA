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
    """Focal binary cross entropy для objectness."""
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    pt = target * p + (1.0 - target) * (1.0 - p)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal = torch.pow(torch.clamp(1.0 - pt, min=0.0), gamma)
    return (alpha_t * focal * bce).mean()


def focal_bce_balanced(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    pos_weight: float = 0.75,
) -> torch.Tensor:
    """
    Focal-BCE с раздельной нормализацией positive/negative.

    Для детектора мелких объектов positive обычно очень мало,
    поэтому positive и negative нормализуются отдельно.

    pos_weight:
        0.75 -> 75% веса positive, 25% negative.

    ВАЖНО:
    pos_weight должен быть строго в (0, 1).

    pos_weight == 1.0 отключает negative терм целиком
    (0.0 * l_neg), и объектность вырождается: модели выгодно
    предсказывать "объект" во всех клетках, т.к. штраф за
    фон полностью отсутствует. Отсюда obj -> 1e-5 при
    precision -> ~0.01: один и тот же деградированный режим
    с двух сторон.

    pos_weight == 0.0, наоборот, игнорирует positive терм.
    """
    p = torch.sigmoid(logits)

    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )

    pt = target * p + (1.0 - target) * (1.0 - p)

    alpha_t = (
        alpha * target
        + (1.0 - alpha) * (1.0 - target)
    )

    focal = torch.pow(
        torch.clamp(1.0 - pt, min=0.0),
        gamma,
    )

    loss_el = alpha_t * focal * bce

    pos = target > 0.5

    n_pos = int(pos.sum().item())
    n_neg = int(pos.numel() - n_pos)

    if n_pos > 0:
        l_pos = loss_el[pos].mean()
    else:
        l_pos = torch.zeros(
            (),
            device=logits.device,
            dtype=loss_el.dtype,
        )

    if n_neg > 0:
        l_neg = loss_el[~pos].mean()
    else:
        l_neg = torch.zeros(
            (),
            device=logits.device,
            dtype=loss_el.dtype,
        )

    pos_weight = float(pos_weight)

    if not 0.0 < pos_weight < 1.0:
        raise ValueError(
            "focal_bce_balanced: pos_weight должен быть строго "
            f"в (0, 1), получено {pos_weight}. "
            "pos_weight = 1.0 отключает negative loss и приводит "
            "к вырождению objectness ('объект везде', obj -> 0, "
            "precision -> ~0.01). Используй значение < 1.0, "
            "например 0.75."
        )

    return (
        pos_weight * l_pos
        + (1.0 - pos_weight) * l_neg
    )


def _giou_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
) -> torch.Tensor:
    """
    GIoU loss.

    pred/tgt:
        (N, 4), xyxy.

    Возвращает скаляр.
    При N == 0 возвращает 0.
    """
    if pred.shape[0] == 0:
        return torch.zeros(
            (),
            device=pred.device,
            dtype=torch.float32,
        )

    # Безопаснее выполнять геометрию в FP32 при AMP.
    pred = pred.float()
    tgt = tgt.float()

    inter_x1 = torch.maximum(
        pred[:, 0],
        tgt[:, 0],
    )
    inter_y1 = torch.maximum(
        pred[:, 1],
        tgt[:, 1],
    )
    inter_x2 = torch.minimum(
        pred[:, 2],
        tgt[:, 2],
    )
    inter_y2 = torch.minimum(
        pred[:, 3],
        tgt[:, 3],
    )

    inter_w = torch.clamp(
        inter_x2 - inter_x1,
        min=0.0,
    )
    inter_h = torch.clamp(
        inter_y2 - inter_y1,
        min=0.0,
    )

    inter = inter_w * inter_h

    p_w = torch.clamp(
        pred[:, 2] - pred[:, 0],
        min=0.0,
    )
    p_h = torch.clamp(
        pred[:, 3] - pred[:, 1],
        min=0.0,
    )

    t_w = torch.clamp(
        tgt[:, 2] - tgt[:, 0],
        min=0.0,
    )
    t_h = torch.clamp(
        tgt[:, 3] - tgt[:, 1],
        min=0.0,
    )

    p_area = p_w * p_h
    t_area = t_w * t_h

    union = p_area + t_area - inter
    union = torch.clamp(union, min=1e-7)

    iou = inter / union

    enclose_x1 = torch.minimum(
        pred[:, 0],
        tgt[:, 0],
    )
    enclose_y1 = torch.minimum(
        pred[:, 1],
        tgt[:, 1],
    )
    enclose_x2 = torch.maximum(
        pred[:, 2],
        tgt[:, 2],
    )
    enclose_y2 = torch.maximum(
        pred[:, 3],
        tgt[:, 3],
    )

    enclose_w = torch.clamp(
        enclose_x2 - enclose_x1,
        min=0.0,
    )
    enclose_h = torch.clamp(
        enclose_y2 - enclose_y1,
        min=0.0,
    )

    enclose_area = (
        enclose_w * enclose_h
    )

    enclose_area = torch.clamp(
        enclose_area,
        min=1e-7,
    )

    giou = (
        iou
        - (enclose_area - union)
        / enclose_area
    )

    return (1.0 - giou).mean()


def _eiou_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
) -> torch.Tensor:
    """
    EIoU loss.

    pred/tgt:
        (N, 4), xyxy.

    Возвращает скаляр.
    При N == 0 возвращает 0.
    """
    if pred.shape[0] == 0:
        return torch.zeros(
            (),
            device=pred.device,
            dtype=torch.float32,
        )

    pred = pred.float()
    tgt = tgt.float()

    inter_x1 = torch.maximum(
        pred[:, 0],
        tgt[:, 0],
    )
    inter_y1 = torch.maximum(
        pred[:, 1],
        tgt[:, 1],
    )
    inter_x2 = torch.minimum(
        pred[:, 2],
        tgt[:, 2],
    )
    inter_y2 = torch.minimum(
        pred[:, 3],
        tgt[:, 3],
    )

    inter_w = torch.clamp(
        inter_x2 - inter_x1,
        min=0.0,
    )
    inter_h = torch.clamp(
        inter_y2 - inter_y1,
        min=0.0,
    )

    inter = inter_w * inter_h

    p_w = torch.clamp(
        pred[:, 2] - pred[:, 0],
        min=0.0,
    )
    p_h = torch.clamp(
        pred[:, 3] - pred[:, 1],
        min=0.0,
    )

    t_w = torch.clamp(
        tgt[:, 2] - tgt[:, 0],
        min=0.0,
    )
    t_h = torch.clamp(
        tgt[:, 3] - tgt[:, 1],
        min=0.0,
    )

    p_area = p_w * p_h
    t_area = t_w * t_h

    union = p_area + t_area - inter
    union = torch.clamp(
        union,
        min=1e-7,
    )

    iou = inter / union

    enclose_x1 = torch.minimum(
        pred[:, 0],
        tgt[:, 0],
    )
    enclose_y1 = torch.minimum(
        pred[:, 1],
        tgt[:, 1],
    )
    enclose_x2 = torch.maximum(
        pred[:, 2],
        tgt[:, 2],
    )
    enclose_y2 = torch.maximum(
        pred[:, 3],
        tgt[:, 3],
    )

    cw = torch.clamp(
        enclose_x2 - enclose_x1,
        min=1e-7,
    )
    ch = torch.clamp(
        enclose_y2 - enclose_y1,
        min=1e-7,
    )

    c2 = cw * cw + ch * ch
    c2 = torch.clamp(
        c2,
        min=1e-7,
    )

    pcx = (
        pred[:, 0]
        + pred[:, 2]
    ) / 2.0

    pcy = (
        pred[:, 1]
        + pred[:, 3]
    ) / 2.0

    tcx = (
        tgt[:, 0]
        + tgt[:, 2]
    ) / 2.0

    tcy = (
        tgt[:, 1]
        + tgt[:, 3]
    ) / 2.0

    rho2 = (
        (pcx - tcx) ** 2
        + (pcy - tcy) ** 2
    )

    rw2 = (p_w - t_w) ** 2
    rh2 = (p_h - t_h) ** 2

    eiou = (
        iou
        - rho2 / c2
        - rw2 / torch.clamp(cw * cw, min=1e-7)
        - rh2 / torch.clamp(ch * ch, min=1e-7)
    )

    return (1.0 - eiou).mean()


def _assign_level(
    boxes: torch.Tensor,
    class_ids: torch.Tensor,
    mask: torch.Tensor,
    stride: int,
    input_size: int,
    num_classes: int,
    obj_iou: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Назначает цели для одного detection level.

    boxes:
        [B, N, 4] в координатах input_size.

    class_ids:
        [B, N].

    mask:
        [B, N].

    Для обычных объектов используется покрытие всех grid cells,
    пересекаемых bbox.

    Для маленьких объектов используется центральная клетка
    плюс соседние клетки, но без чрезмерного размножения
    positive assignment.
    """
    B, N, _ = boxes.shape

    H = input_size // stride
    W = input_size // stride

    device = boxes.device
    S = float(stride)

    obj_t = torch.zeros(
        B,
        1,
        H,
        W,
        device=device,
        dtype=torch.float32,
    )

    box_t = torch.zeros(
        B,
        4,
        H,
        W,
        device=device,
        dtype=torch.float32,
    )

    gt_box_xyxy = torch.zeros(
        B,
        4,
        H,
        W,
        device=device,
        dtype=torch.float32,
    )

    gt_class = torch.zeros(
        B,
        H,
        W,
        dtype=torch.long,
        device=device,
    )

    pos_mask = torch.zeros(
        B,
        1,
        H,
        W,
        device=device,
        dtype=torch.float32,
    )

    mask_np = (
        (mask > 0.5)
        .detach()
        .cpu()
        .numpy()
    )

    b_idx, j_idx = mask_np.nonzero()

    if b_idx.size == 0:
        cls_t = torch.zeros(
            B,
            num_classes,
            H,
            W,
            device=device,
            dtype=torch.float32,
        )

        return {
            "obj_t": obj_t,
            "box_t": box_t,
            "cls_t": cls_t,
            "gt_box_xyxy": gt_box_xyxy,
            "gt_class": gt_class,
            "pos_mask": pos_mask,
        }

    boxes_np = (
        boxes.detach()
        .cpu()
        .numpy()
    )

    class_np = (
        class_ids.detach()
        .cpu()
        .numpy()
    )

    x1 = boxes_np[
        b_idx,
        j_idx,
        0,
    ].astype(np.float64)

    y1 = boxes_np[
        b_idx,
        j_idx,
        1,
    ].astype(np.float64)

    x2 = boxes_np[
        b_idx,
        j_idx,
        2,
    ].astype(np.float64)

    y2 = boxes_np[
        b_idx,
        j_idx,
        3,
    ].astype(np.float64)

    cs = class_np[
        b_idx,
        j_idx,
    ].astype(np.int64)

    good = (
        (x2 > x1)
        & (y2 > y1)
        & np.isfinite(x1)
        & np.isfinite(y1)
        & np.isfinite(x2)
        & np.isfinite(y2)
    )

    x1 = x1[good]
    y1 = y1[good]
    x2 = x2[good]
    y2 = y2[good]

    b_idx = b_idx[good]
    j_idx = j_idx[good]
    cs = cs[good]

    if x1.size == 0:
        cls_t = torch.zeros(
            B,
            num_classes,
            H,
            W,
            device=device,
            dtype=torch.float32,
        )

        return {
            "obj_t": obj_t,
            "box_t": box_t,
            "cls_t": cls_t,
            "gt_box_xyxy": gt_box_xyxy,
            "gt_class": gt_class,
            "pos_mask": pos_mask,
        }

    # Ограничиваем координаты valid image area.
    x1 = np.clip(x1, 0.0, float(W * S))
    y1 = np.clip(y1, 0.0, float(H * S))
    x2 = np.clip(x2, 0.0, float(W * S))
    y2 = np.clip(y2, 0.0, float(H * S))

    good = (
        (x2 > x1)
        & (y2 > y1)
    )

    x1 = x1[good]
    y1 = y1[good]
    x2 = x2[good]
    y2 = y2[good]

    b_idx = b_idx[good]
    j_idx = j_idx[good]
    cs = cs[good]

    if x1.size == 0:
        cls_t = torch.zeros(
            B,
            num_classes,
            H,
            W,
            device=device,
            dtype=torch.float32,
        )

        return {
            "obj_t": obj_t,
            "box_t": box_t,
            "cls_t": cls_t,
            "gt_box_xyxy": gt_box_xyxy,
            "gt_class": gt_class,
            "pos_mask": pos_mask,
        }

    w = x2 - x1
    h = y2 - y1

    cx = (
        x1 + x2
    ) / 2.0

    cy = (
        y1 + y2
    ) / 2.0

    area = w * h

    def _clamp_i(
        v: np.ndarray,
        n: int,
    ) -> np.ndarray:
        return np.clip(
            v,
            0.0,
            float(n - 1),
        ).astype(np.int64)

    # Клетки, через которые проходит bbox.
    ix1 = _clamp_i(
        np.floor(x1 / S),
        W,
    )

    iy1 = _clamp_i(
        np.floor(y1 / S),
        H,
    )

    ix2 = _clamp_i(
        np.floor((x2 - 1e-6) / S),
        W,
    )

    iy2 = _clamp_i(
        np.floor((y2 - 1e-6) / S),
        H,
    )

    # Очень маленький объект.
    #
    # Если bbox меньше одной grid-cell хотя бы по одной оси,
    # не размазываем target по большому количеству клеток.
    small = (
        (w < S)
        | (h < S)
    )

    cell_b_parts: list[np.ndarray] = []
    cell_y_parts: list[np.ndarray] = []
    cell_x_parts: list[np.ndarray] = []
    cell_g_parts: list[np.ndarray] = []

    # ---------------------------------------------------------
    # Обычные объекты.
    # ---------------------------------------------------------
    normal = ~small

    normal_ids = np.nonzero(normal)[0]

    if normal_ids.size:
        for g in normal_ids:
            ys = np.arange(
                iy1[g],
                iy2[g] + 1,
                dtype=np.int64,
            )

            xs = np.arange(
                ix1[g],
                ix2[g] + 1,
                dtype=np.int64,
            )

            yy, xx = np.meshgrid(
                ys,
                xs,
                indexing="ij",
            )

            count = yy.size

            cell_b_parts.append(
                np.full(
                    count,
                    b_idx[g],
                    dtype=np.int64,
                )
            )

            cell_y_parts.append(
                yy.reshape(-1)
            )

            cell_x_parts.append(
                xx.reshape(-1)
            )

            cell_g_parts.append(
                np.full(
                    count,
                    g,
                    dtype=np.int64,
                )
            )

    # ---------------------------------------------------------
    # Маленькие объекты.
    #
    # Основная positive cell = cell центра объекта.
    #
    # Дополнительно разрешаем соседние клетки только если
    # центр действительно находится достаточно близко к границе.
    #
    # Это намного мягче старого варианта, который мог давать
    # до 13 positive cells для одного микро-бокса.
    # ---------------------------------------------------------
    small_ids = np.nonzero(small)[0]

    if small_ids.size:
        ixc = _clamp_i(
            np.floor(cx[small_ids] / S),
            W,
        )

        iyc = _clamp_i(
            np.floor(cy[small_ids] / S),
            H,
        )

        for local_idx, g in enumerate(small_ids):
            gx = int(ixc[local_idx])
            gy = int(iyc[local_idx])

            candidate_cells = [
                (gy, gx),
            ]

            # Расстояние центра до границ центральной cell.
            cell_x0 = gx * S
            cell_y0 = gy * S

            rel_x = cx[g] - cell_x0
            rel_y = cy[g] - cell_y0

            # Для маленьких объектов добавляем только ближайшего
            # соседа при нахождении центра около соответствующей
            # границы клетки.
            edge_threshold = 0.25 * S

            if rel_x < edge_threshold and gx > 0:
                candidate_cells.append(
                    (gy, gx - 1)
                )

            if (
                rel_x > S - edge_threshold
                and gx < W - 1
            ):
                candidate_cells.append(
                    (gy, gx + 1)
                )

            if rel_y < edge_threshold and gy > 0:
                candidate_cells.append(
                    (gy - 1, gx)
                )

            if (
                rel_y > S - edge_threshold
                and gy < H - 1
            ):
                candidate_cells.append(
                    (gy + 1, gx)
                )

            # Удаляем дубликаты.
            candidate_cells = list(
                dict.fromkeys(candidate_cells)
            )

            count = len(candidate_cells)

            cell_b_parts.append(
                np.full(
                    count,
                    b_idx[g],
                    dtype=np.int64,
                )
            )

            cell_y_parts.append(
                np.asarray(
                    [p[0] for p in candidate_cells],
                    dtype=np.int64,
                )
            )

            cell_x_parts.append(
                np.asarray(
                    [p[1] for p in candidate_cells],
                    dtype=np.int64,
                )
            )

            cell_g_parts.append(
                np.full(
                    count,
                    g,
                    dtype=np.int64,
                )
            )

    if not cell_b_parts:
        cls_t = torch.zeros(
            B,
            num_classes,
            H,
            W,
            device=device,
            dtype=torch.float32,
        )

        return {
            "obj_t": obj_t,
            "box_t": box_t,
            "cls_t": cls_t,
            "gt_box_xyxy": gt_box_xyxy,
            "gt_class": gt_class,
            "pos_mask": pos_mask,
        }

    cell_b = np.concatenate(
        cell_b_parts
    )

    cell_y = np.concatenate(
        cell_y_parts
    )

    cell_x = np.concatenate(
        cell_x_parts
    )

    cell_g = np.concatenate(
        cell_g_parts
    )

    # ---------------------------------------------------------
    # Если несколько GT претендуют на одну cell,
    # оставляем один target.
    #
    # На stride 4 предпочитаем маленький bbox:
    # мелкие объекты критичны.
    #
    # На более крупных strides предпочитаем объект с большей
    # площадью.
    # ---------------------------------------------------------
    if stride == 4:
        key = area[cell_g]
    else:
        key = -area[cell_g]

    order = np.lexsort(
        (
            cell_g,
            key,
        )
    )

    cells_sorted = np.stack(
        [
            cell_b,
            cell_y,
            cell_x,
        ],
        axis=1,
    )[order]

    _, first = np.unique(
        cells_sorted,
        axis=0,
        return_index=True,
    )

    win = order[first]

    wb = cell_b[win]
    wy = cell_y[win]
    wx = cell_x[win]
    wg = cell_g[win]

    dx = (
        cx[wg]
        - (wx + 0.5) * S
    ) / S

    dy = (
        cy[wg]
        - (wy + 0.5) * S
    ) / S

    dw = np.log(
        np.maximum(
            w[wg] / S,
            1e-4,
        )
    )

    dh = np.log(
        np.maximum(
            h[wg] / S,
            1e-4,
        )
    )

    w_cls = cs[wg]

    gt_x1 = x1[wg]
    gt_y1 = y1[wg]
    gt_x2 = x2[wg]
    gt_y2 = y2[wg]

    def _to_device(
        array: np.ndarray,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.from_numpy(
            np.ascontiguousarray(array)
        ).to(
            device=device,
            dtype=dtype,
        )

    t_b = _to_device(
        wb,
        torch.long,
    )

    t_y = _to_device(
        wy,
        torch.long,
    )

    t_x = _to_device(
        wx,
        torch.long,
    )

    pos_mask[
        t_b,
        0,
        t_y,
        t_x,
    ] = 1.0

    # ---------------------------------------------------------
    # Objectness target.
    # ---------------------------------------------------------
    if obj_iou:
        ax1 = (
            wx + 0.5
        ) * S - S / 2.0

        ay1 = (
            wy + 0.5
        ) * S - S / 2.0

        ax2 = ax1 + S
        ay2 = ay1 + S

        ix1_n = np.maximum(
            ax1,
            gt_x1,
        )

        iy1_n = np.maximum(
            ay1,
            gt_y1,
        )

        ix2_n = np.minimum(
            ax2,
            gt_x2,
        )

        iy2_n = np.minimum(
            ay2,
            gt_y2,
        )

        iw = np.maximum(
            0.0,
            ix2_n - ix1_n,
        )

        ih = np.maximum(
            0.0,
            iy2_n - iy1_n,
        )

        inter = iw * ih

        a_area = (
            (ax2 - ax1)
            * (ay2 - ay1)
        )

        g_area = (
            (gt_x2 - gt_x1)
            * (gt_y2 - gt_y1)
        )

        union = (
            a_area
            + g_area
            - inter
        )

        iou = np.where(
            union > 0.0,
            inter / np.maximum(
                union,
                1e-9,
            ),
            np.zeros_like(union),
        )

        obj_t[
            t_b,
            0,
            t_y,
            t_x,
        ] = _to_device(
            iou.astype(np.float32),
            torch.float32,
        )
    else:
        obj_t = pos_mask.clone()

    # ---------------------------------------------------------
    # Box regression target.
    # ---------------------------------------------------------
    box_t[
        t_b,
        0,
        t_y,
        t_x,
    ] = _to_device(
        dx.astype(np.float32),
        torch.float32,
    )

    box_t[
        t_b,
        1,
        t_y,
        t_x,
    ] = _to_device(
        dy.astype(np.float32),
        torch.float32,
    )

    box_t[
        t_b,
        2,
        t_y,
        t_x,
    ] = _to_device(
        dw.astype(np.float32),
        torch.float32,
    )

    box_t[
        t_b,
        3,
        t_y,
        t_x,
    ] = _to_device(
        dh.astype(np.float32),
        torch.float32,
    )

    # ---------------------------------------------------------
    # Ground-truth xyxy для IoU loss.
    # ---------------------------------------------------------
    gt_box_xyxy[
        t_b,
        0,
        t_y,
        t_x,
    ] = _to_device(
        gt_x1.astype(np.float32),
        torch.float32,
    )

    gt_box_xyxy[
        t_b,
        1,
        t_y,
        t_x,
    ] = _to_device(
        gt_y1.astype(np.float32),
        torch.float32,
    )

    gt_box_xyxy[
        t_b,
        2,
        t_y,
        t_x,
    ] = _to_device(
        gt_x2.astype(np.float32),
        torch.float32,
    )

    gt_box_xyxy[
        t_b,
        3,
        t_y,
        t_x,
    ] = _to_device(
        gt_y2.astype(np.float32),
        torch.float32,
    )

    # ---------------------------------------------------------
    # Class target.
    # ---------------------------------------------------------
    gt_class[
        t_b,
        t_y,
        t_x,
    ] = _to_device(
        w_cls,
        torch.long,
    )

    cls_t = torch.zeros(
        B,
        num_classes,
        H,
        W,
        device=device,
        dtype=torch.float32,
    )

    if num_classes > 1:
        one_hot = F.one_hot(
            gt_class,
            num_classes=num_classes,
        ).permute(
            0,
            3,
            1,
            2,
        ).float()

        cls_t = (
            one_hot
            * pos_mask
        )

    return {
        "obj_t": obj_t,
        "box_t": box_t,
        "cls_t": cls_t,
        "gt_box_xyxy": gt_box_xyxy,
        "gt_class": gt_class,
        "pos_mask": pos_mask,
    }


def detector_loss(
    model: CustomDetector,
    images: torch.Tensor,
    boxes: torch.Tensor,
    class_ids: torch.Tensor,
    mask: torch.Tensor,
    use_focal: bool = True,
    obj_iou: bool = False,
    box_loss: str = "giou",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    obj_pos_weight: float = 0.75,
    weights: dict | None = None,
) -> dict[str, torch.Tensor]:
    """
    Считает total/objectness/box/class losses.

    boxes:
        [B, N, 4] xyxy в координатах input_size.

    class_ids:
        [B, N].

    mask:
        [B, N].
    """
    obj_pos_weight = float(obj_pos_weight)

    if not 0.0 < obj_pos_weight < 1.0:
        raise ValueError(
            "detector_loss: obj_pos_weight должен быть строго в "
            f"(0, 1), получено {obj_pos_weight}. Значение 1.0 "
            "отключает negative objectness loss и приводит к "
            "вырождению модели ('объект везде')."
        )

    outputs = model(images)

    strides = model.strides
    num_classes = model.num_classes

    w_ = weights or {
        "obj": 1.0,
        "box": 1.0,
        "cls": 1.0,
    }

    totals = {
        "obj": torch.zeros(
            (),
            device=images.device,
            dtype=torch.float32,
        ),
        "box": torch.zeros(
            (),
            device=images.device,
            dtype=torch.float32,
        ),
        "cls": torch.zeros(
            (),
            device=images.device,
            dtype=torch.float32,
        ),
    }

    for out, stride in zip(
        outputs,
        strides,
    ):
        targets = _assign_level(
            boxes=boxes,
            class_ids=class_ids,
            mask=mask,
            stride=int(stride),
            input_size=int(model.input_size),
            num_classes=int(num_classes),
            obj_iou=obj_iou,
        )

        obj_logits = out["obj"]
        box_out = out["box"]

        # -----------------------------------------------------
        # Objectness.
        # -----------------------------------------------------
        if use_focal:
            obj_loss = focal_bce_balanced(
                obj_logits,
                targets["obj_t"],
                alpha=focal_alpha,
                gamma=focal_gamma,
                pos_weight=obj_pos_weight,
            )
        else:
            obj_loss = F.binary_cross_entropy_with_logits(
                obj_logits,
                targets["obj_t"],
            )

        totals["obj"] = (
            totals["obj"]
            + obj_loss
        )

        # -----------------------------------------------------
        # Box regression.
        # -----------------------------------------------------
        pos = (
            targets["pos_mask"]
            > 0.5
        )

        n_pos = int(
            pos.sum().item()
        )

        if n_pos > 0:
            pred_xyxy = _decode_reg(
                box_out,
                pos,
                int(stride),
            )

            pred_xyxy = pred_xyxy.view(
                -1,
                4,
            )

            tgt_xyxy = (
                targets["gt_box_xyxy"]
                .permute(
                    0,
                    2,
                    3,
                    1,
                )[pos.squeeze(1)]
                .view(
                    -1,
                    4,
                )
            )

            if box_loss.lower() == "eiou":
                level_box_loss = _eiou_loss(
                    pred_xyxy,
                    tgt_xyxy,
                )
            else:
                level_box_loss = _giou_loss(
                    pred_xyxy,
                    tgt_xyxy,
                )

            totals["box"] = (
                totals["box"]
                + level_box_loss
            )

        # -----------------------------------------------------
        # Classification.
        # -----------------------------------------------------
        if num_classes > 1:
            cls_logits = out["cls"]
            cls_t = targets["cls_t"]

            # Классы имеют смысл только в positive cells.
            cls_pos = (
                targets["pos_mask"]
                .expand(
                    -1,
                    num_classes,
                    -1,
                    -1,
                )
                > 0.5
            )

            if bool(cls_pos.any()):
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_logits[cls_pos],
                    cls_t[cls_pos],
                    reduction="mean",
                )
            else:
                cls_loss = torch.zeros(
                    (),
                    device=images.device,
                    dtype=torch.float32,
                )

            totals["cls"] = (
                totals["cls"]
                + cls_loss
            )

    n_levels = max(
        len(strides),
        1,
    )

    obj_mean = (
        totals["obj"]
        / float(n_levels)
    )

    box_mean = (
        totals["box"]
        / float(n_levels)
    )

    cls_mean = (
        totals["cls"]
        / float(n_levels)
    )

    total = (
        obj_mean
        * float(w_.get("obj", 1.0))
        + box_mean
        * float(w_.get("box", 1.0))
        + cls_mean
        * float(w_.get("cls", 1.0))
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
    """
    Декодирует:

        dx
        dy
        log(w / stride)
        log(h / stride)

    в xyxy.

    box_out:
        [B, 4, H, W]

    pos:
        [B, 1, H, W]
    """
    # Геометрию и exp выполняем в FP32.
    box_out = box_out.float()

    B, _, H, W = box_out.shape
    device = box_out.device

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

    iy = (
        iy.float()
        .unsqueeze(0)
        .expand(
            B,
            -1,
            -1,
        )
    )

    ix = (
        ix.float()
        .unsqueeze(0)
        .expand(
            B,
            -1,
            -1,
        )
    )

    dx = box_out[:, 0]
    dy = box_out[:, 1]

    # Ограничиваем ширину/высоту.
    #
    # Для твоих мелких объектов этого более чем достаточно:
    #
    # exp(-4) * stride
    # exp(+4) * stride
    #
    # при stride=16:
    #
    # 0.29 px ... 873 px
    #
    # Это существенно безопаснее, чем exp(8).
    dw = torch.clamp(
        box_out[:, 2],
        -4.0,
        4.0,
    )

    dh = torch.clamp(
        box_out[:, 3],
        -4.0,
        4.0,
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

    # ВАЖНО:
    # X ограничивается через W,
    # Y ограничивается через H.
    max_x = float(W * stride)
    max_y = float(H * stride)

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

    xyxy = torch.stack(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        dim=1,
    )

    # [B, 4, H, W]
    # ->
    # [B, H, W, 4]
    xyxy = xyxy.permute(
        0,
        2,
        3,
        1,
    )

    return xyxy[
        pos.squeeze(1)
    ]