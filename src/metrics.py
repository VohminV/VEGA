"""Метрики детектора: precision, recall, mAP@0.5 и метрики мелких объектов."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .boxes import iou_matrix


def is_small_box(
    box_xyxy: np.ndarray,
    img_w: int,
    img_h: int,
    small_min_side_px: float,
    small_area_fraction: float,
) -> np.ndarray:
    """
    Возвращает True для маленьких боксов.

    Бокс считается маленьким, если выполняется хотя бы одно условие:

        min(width, height) <= small_min_side_px

    ИЛИ

        area / image_area <= small_area_fraction
    """
    if box_xyxy.shape[0] == 0:
        return np.zeros(
            0,
            dtype=bool,
        )

    w = np.maximum(
        box_xyxy[:, 2] - box_xyxy[:, 0],
        0.0,
    )

    h = np.maximum(
        box_xyxy[:, 3] - box_xyxy[:, 1],
        0.0,
    )

    min_side = np.minimum(
        w,
        h,
    )

    img_area = max(
        float(img_w * img_h),
        1.0,
    )

    frac = (
        w * h
        / img_area
    )

    return (
        (min_side <= small_min_side_px)
        | (frac <= small_area_fraction)
    )


def _empty_preds() -> np.ndarray:
    """Пустой массив predictions формы (0, 6)."""
    return np.zeros(
        (0, 6),
        dtype=np.float32,
    )


def _empty_gts() -> np.ndarray:
    """Пустой массив GT формы (0, 5)."""
    return np.zeros(
        (0, 5),
        dtype=np.float32,
    )


def _match(
    preds: np.ndarray,
    gts: np.ndarray,
    iou_threshold: float,
):
    """
    Жадное сопоставление prediction -> GT.

    preds:
        [x1, y1, x2, y2, confidence, class]

    gts:
        [x1, y1, x2, y2, class]

    Алгоритм:

    1. Predictions сортируются по confidence descending.
    2. Для каждого prediction рассматриваются GT того же класса.
    3. Из свободных GT выбирается лучший IoU.
    4. Prediction становится TP только если IoU >= threshold.
    5. Уже использованный GT больше не может быть назначен другому prediction.

    ВАЖНО:
    если лучший GT занят или имеет неправильный класс, поиск продолжается.
    """
    matched = np.zeros(
        preds.shape[0],
        dtype=bool,
    )

    gt_used = np.zeros(
        gts.shape[0],
        dtype=bool,
    )

    if (
        preds.shape[0] == 0
        or gts.shape[0] == 0
    ):
        return matched, gt_used

    ious = iou_matrix(
        preds[:, :4],
        gts[:, :4],
    )

    order = np.argsort(
        -preds[:, 4],
        kind="stable",
    )

    for pi in order:
        pred_class = int(
            preds[pi, 5]
        )

        # Только GT нужного класса.
        class_candidates = np.flatnonzero(
            gts[:, 4].astype(int)
            == pred_class
        )

        if class_candidates.size == 0:
            continue

        # Только свободные GT.
        free_candidates = class_candidates[
            ~gt_used[class_candidates]
        ]

        if free_candidates.size == 0:
            continue

        # Сортируем свободные GT по IoU.
        candidate_ious = ious[
            pi,
            free_candidates,
        ]

        candidate_order = np.argsort(
            -candidate_ious,
            kind="stable",
        )

        # Ищем первый свободный GT,
        # который реально проходит IoU threshold.
        for local_idx in candidate_order:
            gi = int(
                free_candidates[local_idx]
            )

            if (
                ious[pi, gi]
                >= iou_threshold
            ):
                matched[pi] = True
                gt_used[gi] = True
                break

    return matched, gt_used


def _pr_points(
    preds_list: list[np.ndarray],
    gts_list: list[np.ndarray],
    iou_threshold: float,
):
    """
    Собирает (score, TP) для PR-кривой.

    Matching выполняется отдельно для каждого изображения.

    Prediction из одного изображения никогда не может
    заматчиться с GT другого изображения.
    """
    scores: list[float] = []
    tps: list[bool] = []

    for img_preds, img_gts in zip(
        preds_list,
        gts_list,
    ):
        if img_preds.shape[0] == 0:
            continue

        matched, _ = _match(
            img_preds,
            img_gts,
            iou_threshold,
        )

        order = np.argsort(
            -img_preds[:, 4],
            kind="stable",
        )

        scores.extend(
            img_preds[
                order,
                4,
            ].astype(float).tolist()
        )

        tps.extend(
            matched[
                order
            ].tolist()
        )

    return (
        np.asarray(
            scores,
            dtype=np.float32,
        ),
        np.asarray(
            tps,
            dtype=bool,
        ),
    )


def _pr_curve_from_points(
    scores: np.ndarray,
    tps: np.ndarray,
    n_gt: int,
):
    """
    Строит PR-кривую.

    scores:
        confidence predictions.

    tps:
        True/False для каждого prediction.

    n_gt:
        Общее количество GT данного класса.
    """
    if scores.size == 0:
        return (
            np.array(
                [1.0],
                dtype=np.float64,
            ),
            np.array(
                [0.0],
                dtype=np.float64,
            ),
        )

    order = np.argsort(
        -scores,
        kind="stable",
    )

    sorted_tps = tps[order]

    cum_tp = np.cumsum(
        sorted_tps,
        dtype=np.int64,
    )

    cum_fp = np.cumsum(
        ~sorted_tps,
        dtype=np.int64,
    )

    precisions = (
        cum_tp
        / np.maximum(
            cum_tp + cum_fp,
            1,
        )
    ).astype(np.float64)

    if n_gt > 0:
        recalls = (
            cum_tp
            / float(n_gt)
        ).astype(np.float64)
    else:
        recalls = np.zeros(
            len(cum_tp),
            dtype=np.float64,
        )

    return (
        np.concatenate(
            [
                np.array(
                    [1.0],
                    dtype=np.float64,
                ),
                precisions,
            ]
        ),
        np.concatenate(
            [
                np.array(
                    [0.0],
                    dtype=np.float64,
                ),
                recalls,
            ]
        ),
    )


def _ap(
    precisions: np.ndarray,
    recalls: np.ndarray,
) -> float:
    """
    AP по 101-точечной интерполяции.

    Используется схема COCO-style 101-point interpolation
    на фиксированном IoU threshold.
    """
    if (
        recalls.size == 0
        or recalls[-1] <= 0.0
    ):
        return 0.0

    ps = precisions.copy()

    # Precision envelope.
    for i in range(
        len(ps) - 2,
        -1,
        -1,
    ):
        ps[i] = max(
            ps[i],
            ps[i + 1],
        )

    grid = np.linspace(
        0.0,
        1.0,
        101,
    )

    ap = 0.0

    for recall_target in grid:
        # Берём максимальную precision при recall >= target.
        valid = np.flatnonzero(
            recalls >= recall_target
        )

        if valid.size:
            ap += float(
                np.max(
                    ps[valid]
                )
            )

    return ap / 101.0


def _filter_predictions(
    preds: np.ndarray,
    num_classes: int,
    conf_threshold: float,
) -> np.ndarray:
    """Фильтрует predictions по confidence и valid class."""
    if preds.shape[0] == 0:
        return _empty_preds()

    if preds.shape[1] < 6:
        raise ValueError(
            "Predictions должны иметь форму (N, 6): "
            "[x1, y1, x2, y2, conf, cls]"
        )

    cls = preds[:, 5].astype(
        np.int64,
        copy=False,
    )

    keep = (
        np.isfinite(preds[:, 4])
        & (preds[:, 4] >= conf_threshold)
        & (cls >= 0)
        & (cls < num_classes)
    )

    return preds[keep]


def _filter_gts(
    gts: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Фильтрует GT с невалидным class id."""
    if gts.shape[0] == 0:
        return _empty_gts()

    if gts.shape[1] < 5:
        raise ValueError(
            "GT должны иметь форму (N, 5): "
            "[x1, y1, x2, y2, cls]"
        )

    cls = gts[:, 4].astype(
        np.int64,
        copy=False,
    )

    keep = (
        np.isfinite(gts[:, 4])
        & (cls >= 0)
        & (cls < num_classes)
    )

    return gts[keep]


def _safe_div(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """Безопасное деление для метрик."""
    if denominator <= 0:
        return 0.0

    return float(
        numerator / denominator
    )


def evaluate(
    preds_per_image: list[np.ndarray],
    gts_per_image: list[np.ndarray],
    img_shapes: list[tuple[int, int]],
    num_classes: int,
    iou_threshold: float,
    conf_threshold: float,
    small_min_side_px: float,
    small_area_fraction: float,
) -> dict:
    """
    Рассчитывает метрики.

    preds_per_image[i]:
        (N, 6)
        [x1, y1, x2, y2, conf, cls]

    gts_per_image[i]:
        (M, 5)
        [x1, y1, x2, y2, cls]

    img_shapes[i]:
        (img_w, img_h)

    Возвращает:

        precision
        recall
        mAP@0.5

        small_precision
        small_recall

        small_true_positives
        small_gt_objects

        gt_objects
        tp
        fp
        fn

        small_pred_objects
        small_pred_true_positives
        small_pred_false_positives

    Дополнительно:

        small_gt_tp
        small_gt_fn

    чтобы отдельно видеть качество именно на маленьких GT.
    """
    if len(preds_per_image) != len(
        gts_per_image
    ):
        raise ValueError(
            "preds_per_image и gts_per_image "
            "должны иметь одинаковую длину"
        )

    if len(preds_per_image) != len(
        img_shapes
    ):
        raise ValueError(
            "preds_per_image и img_shapes "
            "должны иметь одинаковую длину"
        )

    if num_classes <= 0:
        raise ValueError(
            "num_classes должен быть > 0"
        )

    # ---------------------------------------------------------
    # Сначала нормализуем вход.
    # ---------------------------------------------------------
    filtered_preds: list[np.ndarray] = []
    filtered_gts: list[np.ndarray] = []

    for preds, gts in zip(
        preds_per_image,
        gts_per_image,
    ):
        filtered_preds.append(
            _filter_predictions(
                preds,
                num_classes,
                conf_threshold,
            )
        )

        filtered_gts.append(
            _filter_gts(
                gts,
                num_classes,
            )
        )

    # ---------------------------------------------------------
    # Общие TP / FP / FN.
    # ---------------------------------------------------------
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0

    # ---------------------------------------------------------
    # Метрики маленьких GT.
    #
    # Это основной показатель для твоей задачи.
    #
    # small_gt_tp:
    #   маленький GT был найден prediction'ом.
    #
    # small_gt_fn:
    #   маленький GT не был найден.
    # ---------------------------------------------------------
    small_gt_tp = 0
    small_gt_fn = 0

    # ---------------------------------------------------------
    # Метрики маленьких predictions.
    #
    # Это отдельный диагностический показатель:
    # сколько FP создаёт именно детектор на маленьких bbox.
    # ---------------------------------------------------------
    small_pred_tp = 0
    small_pred_fp = 0
    small_pred_objects = 0

    # ---------------------------------------------------------
    # AP по классам.
    # ---------------------------------------------------------
    ap_list: list[float] = []
    classes_with_gt: list[bool] = []

    for c in range(num_classes):
        cat_preds: list[np.ndarray] = []
        cat_gts: list[np.ndarray] = []

        for preds, gts in zip(
            filtered_preds,
            filtered_gts,
        ):
            if preds.shape[0]:
                pred_cls = (
                    preds[:, 5].astype(
                        np.int64,
                        copy=False,
                    )
                )

                cat_preds.append(
                    preds[
                        pred_cls == c
                    ]
                )
            else:
                cat_preds.append(
                    _empty_preds()
                )

            if gts.shape[0]:
                gt_cls = (
                    gts[:, 4].astype(
                        np.int64,
                        copy=False,
                    )
                )

                cat_gts.append(
                    gts[
                        gt_cls == c
                    ]
                )
            else:
                cat_gts.append(
                    _empty_gts()
                )

        n_gt_class = sum(
            g.shape[0]
            for g in cat_gts
        )

        has_gt = (
            n_gt_class > 0
        )

        classes_with_gt.append(
            has_gt
        )

        scores, tps = _pr_points(
            cat_preds,
            cat_gts,
            iou_threshold,
        )

        precision_curve, recall_curve = (
            _pr_curve_from_points(
                scores,
                tps,
                n_gt_class,
            )
        )

        ap = _ap(
            precision_curve,
            recall_curve,
        )

        ap_list.append(ap)

    # ---------------------------------------------------------
    # Image-level matching.
    # ---------------------------------------------------------
    for (
        preds,
        gts,
        shape,
    ) in zip(
        filtered_preds,
        filtered_gts,
        img_shapes,
    ):
        img_w, img_h = shape

        if gts.shape[0] == 0:
            # Нет GT.
            #
            # Любое prediction является FP.
            total_fp += preds.shape[0]

            if preds.shape[0]:
                pred_small = is_small_box(
                    preds[:, :4],
                    img_w,
                    img_h,
                    small_min_side_px,
                    small_area_fraction,
                )

                small_pred_objects += int(
                    pred_small.sum()
                )

                small_pred_fp += int(
                    pred_small.sum()
                )

            continue

        total_gt += gts.shape[0]

        matched, gt_used = _match(
            preds,
            gts,
            iou_threshold,
        )

        tp_count = int(
            matched.sum()
        )

        fp_count = int(
            (~matched).sum()
        )

        fn_count = int(
            (~gt_used).sum()
        )

        total_tp += tp_count
        total_fp += fp_count
        total_fn += fn_count

        # -----------------------------------------------------
        # Маленькие GT.
        # -----------------------------------------------------
        gt_small = is_small_box(
            gts[:, :4],
            img_w,
            img_h,
            small_min_side_px,
            small_area_fraction,
        )

        small_gt_tp += int(
            np.sum(
                gt_used
                & gt_small
            )
        )

        small_gt_fn += int(
            np.sum(
                (~gt_used)
                & gt_small
            )
        )

        # -----------------------------------------------------
        # Маленькие predictions.
        # -----------------------------------------------------
        pred_small = is_small_box(
            preds[:, :4],
            img_w,
            img_h,
            small_min_side_px,
            small_area_fraction,
        )

        small_pred_objects += int(
            pred_small.sum()
        )

        small_pred_tp += int(
            np.sum(
                matched
                & pred_small
            )
        )

        small_pred_fp += int(
            np.sum(
                (~matched)
                & pred_small
            )
        )

    # ---------------------------------------------------------
    # mAP.
    #
    # Классы без GT не должны искусственно занижать mAP.
    # ---------------------------------------------------------
    valid_aps = [
        ap
        for ap, has_gt in zip(
            ap_list,
            classes_with_gt,
        )
        if has_gt
    ]

    mAP = (
        float(
            np.mean(valid_aps)
        )
        if valid_aps
        else 0.0
    )

    # ---------------------------------------------------------
    # Общие precision / recall.
    # ---------------------------------------------------------
    precision = _safe_div(
        total_tp,
        total_tp + total_fp,
    )

    recall = _safe_div(
        total_tp,
        total_tp + total_fn,
    )

    # ---------------------------------------------------------
    # Основные small-object метрики.
    #
    # small_recall:
    #   главный показатель для твоей задачи.
    #
    # small_precision:
    #   precision среди predictions, которые сами
    #   выглядят как маленькие.
    # ---------------------------------------------------------
    small_recall = _safe_div(
        small_gt_tp,
        small_gt_tp + small_gt_fn,
    )

    small_precision = _safe_div(
        small_pred_tp,
        small_pred_tp + small_pred_fp,
    )

    return {
        "precision": precision,
        "recall": recall,
        "mAP@0.5": mAP,

        "small_precision": small_precision,
        "small_recall": small_recall,

        "small_true_positives": int(
            small_gt_tp
        ),

        "small_gt_objects": int(
            small_gt_tp + small_gt_fn
        ),

        "small_gt_tp": int(
            small_gt_tp
        ),

        "small_gt_fn": int(
            small_gt_fn
        ),

        "small_pred_objects": int(
            small_pred_objects
        ),

        "small_pred_true_positives": int(
            small_pred_tp
        ),

        "small_pred_false_positives": int(
            small_pred_fp
        ),

        "gt_objects": int(
            total_gt
        ),

        "tp": int(
            total_tp
        ),

        "fp": int(
            total_fp
        ),

        "fn": int(
            total_fn
        ),
    }


def save_metrics(
    metrics: dict,
    out_dir: Path,
) -> None:
    """Сохраняет метрики в JSON и TXT."""
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    lines = [
        f"precision: {metrics['precision']:.4f}",
        f"recall: {metrics['recall']:.4f}",
        f"mAP@0.5: {metrics['mAP@0.5']:.4f}",
        "",
        f"small_precision: {metrics['small_precision']:.4f}",
        f"small_recall: {metrics['small_recall']:.4f}",
        "",
        f"small_tp: {metrics['small_true_positives']}",
        f"small_gt_objects: {metrics['small_gt_objects']}",
        f"small_gt_tp: {metrics['small_gt_tp']}",
        f"small_gt_fn: {metrics['small_gt_fn']}",
        "",
        f"small_pred_objects: {metrics['small_pred_objects']}",
        (
            "small_pred_true_positives: "
            f"{metrics['small_pred_true_positives']}"
        ),
        (
            "small_pred_false_positives: "
            f"{metrics['small_pred_false_positives']}"
        ),
        "",
        f"gt_objects: {metrics['gt_objects']}",
        f"tp: {metrics['tp']}",
        f"fp: {metrics['fp']}",
        f"fn: {metrics['fn']}",
    ]

    with open(
        out_dir / "metrics.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(lines)
            + "\n"
        )