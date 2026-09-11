"""Regression tests for NMS (boxes.nms, tiled_inference.final_nms)."""
import numpy as np
import pytest

from src.boxes import nms
from src.tiled_inference import final_nms


@pytest.mark.parametrize("class_aware", [False, True])
def test_nms_suppresses_duplicate(class_aware):
    boxes = np.array(
        [
            [10, 10, 50, 50],
            [12, 12, 52, 52],
            [100, 100, 140, 140],
            [102, 102, 142, 142],
            [30, 200, 60, 230],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.85, 0.8, 0.7, 0.6], dtype=np.float32)
    classes = np.zeros(5, dtype=np.int64)
    keep = nms(boxes, scores, 0.5, classes=(classes if class_aware else None))
    assert len(keep) == 3
    assert len(set(keep)) == len(keep)


def test_nms_keeps_low_iou():
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [20, 0, 30, 10],
            [0, 20, 10, 30],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, 0.5)
    assert keep == [0, 1, 2]


def test_nms_empty_returns_empty():
    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros(0, dtype=np.float32)
    assert nms(boxes, scores, 0.5) == []
    assert final_nms(np.zeros((0, 6), dtype=np.float32), 0.5).shape[0] == 0


def test_class_aware_keeps_boxes_of_different_classes():
    boxes = np.array(
        [
            [100, 100, 140, 140],
            [105, 105, 145, 145],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.6], dtype=np.float32)
    classes = np.array([0, 1], dtype=np.int64)
    keep = nms(boxes, scores, 0.5, classes=classes)
    assert len(keep) == 2


def test_final_nms_respects_candidate_cap():
    n = 5000
    boxes = np.random.rand(n, 4) * 100
    preds = np.concatenate([boxes, np.random.rand(n, 1) * 0.4 + 0.6, np.zeros((n, 1))], axis=1)
    out = final_nms(preds, 0.5)
    assert out.shape[1] == 6
    assert out.shape[0] > 0
    assert out.shape[0] <= 4096


def test_final_nms_sorts_by_confidence():
    preds = np.array(
        [
            [0, 0, 50, 50, 0.6, 0],
            [5, 5, 55, 55, 0.95, 0],
            [200, 200, 250, 250, 0.7, 0],
        ],
        dtype=np.float32,
    )
    out = final_nms(preds, 0.5)
    assert out[0, 4] == pytest.approx(0.95)