"""Tests for augmentation: random scale/crop keeps normalized boxes consistent."""
import numpy as np
import pytest

from src.dataset import YoloDataset
from tests.fake_config import make_cfg


class _FakeDs(YoloDataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_classes = 1
        aug = cfg.augmentation
        self.flip_lr = 0.0
        self.flip_ud = 0.0
        self.bc = 0.0
        self.scale_crop = float(aug.get("scale_crop", 0.0))
        self.scale_min = float(aug.get("scale_min", 0.7))
        self.scale_max = float(aug.get("scale_max", 1.0))


def _make_ds(scale_crop=1.0, scale_min=0.7, scale_max=1.0) -> _FakeDs:
    cfg = make_cfg()
    cfg.augmentation["scale_crop"] = scale_crop
    cfg.augmentation["scale_min"] = scale_min
    cfg.augmentation["scale_max"] = scale_max
    return _FakeDs(cfg)


@pytest.mark.parametrize("seed", [1, 2, 7, 13])
def test_scale_crop_maps_center_and_size(seed, monkeypatch):
    import random

    ds = _make_ds(scale_min=0.7, scale_max=0.9)
    w, h = 320, 240
    s = 0.8
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    ox = 10
    oy = 20

    monkeypatch.setattr(random, "uniform", lambda a, b: s)
    calls = {"n": 0}
    def _randint(a, b):
        calls["n"] += 1
        return ox if calls["n"] == 1 else oy
    monkeypatch.setattr(random, "randint", _randint)

    img = np.zeros((h, w, 3), dtype=np.uint8)
    box = np.array([[0.0, 0.5, 0.5, 0.2, 0.1]], dtype=np.float32)
    out_img, out_box = ds._random_scale_crop(img, box.copy())

    assert out_img.shape == img.shape
    assert out_box[0, 1] == pytest.approx((0.5 - ox / w) / s, abs=1e-6)
    assert out_box[0, 2] == pytest.approx((0.5 - oy / h) / s, abs=1e-6)
    assert out_box[0, 3] == pytest.approx(0.2 / s, abs=1e-6)
    assert out_box[0, 4] == pytest.approx(0.1 / s, abs=1e-6)


def test_scale_crop_default_prob_zero():
    ds = _make_ds(scale_crop=0.0)
    assert ds.scale_crop == 0.0


def test_scale_crop_noop_on_scale_one(monkeypatch):
    import random

    ds = _make_ds(scale_min=1.0, scale_max=1.0)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)
    w, h = 100, 80
    img = np.zeros((h, w, 3), dtype=np.uint8)
    box = np.array([[0.0, 0.5, 0.5, 0.2, 0.1]], dtype=np.float32)
    out_img, out_box = ds._random_scale_crop(img, box.copy())
    assert out_img.shape == (h, w, 3)
    assert np.array_equal(out_box, box)


def test_scale_crop_preserves_finite_boxes(monkeypatch):
    import random

    ds = _make_ds(scale_min=0.7, scale_max=0.9)
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.8)
    monkeypatch.setattr(random, "randint", lambda a, b: 0)
    w, h = 128, 128
    img = np.zeros((h, w, 3), dtype=np.uint8)
    box = np.array([[0.0, 0.95, 0.5, 0.2, 0.1]], dtype=np.float32)
    out_img, out_box = ds._random_scale_crop(img, box.copy())
    assert out_box.shape == (1, 5)
    assert np.isfinite(out_box).all()


def test_augment_norm_box_collapse_after_clip(monkeypatch):
    """Бокс, схлопнувшийся после clip (выпал из кадра), не роняет broadcast."""
    import random

    ds = _make_ds(scale_crop=0.0)  # только flip/bc выключены => чистое схлопывание
    monkeypatch.setattr(random, "random", lambda: 1.0)  # без flip/bc
    w, h = 64, 64
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # 2 валидных + 1 далеко за кадром (cx>1): после clip схлопнётся
    boxes = np.array(
        [
            [0.0, 0.3, 0.3, 0.2, 0.2],
            [0.0, 0.6, 0.6, 0.15, 0.15],
            [0.0, 1.5, 0.5, 0.1, 0.1],  # x1=x2=1 после clip
        ],
        dtype=np.float32,
    )
    out_img, out_box = ds._augment_norm(img, boxes.copy())
    assert out_box.shape[0] == 2
    # центры-размеры пересчитаны от клипнутых xyxy, а не сломаны
    assert np.all(out_box[:, 1] >= 0.0) and np.all(out_box[:, 1] <= 1.0)
    assert np.all(out_box[:, 3] > 0.0)