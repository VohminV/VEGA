"""Regression tests for LR schedule (warmup + cosine) and config keys."""
import math

import numpy as np
import pytest

from tests.fake_config import make_cfg
from train import lr_schedule


def test_warmup_linear():
    for e in range(5):
        assert lr_schedule(e, 5, 100) == pytest.approx((e + 1) / 5)


def test_cosine_end_near_base():
    epochs, warmup = 100, 5
    last = epochs - 1
    val = lr_schedule(last, warmup, epochs)
    progress = (last - warmup) / (epochs - warmup)
    expected = 0.01 + 0.5 * (1 - 0.01) * (1 + math.cos(math.pi * progress))
    assert val == pytest.approx(expected, abs=1e-12)
    assert val < 0.02  # конец cosine близок к базовому 0.01


def test_cosine_reference_formula():
    # сверяем с аналитической формулой на произвольной точке
    epochs, warmup = 100, 5
    e = 52
    progress = (e - warmup) / (epochs - warmup)
    expected = 0.01 + 0.5 * (1 - 0.01) * (1 + math.cos(math.pi * progress))
    assert lr_schedule(e, warmup, epochs) == pytest.approx(expected, abs=1e-12)


def test_monotonic_after_warmup():
    epochs, warmup = 100, 5
    vals = [lr_schedule(e, warmup, epochs) for e in range(warmup, epochs)]
    assert all(b <= a for a, b in zip(vals, vals[1:]))


def test_lr_schedule_sane_for_zero_epochs():
    # деление на ноль не роняет
    assert np.isfinite(lr_schedule(0, 5, 0))


def test_config_has_clip_and_warmup_keys():
    raw = make_cfg()
    assert "grad_clip" in raw.train
    assert "warmup_epochs" in raw.train
    assert int(raw.train["grad_clip"]) > 0
    assert int(raw.train["warmup_epochs"]) >= 1