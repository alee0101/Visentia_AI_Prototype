"""Fast checks for seg post-processing and the predict() resize contract."""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from visentia_te.imaging.seg import PostProc, dice, postprocess, predict


def test_threshold_and_min_size() -> None:
    p = np.zeros((40, 40), np.float32)
    p[2:12, 2:12] = 0.9  # 100 px blob
    p[30:33, 30:33] = 0.8  # 9 px blob
    p[20, 20] = 0.4  # below threshold
    m = postprocess(p, 0.5, 0)
    assert m.dtype == np.uint8 and set(np.unique(m)) == {0, 1} and m.sum() == 109
    assert postprocess(p, 0.5, 10).sum() == 100  # small blob removed
    assert postprocess(p, 0.5, 101).sum() == 0
    assert postprocess(p, 0.3, 0)[20, 20] == 1
    assert np.array_equal(PostProc(0.5, 10)(p), postprocess(p, 0.5, 10))


def test_diagonal_pixels_are_one_component() -> None:
    p = np.zeros((10, 10), np.float32)
    p[np.arange(6), np.arange(6)] = 1.0  # 8-connected diagonal of 6 px
    assert postprocess(p, 0.5, 6).sum() == 6


def test_dice_edge_cases() -> None:
    z = np.zeros((4, 4), np.uint8)
    assert np.isnan(dice(z, z))
    o = np.ones((4, 4), np.uint8)
    assert dice(o, o) == 1.0 and dice(o, z) == 0.0


def test_predict_returns_input_shape() -> None:
    model = nn.Conv2d(1, 1, 3, padding=1)
    for shape in [(252, 256), (496, 523), (100, 80)]:
        out = predict(model, np.random.rand(*shape).astype(np.float32), size=(64, 64))
        assert out.shape == shape and 0 <= out.min() and out.max() <= 1
