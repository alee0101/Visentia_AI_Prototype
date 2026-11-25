import numpy as np
import pytest

from visentia_te.imaging.layers import CANVAS, csf_weights, thickness_profile


def test_thickness_profile_band_and_speck():
    p = np.zeros((CANVAS, CANVAS), np.float32)
    p[100:140, :] = 0.9  # retina band, 40 rows
    p[10:13, 50:53] = 0.9  # vitreous speck must be ignored (not largest component)
    top, bot, th = thickness_profile(p)
    assert np.all(top == 100) and np.all(bot == 140) and np.all(th == 40)


def test_thickness_counts_hyporeflective_cyst():
    p = np.zeros((CANVAS, CANVAS), np.float32)
    p[100:160, :] = 0.9
    p[120:140, 100:130] = 0.1  # under-segmented fluid pocket inside the retina
    _, _, th = thickness_profile(p)
    assert th[110] == 60


def test_empty_columns_are_nan():
    p = np.zeros((CANVAS, CANVAS), np.float32)
    p[100:140, :128] = 0.9
    _, _, th = thickness_profile(p)
    assert np.isnan(th[200]) and th[10] == 40


def test_csf_circle_geometry():
    w = csf_weights(np.arange(-4, 5))
    # centre slice spans the full 1 mm (~43 px at 23.4 µm/px); slices beyond 500 µm are empty
    assert w[4].sum() == pytest.approx(1000 / (6000 / 256), abs=2)
    assert w[0].sum() == 0 and w[8].sum() == 0
    assert w[2].sum() < w[4].sum()
