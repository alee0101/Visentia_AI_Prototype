"""Quick checks for the Duke DME loader (skipped if the raw data is absent)."""

from __future__ import annotations

import numpy as np
import pytest

from visentia_te.data import duke

pytestmark = pytest.mark.skipif(not any(duke.RAW_DIR.rglob("Subject_*.mat")), reason="Duke data not downloaded")


@pytest.fixture(scope="module")
def samples() -> list[duke.DukeSample]:
    return duke.load_all()


def test_shapes_and_binary(samples: list[duke.DukeSample]) -> None:
    assert len(samples) == 110
    for s in samples:
        h, w = s.image.shape
        assert s.fluid1.shape == s.fluid2.shape == (h, w)
        assert s.ilm.shape == s.bm.shape == s.valid_cols.shape == (w,)
        assert s.layers1.shape == s.layers2.shape == (8, w)
        assert s.image.dtype == np.float32 and 0 <= s.image.min() and s.image.max() <= 1
        assert set(np.unique(s.fluid1)) <= {0, 1} and set(np.unique(s.fluid2)) <= {0, 1}
        assert s.col_range[1] - s.col_range[0] == w


def test_dataset_item(samples: list[duke.DukeSample]) -> None:
    ds = duke.DukeFluidDataset([1, 2], augment=True, samples=samples)
    assert len(ds) == 22
    img, mask, meta = ds[0]
    assert img.shape == mask.shape == (1, 256, 256)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert meta["mask_other"].shape == (1, 256, 256)


def test_folds_patient_disjoint(tmp_path) -> None:
    folds = duke.make_folds(path=tmp_path / "f.json")
    tests = [s for f in folds for s in f["test"]]
    assert sorted(tests) == duke.SUBJECT_IDS  # every subject tested exactly once
    for f in folds:
        assert not set(f["train"]) & set(f["test"])
        assert sorted(f["train"] + f["test"]) == duke.SUBJECT_IDS
    assert folds == duke.make_folds(path=None)  # deterministic
