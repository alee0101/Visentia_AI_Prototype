"""Loader for the Duke DME OCT segmentation dataset (Chiu et al., BOE 2015).

Dataset: "Kernel regression based segmentation of optical coherence tomography
images with diabetic macular edema", Biomed. Opt. Express 6(4):1172 (2015).
Source: http://www.duke.edu/~sf59/Datasets/2015_BOE_Chiu2.zip

Each ``Subject_XX.mat`` (MATLAB v5, scipy-loadable) holds a 61-B-scan
Spectralis volume ``images`` (496 x 768 x 61, uint8) plus float64 arrays
``manualFluid{1,2}`` / ``automaticFluidDME`` (496 x 768 x 61) and
``manualLayers{1,2}`` / ``automaticLayersDME`` / ``automaticLayersNormal``
(8 x 768 x 61). Two expert graders annotated 11 of the 61 B-scans (frame
indices differ per subject, clustered around the fovea). Unannotated frames are
all-NaN. Within annotated frames, layer boundaries are NaN outside a central
column range (~520-550 of 768 columns); fluid arrays are 0 (not NaN) there.
Fluid arrays store a per-region integer label (0 = background, 1..N = distinct
fluid pockets); we binarise with ``> 0``. Both graders share the same column
range and no fluid is marked outside it. Quirk: B-scans were registered, so
wedge-shaped padding regions at the top/bottom edges are filled with 255
(about 8-10% of pixels); treat 1.0 as "no data" if that matters downstream.

Cropping: every sample is cropped to the column range where *both* graders'
ILM and bottom boundaries are finite (the annotated region). Outside that range
there is no ground truth, so leaving it in would teach the model "no fluid"
where the truth is actually unknown. ``valid_cols`` marks columns inside the
crop that still have a NaN boundary (rare; kept so thickness can mask them).

Layer boundary rows (8 boundaries, top to bottom): index 0 = ILM, index 7 =
Bruch's membrane (outer RPE boundary). Values are 1-based MATLAB row indices;
we convert to 0-based.

Axial pixel spacing: not stored in the .mat files. Spectralis SD-OCT axial
sampling is ~3.87 um/pixel (assumption, consistent with Heidelberg spec;
lateral spacing varies with scan width and is not recoverable here).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "duke_dme"
FOLDS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "duke_folds.json"
SUBJECT_IDS: list[int] = list(range(1, 11))
AXIAL_UM_PER_PIXEL: float = 3.87  # assumption: Spectralis nominal axial sampling
ILM_IDX, BM_IDX = 0, 7


@dataclass
class DukeSample:
    """One annotated B-scan, cropped to the annotated column range."""

    image: np.ndarray  # (H, W) float32 in [0, 1]
    fluid1: np.ndarray  # (H, W) uint8 binary, grader 1
    fluid2: np.ndarray  # (H, W) uint8 binary, grader 2
    ilm: np.ndarray  # (W,) float32 row index (0-based), grader 1, NaN if missing
    bm: np.ndarray  # (W,) float32 bottom boundary (Bruch's membrane), grader 1
    layers1: np.ndarray  # (8, W) float32 all boundaries, grader 1
    layers2: np.ndarray  # (8, W) float32 all boundaries, grader 2
    valid_cols: np.ndarray  # (W,) bool, True where ILM and BM are finite (grader 1)
    subject: int
    frame: int  # 0-based index into the 61-frame volume
    col_range: tuple[int, int]  # [start, stop) columns in the original B-scan


def _subject_path(subject: int, raw_dir: Path) -> Path:
    hits = sorted(raw_dir.rglob(f"Subject_{subject:02d}.mat"))
    if not hits:
        raise FileNotFoundError(f"Subject_{subject:02d}.mat not found under {raw_dir}")
    return hits[0]


def load_subject(subject: int, raw_dir: Path = RAW_DIR) -> list[DukeSample]:
    """Load all manually annotated B-scans of one subject."""
    m = scipy.io.loadmat(_subject_path(subject, raw_dir))
    images = m["images"]
    f1, f2 = m["manualFluid1"].astype(np.float32), m["manualFluid2"].astype(np.float32)
    l1, l2 = m["manualLayers1"].astype(np.float32), m["manualLayers2"].astype(np.float32)
    samples: list[DukeSample] = []
    for k in range(images.shape[2]):
        if np.all(np.isnan(l1[:, :, k])) or np.all(np.isnan(f1[:, :, k])):
            continue  # frame not annotated
        both = np.isfinite(l1[[ILM_IDX, BM_IDX], :, k]).all(0) & np.isfinite(
            l2[[ILM_IDX, BM_IDX], :, k]
        ).all(0)
        cols = np.flatnonzero(both)
        if cols.size == 0:
            continue
        c0, c1 = int(cols[0]), int(cols[-1]) + 1
        lay1 = l1[:, c0:c1, k] - 1.0  # MATLAB 1-based -> 0-based
        lay2 = l2[:, c0:c1, k] - 1.0
        samples.append(
            DukeSample(
                image=images[:, c0:c1, k].astype(np.float32) / 255.0,
                fluid1=(np.nan_to_num(f1[:, c0:c1, k]) > 0).astype(np.uint8),
                fluid2=(np.nan_to_num(f2[:, c0:c1, k]) > 0).astype(np.uint8),
                ilm=lay1[ILM_IDX].copy(),
                bm=lay1[BM_IDX].copy(),
                layers1=lay1,
                layers2=lay2,
                valid_cols=np.isfinite(lay1[[ILM_IDX, BM_IDX]]).all(0),
                subject=subject,
                frame=k,
                col_range=(c0, c1),
            )
        )
    return samples


def load_all(subjects: list[int] | None = None, raw_dir: Path = RAW_DIR) -> list[DukeSample]:
    """Load annotated B-scans for the given subjects (default: all 10)."""
    out: list[DukeSample] = []
    for s in subjects or SUBJECT_IDS:
        out.extend(load_subject(s, raw_dir))
    return out


def retinal_thickness(ilm: np.ndarray, bm: np.ndarray) -> np.ndarray:
    """Per-column retinal thickness (ILM to bottom boundary) in pixels; NaN where missing.

    Multiply by ``AXIAL_UM_PER_PIXEL`` for an approximate value in micrometres.
    """
    return (bm - ilm).astype(np.float32)


def make_folds(
    subjects: list[int] = SUBJECT_IDS, n_folds: int = 5, seed: int = 0,
    path: Path | None = FOLDS_PATH,
) -> list[dict[str, list[int]]]:
    """Deterministic patient-level K-fold splits; optionally written to JSON."""
    order = np.random.default_rng(seed).permutation(subjects).tolist()
    folds = []
    for i in range(n_folds):
        test = sorted(int(s) for s in order[i::n_folds])
        folds.append({"fold": i, "train": sorted(int(s) for s in subjects if s not in test), "test": test})
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"seed": seed, "n_folds": n_folds, "folds": folds}, indent=2))
    return folds


def _augment(img: np.ndarray, masks: list[np.ndarray], rng: np.random.Generator):
    if rng.random() < 0.5:
        img, masks = img[:, ::-1], [m[:, ::-1] for m in masks]
    img = np.clip(img * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0, 1)
    img = img ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.5:  # multiplicative speckle + additive gaussian
        img = img * rng.gamma(20.0, 1 / 20.0, img.shape) + rng.normal(0, 0.02, img.shape)
    return np.clip(img, 0, 1).astype(np.float32), [np.ascontiguousarray(m) for m in masks]


class DukeFluidDataset(Dataset):
    """B-scan / fluid-mask pairs for a list of subjects.

    Returns ``(image[1,H,W] float32, mask[1,H,W] float32, meta)``. ``grader``
    selects the label (1 by default); the other grader's mask is always in
    ``meta["mask_other"]`` for inter-grader comparison. Images are bilinearly
    and masks nearest-neighbour resized to ``size`` (H, W); pass ``None`` to
    keep native resolution.
    """

    def __init__(
        self,
        subjects: list[int],
        size: tuple[int, int] | None = (256, 256),
        augment: bool = False,
        grader: int = 1,
        seed: int = 0,
        samples: list[DukeSample] | None = None,
        raw_dir: Path = RAW_DIR,
    ) -> None:
        assert grader in (1, 2)
        pool = samples if samples is not None else load_all(subjects, raw_dir)
        self.samples = [s for s in pool if s.subject in set(subjects)]
        self.size, self.augment, self.grader = size, augment, grader
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _resize(self, x: np.ndarray, mode: str) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, None]
        if self.size is not None:
            t = F.interpolate(t, size=self.size, mode=mode, **({"align_corners": False} if mode == "bilinear" else {}))
        return t[0]

    def __getitem__(self, i: int):
        s = self.samples[i]
        lab, oth = (s.fluid1, s.fluid2) if self.grader == 1 else (s.fluid2, s.fluid1)
        img, (lab, oth) = s.image, (lab, oth)
        if self.augment:
            img, (lab, oth) = _augment(img, [lab, oth], self.rng)
        meta = {
            "subject": s.subject, "frame": s.frame, "col_start": s.col_range[0],
            "col_stop": s.col_range[1], "mask_other": self._resize(oth, "nearest"),
        }
        return self._resize(img, "bilinear"), self._resize(lab, "nearest"), meta
