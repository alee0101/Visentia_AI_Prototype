"""Geometric retinal thickness: segment the retina (ILM -> Bruch's membrane) and measure it.

Why: a CNN that regresses CST directly from a B-scan has ~24 µm of within-eye visit-to-visit
noise, about as wide as the protocol's 10% band. Thickness is a geometric quantity, so we
measure it the way the device does: find the inner (ILM) and outer (BM) boundaries and take
their distance, averaged over the ETDRS central subfield (1 mm diameter circle at the fovea).

Training data: Duke DME (Chiu 2015) manual ILM/BM boundaries, 110 frames x 2 graders.
Scale: OLIVES PNGs are 256 x 252 (W x H), downscaled from ~504 x 496 Spectralis originals, so
Duke frames (496 rows) are resized by the same factor (SCALE) with the aspect ratio kept and
padded/cropped into a 256 x 256 canvas. Axial spacing after resize is ~UM_PER_ROW; the final
µm calibration is fitted against device-reported CST on training folds anyway.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

from visentia_te.data.duke import AXIAL_UM_PER_PIXEL, BM_IDX, ILM_IDX, DukeSample

CANVAS = 256
SCALE = 252 / 496  # OLIVES rows / Spectralis native rows
UM_PER_ROW = AXIAL_UM_PER_PIXEL / SCALE  # ~7.6 µm per row at OLIVES resolution
# ETDRS central subfield: 1 mm diameter. OLIVES/TREX raster: 49 B-scans over ~6 mm, 256 px over ~6 mm.
SLICE_SPACING_UM = 6000 / 48
COL_SPACING_UM = 6000 / 256
CSF_RADIUS_UM = 500


def _resize(x: np.ndarray, hw: tuple[int, int], mode: str) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, None]
    kw = dict(align_corners=False, antialias=True) if mode == "bilinear" else {}
    return F.interpolate(t, size=hw, mode=mode, **kw)[0, 0].numpy()


def duke_item(s: DukeSample, grader: int = 1, rng: np.random.Generator | None = None):
    """(image, retina_mask, weight) on a CANVAS x CANVAS grid at OLIVES scale.

    weight is 0 on padded canvas regions and columns without a valid boundary.
    rng=None -> deterministic (centered, no augmentation).
    """
    lay = s.layers1 if grader == 1 else s.layers2
    ilm, bm = lay[ILM_IDX], lay[BM_IDX]
    valid = np.isfinite(ilm) & np.isfinite(bm)
    rows = np.arange(s.image.shape[0], dtype=np.float32)[:, None]
    mask = ((rows >= np.nan_to_num(ilm)) & (rows < np.nan_to_num(bm))).astype(np.float32)
    sy = SCALE * (rng.uniform(0.9, 1.1) if rng is not None else 1.0)
    sx = SCALE * (rng.uniform(0.85, 1.15) if rng is not None else 1.0)
    h, w = round(s.image.shape[0] * sy), round(s.image.shape[1] * sx)
    src = np.where(s.image >= 0.999, 0.0, s.image)  # Duke registration padding is white; OLIVES pads black
    img = _resize(src, (h, w), "bilinear")
    msk = _resize(mask, (h, w), "bilinear") > 0.5
    val = _resize(valid[None].repeat(s.image.shape[0], 0).astype(np.float32), (h, w), "nearest") > 0.5
    if rng is not None:
        img, msk, val = (a[:, ::-1] for a in (img, msk, val)) if rng.random() < 0.5 else (img, msk, val)
        img = np.clip(img * rng.uniform(0.8, 1.2) + rng.uniform(-0.05, 0.05), 0, 1) ** rng.uniform(0.7, 1.4)
        if rng.random() < 0.5:
            img = img * rng.gamma(20.0, 1 / 20.0, img.shape) + rng.normal(0, 0.02, img.shape)
        img = np.clip(img, 0, 1)
    out = [np.zeros((CANVAS, CANVAS), np.float32) for _ in range(3)]
    # vertical placement keeps the whole retina inside the canvas
    rr = np.flatnonzero(msk.any(1))
    lo, hi = (rr[0], rr[-1]) if rr.size else (0, h - 1)
    t_min, t_max = -int(lo) + 8, CANVAS - 9 - int(hi)  # retina rows land inside, 8 px margin
    centered = int(np.clip((CANVAS - h) // 2, min(t_min, t_max), max(t_min, t_max)))
    top = int(rng.integers(min(t_min, t_max), max(t_min, t_max) + 1)) if rng is not None else centered
    left = int(rng.integers(min(0, CANVAS - w), max(0, CANVAS - w) + 1)) if rng is not None else (CANVAS - w) // 2
    for o, a in zip(out, (img, msk.astype(np.float32), val.astype(np.float32))):
        ys, xs = slice(max(0, top), min(CANVAS, top + h)), slice(max(0, left), min(CANVAS, left + w))
        o[ys, xs] = a[ys.start - top : ys.stop - top, xs.start - left : xs.stop - left]
    return out[0], out[1], out[2]


def olives_canvas(img: np.ndarray) -> np.ndarray:
    """OLIVES PNG array (252 x 256, float [0,1]) -> CANVAS x CANVAS (zero-pad bottom)."""
    out = np.zeros((CANVAS, CANVAS), np.float32)
    h, w = min(CANVAS, img.shape[0]), min(CANVAS, img.shape[1])
    out[:h, :w] = img[:h, :w]
    return out


def masked_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).sum() / weight.sum().clamp(min=1)
    p = torch.sigmoid(logits) * weight
    dice = (2 * (p * target).sum() + 1) / (p.sum() + (target * weight).sum() + 1)
    return bce + (1 - dice)


@torch.no_grad()
def predict_prob(model, x: np.ndarray, device, batch: int = 32) -> np.ndarray:
    """Retina probability maps for a stack (N, H, W) with horizontal-flip TTA."""
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        t = torch.from_numpy(np.ascontiguousarray(x[i : i + batch]))[:, None].to(device)
        p = torch.sigmoid(model(t)) + torch.sigmoid(model(t.flip(-1))).flip(-1)
        out.append((p / 2)[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def thickness_profile(prob: np.ndarray, thr: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column (ilm_row, bm_row, thickness_rows) from one probability map.

    Uses the largest connected component of the thresholded mask (drops specks in the
    vitreous / choroid). Thickness = top-to-bottom extent of the retina band per column, so
    hyporeflective fluid pockets the net under-segments still count toward thickness.
    NaN for columns with no retina.
    """
    m = prob > thr
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    any_ = m.any(0)
    top = np.where(any_, m.argmax(0), np.nan).astype(np.float32)
    bot = np.where(any_, m.shape[0] - m[::-1].argmax(0), np.nan).astype(np.float32)
    return top, bot, bot - top


def csf_weights(slice_offsets: np.ndarray, n_cols: int = CANVAS, center_col: float | None = None) -> np.ndarray:
    """(n_slices, n_cols) boolean mask of pixels inside the 1 mm central subfield circle."""
    c = (n_cols - 1) / 2 if center_col is None else center_col
    y = np.asarray(slice_offsets, float)[:, None] * SLICE_SPACING_UM
    x = (np.arange(n_cols)[None] - c) * COL_SPACING_UM
    return (x**2 + y**2) <= CSF_RADIUS_UM**2
