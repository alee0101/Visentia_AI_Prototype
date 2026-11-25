"""Data + engine plumbing for the demo UI (kept out of Streamlit so it is testable).

The UI replays one real TREX-DME eye visit by visit. At each visit the copilot sees what a
clinic would have: the OCT-derived CST (a selectable source), BCVA, the fluid classifier
output, and the interval actually in force (teacher-forced, so dates follow the real trial).
Visit dates are synthesized from trial weeks (the dataset is de-identified, no real dates).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from visentia_te.agent.packet import VisitPacket, step_and_pack
from visentia_te.engine.protocol import EyeState, Observation, load_protocol

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
SOURCES = {
    "geometric": ("Geometric (retina layer segmentation)", P / "oof_cst_geo_visits.csv"),
    "cnn": ("CNN regression", P / "oof_cst_visits.csv"),
    "trial": ("Trial-measured (device CST)", None),
}
ANCHOR = date(2025, 1, 6)  # synthetic date of visit 1 (a Monday)
DEFAULT_EYE = "0232TOS"  # flags, shortenings and 18 decisions: a representative demo eye
DEFAULT_K = 0.5  # flag when ±k·SD crosses a rule boundary (operating point from the replay sweep)
FLUID_COLOR = np.array([236, 70, 71], np.float32)  # red: fluid
ILM_COLOR, BM_COLOR = np.array([242, 166, 0]), np.array([0, 184, 148])  # yellow ILM, aqua BM


@dataclass
class VisitView:
    visit: int
    week: float
    visit_date: date
    cst_true: float
    cst_in: float
    cst_sd: float
    qc_ok: bool
    imaged: bool
    bcva: float | None
    trial_action: str | None
    trial_next_interval: float | None
    packet: VisitPacket


@lru_cache(maxsize=1)
def tables() -> dict[str, pd.DataFrame]:
    v = pd.read_csv(P / "trex_visits.csv")
    v = v[v.arm.isin(["TREX", "GILA"])].copy()
    bio = pd.read_csv(P / "oof_biomarkers_visits.csv")
    geo = pd.read_csv(P / "geo_cst_visits.csv")
    src = {k: pd.read_csv(f) for k, (_, f) in SOURCES.items() if f is not None}
    return dict(visits=v, bio=bio, geo=geo, **{f"src_{k}": d for k, d in src.items()})


def imaged_eyes() -> list[str]:
    t = tables()
    have = set(t["geo"].study_id)
    n = t["visits"].groupby("study_id").visit.count()
    # eyes with images and at least one post-loading visit (something to decide)
    return sorted(s for s in t["visits"].study_id.unique() if s in have and n[s] > 3)


def eye_summary() -> pd.DataFrame:
    t = tables()
    v = t["visits"][t["visits"].study_id.isin(imaged_eyes())]
    return (v.groupby("study_id")
            .agg(arm=("arm", "first"), visits=("visit", "count"), baseline_cst=("cst", "first"),
                 decisions=("action", lambda a: a.notna().sum()), max_interval=("interval_wk", "max"))
            .reset_index())


def run_eye(study_id: str, source: str = "geometric", k: float = DEFAULT_K, use_fluid: bool = True) -> list[VisitView]:
    """Replay the engine over one eye's visits; returns one VisitView (with packet) per visit."""
    t = tables()
    protocol = load_protocol(ROOT / "configs" / "protocol.yaml")
    v = t["visits"][t["visits"].study_id == study_id].sort_values("visit")
    bio = t["bio"][t["bio"].study_id == study_id].set_index("visit")
    srcdf = t.get(f"src_{source}")
    src = srcdf[srcdf.study_id == study_id].set_index("visit") if srcdf is not None else None
    st, out = EyeState(protocol), []
    for _, r in v.iterrows():
        if not np.isfinite(r.cst) and (src is None or r.visit not in src.index):
            continue
        if np.isfinite(r.prev_interval_wk):
            st.interval_wk = int(np.clip(round(r.prev_interval_wk / 2) * 2, 4, protocol["interval"]["max_wk"]))
        imaged = src is not None and r.visit in src.index and np.isfinite(src.loc[r.visit, "cst_pred"])
        if imaged:
            s = src.loc[r.visit]
            cst = float(s.cst_pred)
            sd_base = float(s.cst_sd_visit) if "cst_sd_visit" in s and np.isfinite(s.cst_sd_visit) else 24.3
            sd = sd_base * k / 1.5
            qc = bool(s.qc_ok) if "qc_ok" in s and pd.notna(s.qc_ok) else True
        else:
            cst, sd, qc = float(r.cst), 0.0, True
        b = bio.loc[r.visit] if r.visit in bio.index else None
        p_fluid = float(b.p_fluid_fovea) if (use_fluid and b is not None and np.isfinite(b.p_fluid_fovea)) else None
        fluid_probs = {"IRF": float(b.p_irf), "SRF": float(b.p_srf)} if b is not None else None
        week = float(r.week) if np.isfinite(r.week) else (out[-1].week + 4 if out else 0.0)
        vdate = ANCHOR + timedelta(days=int(round(week * 7)))
        obs = Observation(visit=int(r.visit), cst=cst, bcva=None if np.isnan(r.etdrs) else float(r.etdrs),
                          cst_sd=sd, p_fluid=p_fluid, image_quality_ok=qc)
        pk = step_and_pack(st, obs, study_id=study_id, visit_date=vdate, fluid_probs=fluid_probs)
        out.append(VisitView(visit=int(r.visit), week=week, visit_date=vdate, cst_true=float(r.cst), cst_in=cst, cst_sd=sd,
                             qc_ok=qc, imaged=imaged, bcva=obs.bcva, trial_action=r.action if isinstance(r.action, str) else None,
                             trial_next_interval=float(r.interval_wk) if np.isfinite(r.interval_wk) else None, packet=pk))
    return out


# ---- images ---------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _index() -> pd.DataFrame:
    from visentia_te.data.olives_images import load_index
    idx = load_index(P / "trex_images.csv", ROOT / "data" / "raw" / "olives_trex", require_files=False)
    return idx[["study_id", "visit", "slice", "png"]].drop_duplicates(["study_id", "visit", "slice"])


def bscan_path(study_id: str, visit: int, slice_: int) -> Path | None:
    i = _index()
    m = i[(i.study_id == study_id) & (i.visit == visit) & (i.slice == slice_)]
    p = Path(m.png.iloc[0]) if len(m) else None
    if p is not None and not p.is_absolute():
        p = ROOT / p
    return p if p is not None and p.exists() else None


@lru_cache(maxsize=1)
def _profiles():
    f = P / "geo_profiles.npz"
    return np.load(f) if f.exists() else None


def boundaries(study_id: str, visit: int, slice_: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(ILM rows, BM rows) per column for slices 20..28 from the layer model's saved profiles."""
    pr = _profiles()
    key = f"{study_id}|{visit}"
    if pr is None or key not in pr.files or not (20 <= slice_ <= 28):
        return None
    a = pr[key]
    return a[0, slice_ - 20], a[1, slice_ - 20]


@lru_cache(maxsize=1)
def _seg():
    import torch

    from visentia_te.imaging.seg import load_checkpoint
    f = ROOT / "checkpoints" / "seg_final.pt"
    if not f.exists():
        return None
    torch.set_num_threads(2)
    return load_checkpoint(f, "cpu")


def fluid_mask(img: np.ndarray) -> np.ndarray | None:
    s = _seg()
    if s is None:
        return None
    import torch

    from visentia_te.imaging.seg import postprocess, predict
    model, meta = s
    with torch.no_grad():
        prob = predict(model, img)
    return postprocess(prob, threshold=meta.get("threshold", 0.5), min_size=int(meta.get("min_size", 50)))


def overlay(img: np.ndarray, fluid: np.ndarray | None, bnd: tuple[np.ndarray, np.ndarray] | None,
            scale: int = 2) -> np.ndarray:
    """RGB uint8 render: grayscale B-scan, translucent fluid, ILM/BM polylines, upscaled."""
    rgb = np.repeat(img[..., None] * 255, 3, axis=2).astype(np.float32)
    if fluid is not None and fluid.any():
        m = fluid.astype(bool)
        rgb[m] = 0.45 * rgb[m] + 0.55 * FLUID_COLOR
    rgb = np.kron(rgb, np.ones((scale, scale, 1)))
    if bnd is not None:
        h = rgb.shape[0]
        for rows, col in zip(bnd, (ILM_COLOR, BM_COLOR)):
            xs = np.arange(len(rows) * scale)
            ys = np.repeat(rows, scale) * scale
            ok = np.isfinite(ys)
            for dy in (-1, 0):
                y = np.clip(ys[ok].astype(int) + dy, 0, h - 1)
                rgb[y, xs[ok]] = col
    return np.clip(rgb, 0, 255).astype(np.uint8)
