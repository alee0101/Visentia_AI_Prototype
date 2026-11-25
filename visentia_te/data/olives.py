"""Parse the OLIVES / TREX-DME trial spreadsheets into a tidy per-visit table.

Source: OLIVES_Dataset_Labels/full_labels/OCT-DME.xlsx (Zenodo 7105232, CC-BY-4.0).

Layout quirks this module handles:
  * Arm sheets ("TREX", "GILA", "Monthly") are wide: 27 visit blocks of
    [Visit Date, Week, ETDRS, Snellen, OCT]. Dates/weeks are blanked
    (de-identification). "OCT" is central retinal/subfield thickness in µm.
  * "TREX Vision" / "GILA Vision" sheets hold fractional weeks per visit with ETDRS,
    but no StudyID. We recover the mapping by aligning ETDRS sequences.
  * Treat-and-extend arms inject at every visit, so the gap between consecutive
    visits *is* the treatment interval chosen by the trial investigator. The
    action taken at visit k is therefore the sign of (interval_k+1 - interval_k).

StudyID encoding: "0105TOS" = site 01, subject 05, arm T(REX)/G(ILA)/M(onthly),
eye OS/OD. Image folders are "TREX DME/<arm>/<StudyID>/V<visitnum>/<eye>/".
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABELS_DIR = Path("data/raw/olives_labels/OLIVES_Dataset_Labels")
ARMS = ("TREX", "GILA", "Monthly")
TE_ARMS = ("TREX", "GILA")
FIELD_ROW = 3  # 0-based row with field names; data starts on the next row
LOADING_VISITS = 3  # visits 1-3 are pure loading; visit 4 (week 12, 4th dose) is the first decision
ACTION_TOL_WK = 1.25  # |Δinterval| below this is scheduling jitter -> MAINTAIN
MIN_PLAUSIBLE_WK, MAX_PLAUSIBLE_WK = 2.5, 20.0


def _num(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _arm_visits(raw: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Wide arm sheet -> long rows (study_id, visit, etdrs, cst) + per-eye summary cols."""
    fields = raw.iloc[FIELD_ROW].fillna("nan").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    # Visit number marks the first column of each block (row 1 or row 2 depending on sheet).
    starts: list[tuple[int, int]] = []
    for r in (1, 2):
        for c, v in raw.iloc[r].items():
            n = _num(v)
            if np.isfinite(n) and 1 <= n <= 40 and float(n).is_integer():
                starts.append((int(c), int(n)))
        if starts:
            break
    starts.sort()
    stop = next((c for c, f in fields.items() if c > starts[-1][0] and f in ("nan", "Comments")), raw.shape[1])
    bounds = [s for s, _ in starts] + [stop]

    def idx_of(name: str) -> int | None:
        m = [c for c, f in fields.items() if f.lower().startswith(name.lower())]
        return m[0] if m else None

    elig_col = idx_of("Eligible for extension")
    max_int_col = [c for c, f in fields.items() if f == "Maximal Treatment Interval"]
    rows = []
    for _, rec in raw.iloc[FIELD_ROW + 1 :].iterrows():
        sid = str(rec.iloc[0]).strip()
        if not (len(sid) == 7 and sid[:4].isdigit() and sid[-2:] in ("OD", "OS")):
            continue
        for (c0, vnum), c1 in zip(starts, bounds[1:]):
            block = {fields[c]: rec.iloc[c] for c in range(c0, c1)}
            et = next((_num(v) for k, v in block.items() if k.upper() == "ETDRS"), np.nan)
            cst = next((_num(v) for k, v in block.items() if k.upper() == "OCT"), np.nan)
            rows.append(
                dict(
                    study_id=sid,
                    arm=arm,
                    eye=sid[-2:],
                    visit=vnum,
                    etdrs=et,
                    cst=cst,
                    eligible_ext_wk12=_num(rec.iloc[elig_col]) if elig_col is not None else np.nan,
                    max_interval_wk=_num(rec.iloc[max_int_col[-1]]) if max_int_col else np.nan,
                )
            )
    df = pd.DataFrame(rows)
    return df[df.etdrs.notna() | df.cst.notna()].reset_index(drop=True)


def _vision_sheet(raw: pd.DataFrame) -> list[list[tuple[float, float]]]:
    """'X Vision' sheet -> per row list of (week, etdrs) in visit order."""
    out = []
    for _, rec in raw.iloc[1:].iterrows():
        vals = rec.tolist()
        seq = []
        # first pair is the 'Week 0' label + ETDRS; afterwards alternating week, etdrs
        if _num(vals[1]) == _num(vals[1]):
            seq.append((0.0, _num(vals[1])))
        for i in range(2, len(vals) - 1, 2):
            w, e = _num(vals[i]), _num(vals[i + 1])
            if np.isfinite(e):
                seq.append((w, e))
        if len(seq) >= 3:
            out.append(seq)
    return out


@dataclass
class AlignReport:
    study_id: str
    vision_row: int
    matched_visits: int
    total_visits: int
    match_ratio: float


def _attach_weeks(visits: pd.DataFrame, vision: list[list[tuple[float, float]]]) -> tuple[pd.DataFrame, list[AlignReport]]:
    """Map each study eye to its Vision row by ETDRS-sequence similarity, then copy weeks over."""
    visits = visits.copy()
    visits["week"] = np.nan
    reports: list[AlignReport] = []
    used: set[int] = set()
    for sid, g in visits.groupby("study_id", sort=False):
        g = g.sort_values("visit")
        et = [int(x) for x in g.etdrs.fillna(-1)]
        best, best_r = None, 0.0
        for j, seq in enumerate(vision):
            if j in used:
                continue
            r = difflib.SequenceMatcher(a=et, b=[int(e) for _, e in seq], autojunk=False).ratio()
            if r > best_r:
                best, best_r = j, r
        if best is None or best_r < 0.6:
            reports.append(AlignReport(sid, -1, 0, len(g), best_r))
            continue
        used.add(best)
        seq = vision[best]
        sm = difflib.SequenceMatcher(a=et, b=[int(e) for _, e in seq], autojunk=False)
        n = 0
        for blk in sm.get_matching_blocks():
            for k in range(blk.size):
                wk = seq[blk.b + k][0]
                if np.isfinite(wk):
                    visits.loc[g.index[blk.a + k], "week"] = wk
                    n += 1
        reports.append(AlignReport(sid, best, n, len(g), best_r))
    return visits, reports


def load_trex_visits(labels_dir: Path = LABELS_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (visits, alignment_report).

    visits columns: study_id, arm, eye, visit, etdrs, cst, week, interval_wk (gap to
    NEXT visit = interval chosen at this visit), prev_interval_wk, action
    (EXTEND / MAINTAIN / SHORTEN, T&E arms only, None when undefined),
    baseline_cst, eligible_ext_wk12, max_interval_wk.
    """
    xls = pd.read_excel(labels_dir / "full_labels" / "OCT-DME.xlsx", sheet_name=None, header=None)
    frames, reports = [], []
    for arm in ARMS:
        v = _arm_visits(xls[arm], arm)
        if arm in TE_ARMS:
            v, rep = _attach_weeks(v, _vision_sheet(xls[f"{arm} Vision"]))
            reports += rep
        else:  # Monthly arm: fixed q4w dosing by protocol; weeks not needed
            v["week"] = np.nan
        frames.append(v)
    visits = pd.concat(frames, ignore_index=True).sort_values(["study_id", "visit"]).reset_index(drop=True)

    g = visits.groupby("study_id")
    visits["baseline_cst"] = g.cst.transform("first")
    iv = g.week.shift(-1) - visits.week
    # Alignment glitches produce impossible gaps; the protocol range is 4-16 wk (+ missed visits).
    iv[(iv < MIN_PLAUSIBLE_WK) | (iv > MAX_PLAUSIBLE_WK)] = np.nan
    visits["interval_wk"] = iv
    visits["prev_interval_wk"] = visits.groupby("study_id").interval_wk.shift(1)
    visits["interval_wk_rounded"] = (visits.interval_wk / 2).round() * 2  # protocol works in 2-week steps
    # Protocol steps are 2 wk; real scheduling jitters by ~±1 wk, so use a tolerance, not rounding.
    d = visits.interval_wk - visits.prev_interval_wk
    visits["action"] = np.select(
        [d >= ACTION_TOL_WK, d <= -ACTION_TOL_WK, d.abs() < ACTION_TOL_WK],
        ["EXTEND", "SHORTEN", "MAINTAIN"],
        default=None,
    )
    # No decision exists during the 4 monthly loading doses, outside T&E arms, or without a gap pair.
    visits.loc[~visits.arm.isin(TE_ARMS) | d.isna() | (visits.visit <= LOADING_VISITS), "action"] = None
    rep = pd.DataFrame([r.__dict__ for r in reports])
    return visits, rep


def load_image_labels(labels_dir: Path = LABELS_DIR) -> pd.DataFrame:
    """Per-image table for TREX-DME: path parts, visit key, BCVA, CST, + biomarkers where labeled."""
    c = pd.read_excel(labels_dir / "ml_centric_labels" / "Clinical_Data_Images.xlsx")
    b = pd.read_csv(labels_dir / "ml_centric_labels" / "Biomarker_Clinical_Data_Images.csv")
    b = b.rename(columns={"Path (Trial/Arm/Folder/Visit/Eye/Image Name)": "File_Path"})
    bio_cols = [k for k in b.columns if k not in ("File_Path", "Eye_ID", "BCVA", "CST", "Patient_ID")]
    df = c.merge(b[["File_Path", *bio_cols]], on="File_Path", how="left")
    p = df.File_Path.str.strip("/").str.split("/", expand=True)
    df["trial"], df["arm"], df["study_id"], df["visit_dir"], df["eye"], df["fname"] = p[0], p[1], p[2], p[3], p[4], p[5]
    df = df[df.trial == "TREX DME"].copy()
    df["visit"] = df.visit_dir.str.lstrip("V").astype(int)
    df["has_biomarkers"] = df["Fluid (IRF)"].notna()
    return df.reset_index(drop=True)


def patient_of(study_id: str) -> str:
    """Site+subject, shared by both eyes of a patient (e.g. '0105TOS' -> '0105')."""
    return study_id[:4]


def make_patient_splits(study_ids: list[str], seed: int = 7, frac=(0.6, 0.2, 0.2)) -> dict[str, str]:
    """Patient-level train/val/test assignment so fellow eyes never straddle splits."""
    pats = sorted({patient_of(s) for s in study_ids})
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    n = len(pats)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    split = {p: ("train" if i < a else "val" if i < b else "test") for i, p in enumerate(pats)}
    return {s: split[patient_of(s)] for s in study_ids}
