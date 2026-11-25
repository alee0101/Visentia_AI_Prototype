"""Compare CST sources on OLIVES/TREX visits: CNN regression vs geometric (layer seg) vs stack.

  python scripts/eval_cst_sources.py

Calibration is fitted out-of-fold with the same patient-level folds as the CNN (fold k is
predicted by a linear model fitted on the other folds), so no source sees its test patients.

The metric that matters for the protocol is the WITHIN-eye residual SD: decisions compare CST
with the eye's own reference, so a constant per-eye offset cancels; visit-to-visit noise does not.

Writes data/processed/oof_cst_{geo,stack}_visits.csv (study_id, visit, cst_pred, CST) for
scripts/eval_engine.py --cst-file, and eval/results/cst_sources.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .pth files can get the macOS hidden flag (iCloud), which Python 3.13 skips

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from visentia_te.imaging.oct_models import assign_folds  # noqa: E402

QC_MAX_SPREAD_ROWS = 15.0


def metrics(df: pd.DataFrame, col: str) -> dict:
    d = df.dropna(subset=[col, "CST"])
    r = d[col] - d.CST
    within = r - r.groupby(d.study_id).transform("mean")
    base = d.sort_values("visit").groupby("study_id").first()
    later = d.visit > 1
    rt, rp = d.CST / d.study_id.map(base.CST), d[col] / d.study_id.map(base[col])
    cat = lambda x: np.digitize((x - 1).abs(), [0.10, 0.20])  # noqa: E731
    # visit-to-visit change error: what a T&E decision actually consumes
    dd = d.sort_values(["study_id", "visit"])
    dtrue, dpred = dd.groupby("study_id").CST.diff(), dd.groupby("study_id")[col].diff()
    return dict(n=len(d), mae_um=float(r.abs().mean()), bias_um=float(r.mean()), pearson_r=float(d[[col, "CST"]].corr().iloc[0, 1]),
                within_eye_sd_um=float(within.std()), between_eye_sd_um=float(r.groupby(d.study_id).mean().std()),
                change_mae_um=float((dpred - dtrue).abs().mean()), change_r=float(pd.concat([dpred, dtrue], axis=1).dropna().corr().iloc[0, 1]),
                ratio_mae=float((rp - rt)[later].abs().mean()), band_agreement=float((cat(rp) == cat(rt))[later].mean()))


def oof_linear(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    out = np.full(len(df), np.nan)
    ok = df[feats + ["CST"]].notna().all(axis=1).values
    for k in sorted(df.fold.unique()):
        tr, te = ok & (df.fold != k).values, ok & (df.fold == k).values
        X = np.c_[np.ones(tr.sum()), df.loc[tr, feats].values]
        beta, *_ = np.linalg.lstsq(X, df.loc[tr, "CST"].values, rcond=None)
        out[te] = np.c_[np.ones(te.sum()), df.loc[te, feats].values] @ beta
    return out


def oof_visit_sd(df: pd.DataFrame, col: str) -> np.ndarray:
    """Per-visit CST uncertainty (µm) from the segmentation's own thickness spread.

    Fitted out-of-fold: on training-fold eyes, regress |within-eye residual| on the spread
    (QC-passing visits only), then convert mean-absolute to SD (x sqrt(pi/2), Gaussian).
    """
    out = np.full(len(df), np.nan)
    r = df[col] - df.CST
    w = (r - r.groupby(df.study_id).transform("mean")).abs()
    ok = (w.notna() & df.geo_csf_sd_rows.notna() & df.qc_ok).values
    for k in sorted(df.fold.unique()):
        tr, te = ok & (df.fold != k).values, (df.fold == k).values & df.geo_csf_sd_rows.notna().values
        b, a = np.polyfit(df.geo_csf_sd_rows[tr], w[tr], 1)
        out[te] = np.sqrt(np.pi / 2) * np.maximum(a + b * df.geo_csf_sd_rows[te], 2.0)
    return out


def main() -> None:
    cnn = pd.read_csv("data/processed/oof_cst_visits.csv")[["study_id", "visit", "cst_pred", "CST"]].rename(columns={"cst_pred": "cnn"})
    geo = pd.read_csv("data/processed/geo_cst_visits.csv")
    df = geo.drop(columns=["CST"]).merge(cnn, on=["study_id", "visit"], how="outer")
    df["fold"] = df.study_id.map(assign_folds(sorted(df.study_id.unique())))
    df["geo_raw"] = df.geo_csf_um
    df["geo_cal"] = oof_linear(df, ["geo_csf_um"])
    df["stack"] = oof_linear(df, ["geo_csf_um", "cnn"])
    # Image QC: a large thickness spread inside the 1 mm circle means the retina mask failed
    # (vitreous/choroid included). Threshold set from the error-vs-spread curve: errors are
    # ~3-15 µm median below 15 rows and ~100 µm above it.
    df["qc_ok"] = ~(df.geo_csf_sd_rows > QC_MAX_SPREAD_ROWS)
    res = {c: metrics(df, c) for c in ("cnn", "geo_raw", "geo_cal", "stack")}
    res.update({f"{c}_qc_ok": metrics(df[df.qc_ok], c) for c in ("geo_cal", "stack")})
    res["qc_fail_rate"] = float((~df.qc_ok).mean())
    for c in ("geo_cal", "stack"):
        df[f"{c}_sd"] = oof_visit_sd(df, c)
        o = df[["study_id", "visit", c, "CST", "qc_ok", f"{c}_sd"]].rename(columns={c: "cst_pred", f"{c}_sd": "cst_sd_visit"})
        o.to_csv(f"data/processed/oof_cst_{c.replace('_cal', '')}_visits.csv", index=False)
    Path("eval/results/cst_sources.json").write_text(json.dumps(res, indent=1))
    print(pd.DataFrame(res).T.round(3).to_string())


if __name__ == "__main__":
    main()
