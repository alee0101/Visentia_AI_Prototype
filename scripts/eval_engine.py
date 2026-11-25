"""Replay the T&E engine over real TREX-DME eyes and score it against investigator actions.

  python scripts/eval_engine.py                 # oracle inputs (trial-measured CST/BCVA)
  python scripts/eval_engine.py --inputs model  # OOF image-model CST (+ fluid), imaged eyes only

Teacher forcing: before each visit the engine's current interval is set to the interval the
trial actually used, so interval caps match and each decision is scored independently.

Metrics: accuracy, linear-weighted Cohen's kappa (ordinal SHORTEN<MAINTAIN<EXTEND), confusion
matrix, and the safety metric **unsafe extension rate** = P(engine EXTEND | investigator SHORTEN).
With model inputs, FLAG (physician review) is reported separately as a coverage/accuracy trade-off.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score, confusion_matrix  # noqa: E402

from visentia_te.engine.protocol import EyeState, Observation, load_protocol  # noqa: E402

ORDER = ["SHORTEN", "MAINTAIN", "EXTEND"]


def replay(visits: pd.DataFrame, protocol: dict, cst_col: str, sd_col: str | None, fluid_col: str | None) -> pd.DataFrame:
    rows = []
    for sid, g in visits.groupby("study_id"):
        st = EyeState(protocol)
        for _, r in g.sort_values("visit").iterrows():
            if np.isfinite(r.prev_interval_wk):
                st.interval_wk = int(np.clip(round(r.prev_interval_wk / 2) * 2, 4, protocol["interval"]["max_wk"]))
            cst = r[cst_col] if np.isfinite(r[cst_col]) else r.cst  # fall back to measured CST when no image
            obs = Observation(
                visit=int(r.visit), cst=float(cst), bcva=None if np.isnan(r.etdrs) else float(r.etdrs),
                cst_sd=float(r[sd_col]) if sd_col and np.isfinite(r[cst_col]) else 0.0,
                p_fluid=float(r[fluid_col]) if fluid_col and np.isfinite(r.get(fluid_col, np.nan)) else None,
            )
            if not np.isfinite(obs.cst):
                continue
            d = st.step(obs)
            rows.append(dict(study_id=sid, visit=int(r.visit), arm=r.arm, trial=r.action, engine=d.action, shown=d.recommended,
                             flagged=d.flagged, cst_input=obs.cst, cst_true=r.cst, imaged=bool(np.isfinite(r[cst_col])),
                             interval_in=st.interval_wk, trace=" | ".join(d.trace)))
    return pd.DataFrame(rows)


def score(df: pd.DataFrame) -> dict:
    d = df[df.trial.isin(ORDER) & df.engine.isin(ORDER)]
    if d.empty:
        return {}
    y, p = d.trial.map(ORDER.index), d.engine.map(ORDER.index)
    cm = confusion_matrix(d.trial, d.engine, labels=ORDER)
    out = dict(
        n=len(d), accuracy=float((y == p).mean()), kappa_linear=float(cohen_kappa_score(y, p, weights="linear")),
        unsafe_extension_rate=float(((d.engine == "EXTEND") & (d.trial == "SHORTEN")).sum() / max(1, (d.trial == "SHORTEN").sum())),
        engine_extend_precision=float(((d.engine == "EXTEND") & (d.trial != "SHORTEN")).sum() / max(1, (d.engine == "EXTEND").sum())),
        confusion={"rows=trial, cols=engine": ORDER, "matrix": cm.tolist()},
        majority_baseline_accuracy=float((d.trial == "MAINTAIN").mean()),
    )
    if d.flagged.any():
        nf = d[~d.flagged]
        out.update(flag_rate=float(d.flagged.mean()), accuracy_unflagged=float((nf.trial == nf.engine).mean()),
                   unsafe_extension_rate_unflagged=float(((nf.engine == "EXTEND") & (nf.trial == "SHORTEN")).sum() / max(1, (nf.trial == "SHORTEN").sum())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", choices=["oracle", "model"], default="oracle")
    args = ap.parse_args()
    protocol = load_protocol()
    v = pd.read_csv("data/processed/trex_visits.csv")
    v = v[v.arm.isin(["TREX", "GILA"])].copy()
    res: dict = {"inputs": args.inputs}
    if args.inputs == "oracle":
        v["cst_in"] = v.cst
        out = replay(v, protocol, "cst_in", None, None)
        res["all_te_eyes"] = score(out)
        imaged = set(pd.read_csv("data/processed/trex_images.csv", usecols=["study_id"]).study_id)
        res["imaged_eyes"] = score(out[out.study_id.isin(imaged)])
    else:
        cst = pd.read_csv("data/processed/oof_cst_visits.csv")[["study_id", "visit", "cst_pred"]]
        cv = json.loads(Path("eval/results/cst_cv.json").read_text())
        v = v.merge(cst, on=["study_id", "visit"], how="left")
        fl = Path("data/processed/oof_biomarkers_visits.csv")
        if fl.exists():
            v = v.merge(pd.read_csv(fl)[["study_id", "visit", "p_fluid_fovea"]], on=["study_id", "visit"], how="left")
        # per-visit CST uncertainty: OOF residual SD (global; a per-visit SD from slice spread is a TODO)
        v = v[v.study_id.isin(cst.study_id)]
        # Decisions compare CST to the eye's own (model-measured) reference, so per-eye bias cancels;
        # the relevant noise is the WITHIN-eye residual SD of the OOF predictions (a single scalar).
        s = pd.read_csv("data/processed/oof_cst_visits.csv").dropna(subset=["CST", "cst_pred"])
        res_ = s.cst_pred - s.CST
        within_sd = float((res_ - res_.groupby(s.study_id).transform("mean")).std())
        res["within_eye_cst_sd_um"] = within_sd
        fluid = "p_fluid_fovea" if "p_fluid_fovea" in v else None
        # no-uncertainty run: sd=0 and no fluid gate, i.e. the protocol applied to model CST as-is
        v["cst_sd"] = 0.0
        out = replay(v, protocol, "cst_pred", "cst_sd", None)
        res["model_no_uncertainty"] = score(out[out.imaged])
        out.to_csv("eval/results/engine_replay_model_noflag.csv", index=False)
        # abstention sweep: flag when ±k·sd crosses a rule boundary (and fluid gate)
        res["abstention_sweep"] = []
        for k in (0.25, 0.5, 0.75, 1.0, 1.5):
            v["cst_sd"] = within_sd * k / 1.5  # engine uses ±1.5·cst_sd internally
            o = replay(v, protocol, "cst_pred", "cst_sd", fluid)
            sc = score(o[o.imaged])
            res["abstention_sweep"].append(dict(k_sd=k, **{x: sc.get(x) for x in (
                "flag_rate", "accuracy_unflagged", "unsafe_extension_rate_unflagged", "accuracy", "unsafe_extension_rate")}))
            if k == 0.5:
                out = o  # default operating point saved for the UI / error analysis
        res["imaged_decisions"] = score(out[out.imaged])
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    out.to_csv(f"eval/results/engine_replay_{args.inputs}.csv", index=False)
    Path(f"eval/results/engine_{args.inputs}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
