"""5-fold patient-level CV for the OLIVES B-scan models; writes out-of-fold predictions.

  python scripts/train_oct.py --task biomarkers   # IRF / SRF / DRT classifier
  python scripts/train_oct.py --task cst          # central subfield thickness regression

For fold k: test = fold k, val = fold (k+1) % 5 (early stopping, temperature, no test peeking),
train = remaining 3 folds. Every visit is predicted only by the model whose test fold holds it.

Outputs:
  checkpoints/{task}_fold{k}.pt
  data/processed/oof_{task}_slices.csv   per-slice OOF outputs
  data/processed/oof_{task}_visits.csv   per-visit aggregated OOF outputs
  eval/results/{task}_cv.json            metrics
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .pth files can get the macOS hidden flag (iCloud), which Python 3.13 skips

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from visentia_te.data.olives_images import CENTER, FLUID_COLS, load_index  # noqa: E402
from visentia_te.imaging.oct_models import (  # noqa: E402
    N_FOLDS,
    TrainCfg,
    assign_folds,
    fit_temperature,
    predict_raw,
    preload,
    train_one,
)

INFER_HALFWIDTH = 8  # slices 16..32 (~±1 mm around fovea on a 6 mm raster)
CST_HALFWIDTH = 3  # central subfield (1 mm) ~ ±4 slices; ±3 keeps training time in budget
SHORT = {"Fluid (IRF)": "irf", "Fluid (SRF)": "srf", "DRT/ME": "drt"}


def _fin(y, p):
    m = np.isfinite(np.asarray(p, dtype=float))
    return np.asarray(y)[m], np.asarray(p, dtype=float)[m]


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    y, p = _fin(y, p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def safe_auc(y, p):
    y, p = _fin(y, p)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["biomarkers", "cst"], required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    args = ap.parse_args()
    task = args.task
    Path("checkpoints").mkdir(exist_ok=True)
    Path("eval/results").mkdir(parents=True, exist_ok=True)

    idx = load_index().drop_duplicates(["study_id", "visit", "slice"]).reset_index(drop=True)
    folds = assign_folds(sorted(idx.study_id.unique()))
    idx["fold"] = idx.study_id.map(folds)

    # inference set: central slices of every visit (needed downstream by the engine)
    inf = idx[(idx.slice - CENTER).abs() <= INFER_HALFWIDTH].reset_index(drop=True)
    if task == "biomarkers":
        lab = idx[idx.has_biomarkers].reset_index(drop=True)
        Y = lab[FLUID_COLS].astype(np.float32).values
    else:
        lab = idx[((idx.slice - CENTER).abs() <= CST_HALFWIDTH) & idx.CST.notna()].reset_index(drop=True)
        Y = lab[["CST"]].astype(np.float32).values
    t0 = time.time()
    X_lab, X_inf = preload(lab.png), preload(inf.png)
    print(f"[{task}] labeled {len(lab)} slices / {lab[['study_id', 'visit']].drop_duplicates().shape[0]} visits; "
          f"inference {len(inf)} slices; preload {time.time() - t0:.0f}s", flush=True)

    inf_out = np.full((len(inf), Y.shape[1]), np.nan, dtype=np.float32)
    lab_out = np.full_like(Y, np.nan)
    fold_info = []
    for k in [int(f) for f in args.folds.split(",")]:
        v = (k + 1) % N_FOLDS
        tr, va, te = (~lab.fold.isin([k, v])).values, (lab.fold == v).values, (lab.fold == k).values
        if task == "cst":  # standardize with train stats only
            mu, sd = float(Y[tr].mean()), float(Y[tr].std())
            Yn = (Y - mu) / sd
        else:
            Yn = Y
        cfg = TrainCfg(task=task, epochs=args.epochs, seed=k)
        model, info = train_one(cfg, X_lab[tr], Yn[tr], X_lab[va], Yn[va], log_prefix=f"[{task} f{k}]")
        va_raw, te_raw = predict_raw(model, X_lab[va]), predict_raw(model, X_lab[te])
        inf_raw = predict_raw(model, X_inf[(inf.fold == k).values])
        if task == "biomarkers":
            T = fit_temperature(va_raw, Y[va])
            sig = lambda z: 1 / (1 + np.exp(-z / T))  # noqa: E731
            lab_out[te], inf_out[(inf.fold == k).values] = sig(te_raw), sig(inf_raw)
            extra = dict(temperature=T.tolist())
        else:
            lab_out[te] = te_raw * sd + mu
            inf_out[(inf.fold == k).values] = inf_raw * sd + mu
            extra = dict(mu=mu, sd=sd)
        torch.save(dict(state_dict=model.state_dict(), cfg=cfg.__dict__, task=task, fold=k, **extra), f"checkpoints/{task}_fold{k}.pt")
        fold_info.append(dict(fold=k, val_fold=v, n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()),
                              best_epoch=info["best_epoch"], best_val=info["best_val"], **extra))

    # ---- outputs -------------------------------------------------------------------------
    names = [SHORT[c] for c in FLUID_COLS] if task == "biomarkers" else ["cst_pred"]
    keys = ["study_id", "arm", "eye", "visit", "slice", "fold", "CST", "BCVA"]
    inf_df = pd.concat([inf[keys], pd.DataFrame(inf_out, columns=[f"p_{n}" if task == "biomarkers" else n for n in names])], axis=1)
    inf_df.to_csv(f"data/processed/oof_{task}_slices.csv", index=False)
    g = inf_df.groupby(["study_id", "visit"])
    if task == "biomarkers":
        visits = g[[f"p_{n}" for n in names]].max()
        visits["p_fluid"] = visits[["p_irf", "p_srf"]].max(axis=1)  # max over slices & types (noisy-OR overcounts correlated slices)
        # continuous burden: mean over central slices of P(any fluid); fovea-only variant (±3 slices)
        inf_df["p_any"] = inf_df[["p_irf", "p_srf"]].max(axis=1)
        visits["fluid_burden"] = g.p_any.mean()
        visits["p_fluid_fovea"] = inf_df[(inf_df.slice - CENTER).abs() <= 3].groupby(["study_id", "visit"]).p_any.max()
    else:
        c = inf_df[(inf_df.slice - CENTER).abs() <= CST_HALFWIDTH].groupby(["study_id", "visit"])
        visits = c[["cst_pred"]].mean().join(c[["CST"]].first())
    visits = visits.reset_index()
    visits.to_csv(f"data/processed/oof_{task}_visits.csv", index=False)

    # ---- metrics -------------------------------------------------------------------------
    res: dict = dict(task=task, folds=fold_info, n_labeled_slices=len(lab))
    if task == "biomarkers":
        for j, c in enumerate(FLUID_COLS):
            y, p = Y[:, j], lab_out[:, j]
            yy, pp = _fin(y, p)
            res[f"slice_{SHORT[c]}"] = dict(auroc=safe_auc(y, p), auprc=float(average_precision_score(yy, pp)) if yy.sum() else None,
                                            prevalence=float(y.mean()), ece=ece(p, y))
        # visit-level: GT = any positive labeled slice within the inference window; pred = OOF visit max
        lv = lab[(lab.slice - CENTER).abs() <= INFER_HALFWIDTH].groupby(["study_id", "visit"])[FLUID_COLS].max().reset_index()
        lv = lv.merge(visits, on=["study_id", "visit"])
        for c in FLUID_COLS:
            n = SHORT[c]
            res[f"visit_{n}"] = dict(auroc=safe_auc(lv[c].values, lv[f"p_{n}"].values), n=len(lv), prevalence=float(lv[c].mean()),
                                     ece=ece(lv[f"p_{n}"].values, lv[c].values))
        yf = lv[["Fluid (IRF)", "Fluid (SRF)"]].max(axis=1).values
        for col in ("fluid_burden", "p_fluid_fovea"):
            res[f"visit_{col}_auroc_vs_any"] = safe_auc(lv[["Fluid (IRF)", "Fluid (SRF)"]].max(axis=1).values, lv[col].values)
        fov = lab[(lab.slice - CENTER).abs() <= 3].groupby(["study_id", "visit"])[["Fluid (IRF)", "Fluid (SRF)"]].max().max(axis=1)
        fv = lv.set_index(["study_id", "visit"]).join(fov.rename("gt_fovea"))
        res["visit_fovea_fluid"] = dict(auroc=safe_auc(fv.gt_fovea.values, fv.p_fluid_fovea.values), prevalence=float(fv.gt_fovea.mean()))
        res["visit_fluid_any"] = dict(auroc=safe_auc(yf, lv.p_fluid.values), ece=ece(lv.p_fluid.values, yf), n=len(lv), prevalence=float(yf.mean()))
    else:
        err = visits.cst_pred - visits.CST
        base = visits.sort_values("visit").groupby("study_id").first()
        visits["ratio_true"] = visits.CST / visits.study_id.map(base.CST)
        visits["ratio_pred"] = visits.cst_pred / visits.study_id.map(base.cst_pred)
        cat = lambda r: np.digitize((r - 1).abs() if hasattr(r, "abs") else abs(r - 1), [0.10, 0.20])  # noqa: E731
        later = visits.visit > 1
        res.update(
            visit_mae_um=float(err.abs().mean()), visit_median_ae_um=float(err.abs().median()), bias_um=float(err.mean()),
            pearson_r=float(visits[["cst_pred", "CST"]].dropna().corr().iloc[0, 1]), n_visits=len(visits),
            baseline_mae_predict_train_mean=float((visits.CST - visits.CST.mean()).abs().mean()),
            ratio_mae=float((visits.ratio_pred - visits.ratio_true)[later].abs().mean()),
            protocol_band_agreement=float((cat(visits.ratio_pred) == cat(visits.ratio_true))[later].mean()),
            within_10pct_of_truth=float((err.abs() / visits.CST <= 0.10).mean()),
        )
        visits.to_csv(f"data/processed/oof_{task}_visits.csv", index=False)
    Path(f"eval/results/{task}_cv.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "folds"}, indent=1))
    print(f"[{task}] total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
