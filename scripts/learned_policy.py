"""What do investigators weigh beyond the written protocol? Learned policy vs TREX rules.

  python scripts/learned_policy.py

Predict the investigator's action (SHORTEN / MAINTAIN / EXTEND) at every T&E decision from what
the clinician saw at that visit, with patient-level 5-fold CV, and compare with the rule engine.
Features use device CST (all 120 T&E eyes, 1,284 decisions) so measurement error is not a factor.

This is imitation, not optimisation: matching investigators is not the same as better outcomes.
The point is (a) to quantify how much of the protocol-investigator gap is predictable from
visit data at all, and (b) to see which signals investigators respond to that the rules ignore.

Writes eval/results/learned_policy.json and eval/figures/learned_policy_importance.png.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from visentia_te.engine.protocol import EyeState, Observation, load_protocol  # noqa: E402
from visentia_te.imaging.oct_models import assign_folds  # noqa: E402

ORDER = ["SHORTEN", "MAINTAIN", "EXTEND"]


def build() -> pd.DataFrame:
    """One row per T&E decision: clinician-visible features + protocol action + investigator action."""
    protocol = load_protocol()
    v = pd.read_csv("data/processed/trex_visits.csv")
    v = v[v.arm.isin(["TREX", "GILA"])]
    rows = []
    for sid, g in v.groupby("study_id"):
        st = EyeState(protocol)
        prev = None
        for _, r in g.sort_values("visit").iterrows():
            if not np.isfinite(r.cst):
                continue
            if np.isfinite(r.prev_interval_wk):
                st.interval_wk = int(np.clip(round(r.prev_interval_wk / 2) * 2, 4, protocol["interval"]["max_wk"]))
            iv_in, best_before = st.interval_wk, st.best_bcva
            d = st.step(Observation(int(r.visit), float(r.cst), None if np.isnan(r.etdrs) else float(r.etdrs)))
            if r.action in ORDER and d.action in ORDER:
                rows.append(dict(
                    study_id=sid, visit=int(r.visit), y=ORDER.index(r.action), protocol=ORDER.index(d.action),
                    cst=r.cst, cst_vs_ref=(d.ratio - 1) if d.ratio is not None else np.nan, cst_vs_prev=(r.cst / prev.cst - 1) if prev is not None else 0.0,
                    cst_vs_baseline=r.cst / r.baseline_cst - 1, above_gate=float(r.cst > protocol["loading_cst_gate_um"]),
                    bcva=r.etdrs, bcva_vs_prev=(r.etdrs - prev.etdrs) if prev is not None else 0.0,
                    bcva_below_best=(best_before - r.etdrs) if best_before is not None and np.isfinite(r.etdrs) else 0.0,
                    interval_wk=iv_in, at_max=float(iv_in >= protocol["interval"]["max_wk"]), visit_no=int(r.visit)))
            prev = r
    return pd.DataFrame(rows)


FEATS = ["cst_vs_ref", "cst_vs_prev", "cst_vs_baseline", "cst", "above_gate", "bcva", "bcva_vs_prev", "bcva_below_best",
         "interval_wk", "at_max", "visit_no"]


def score(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return dict(accuracy=float((y == p).mean()), kappa_linear=float(cohen_kappa_score(y, p, weights="linear")),
                unsafe_extension_rate=float(((p == 2) & (y == 0)).sum() / max(1, (y == 0).sum())),
                extend_rate=float((p == 2).mean()))


def cv_predict(df: pd.DataFrame, make, feats) -> np.ndarray:
    out = np.zeros(len(df), int)
    X, y = df[feats].fillna(0).values, df.y.values
    for k in sorted(df.fold.unique()):
        tr, te = (df.fold != k).values, (df.fold == k).values
        m = make().fit(X[tr], y[tr])
        out[te] = m.predict(X[te])
    return out


def main() -> None:
    df = build()
    df["fold"] = df.study_id.map(assign_folds(sorted(df.study_id.unique())))
    df["protocol_extend"] = (df.protocol == 2).astype(float)
    df["protocol_shorten"] = (df.protocol == 0).astype(float)
    hyb = FEATS + ["protocol_extend", "protocol_shorten"]
    gbm = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,  # noqa: E731
                                                 class_weight="balanced", random_state=0)
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))  # noqa: E731
    res = dict(n_decisions=len(df), n_eyes=int(df.study_id.nunique()),
               class_counts={c: int((df.y == i).sum()) for i, c in enumerate(ORDER)},
               majority_maintain=score(df.y, np.ones(len(df), int)),
               protocol=score(df.y, df.protocol))
    preds = {}
    for name, make, feats in (("logreg", lr, FEATS), ("gbm", gbm, FEATS), ("gbm_plus_protocol", gbm, hyb)):
        preds[name] = cv_predict(df, make, feats)
        res[name] = score(df.y, preds[name])
    # eye-level paired bootstrap: is the learned policy's gain over the protocol real?
    rng = np.random.default_rng(0)
    eyes = df.study_id.unique()
    pos = {e: np.flatnonzero(df.study_id.values == e) for e in eyes}
    y = df.y.values
    d_acc, d_kap = [], []
    for _ in range(1000):
        ix = np.concatenate([pos[e] for e in rng.choice(eyes, len(eyes))])
        d_acc.append((preds["gbm"][ix] == y[ix]).mean() - (df.protocol.values[ix] == y[ix]).mean())
        d_kap.append(cohen_kappa_score(y[ix], preds["gbm"][ix], weights="linear")
                     - cohen_kappa_score(y[ix], df.protocol.values[ix], weights="linear"))
    res["gbm_minus_protocol_95ci"] = dict(accuracy=np.percentile(d_acc, [2.5, 97.5]).round(3).tolist(),
                                          kappa=np.percentile(d_kap, [2.5, 97.5]).round(3).tolist())
    # where does the protocol disagree with investigators, and does the learned policy fix it?
    pr, gb = df.protocol.values, preds["gbm"]
    dis = pr != df.y.values
    res["protocol_errors"] = dict(n=int(dis.sum()), learned_fixes=float((gb[dis] == df.y.values[dis]).mean()),
                                  learned_breaks_on_protocol_correct=float((gb[~dis] != df.y.values[~dis]).mean()))
    # protocol-vs-investigator confusion, broken down
    res["protocol_confusion"] = pd.crosstab(df.y.map(dict(enumerate(ORDER))), df.protocol.map(dict(enumerate(ORDER)))).to_dict()
    # what investigators respond to: permutation importance of the full-data GBM (descriptive only)
    m = gbm().fit(df[FEATS].fillna(0).values, df.y.values)
    imp = permutation_importance(m, df[FEATS].fillna(0).values, df.y.values, n_repeats=20, random_state=0,
                                 scoring=lambda est, X, y: cohen_kappa_score(y, est.predict(X), weights="linear"))
    res["importance_kappa_drop"] = dict(sorted(zip(FEATS, imp.importances_mean.round(4).tolist()), key=lambda t: -t[1]))
    # descriptive: investigator EXTEND rate when the protocol says EXTEND, split by interval and CST trend
    ext = df[df.protocol == 2]
    res["when_protocol_says_extend"] = {
        "investigator_extends": float((ext.y == 2).mean()),
        "by_interval_wk": ext.groupby("interval_wk").y.apply(lambda s: round(float((s == 2).mean()), 3)).to_dict(),
        "cst_rising_vs_prev_gt5pct": float((ext[ext.cst_vs_prev > 0.05].y == 2).mean()),
        "cst_not_rising": float((ext[ext.cst_vs_prev <= 0.05].y == 2).mean()),
        "n_rising": int((ext.cst_vs_prev > 0.05).sum()),
    }
    Path("eval/results/learned_policy.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "protocol_confusion"}, indent=1, default=str))
    figure(res["importance_kappa_drop"])


def figure(imp: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"cst_vs_ref": "CST vs T&E reference", "cst_vs_prev": "CST vs previous visit", "cst_vs_baseline": "CST vs screening",
              "cst": "CST (µm)", "above_gate": "CST > 325 µm", "bcva": "BCVA", "bcva_vs_prev": "BCVA vs previous visit",
              "bcva_below_best": "BCVA below best", "interval_wk": "Current interval", "at_max": "At max interval", "visit_no": "Visit number"}
    items = sorted(imp.items(), key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh([labels[k] for k, _ in items], [v for _, v in items], color="#257de0", height=0.6)
    ax.set_xlabel("Drop in linear κ when shuffled")
    ax.set_title("What investigators respond to (learned policy)", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e7e7e7")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("eval/figures/learned_policy_importance.png", dpi=130)


if __name__ == "__main__":
    main()
