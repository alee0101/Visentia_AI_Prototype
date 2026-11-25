"""Retina (ILM->BM) segmentation on Duke, then geometric CST on OLIVES/TREX.

  python scripts/train_layers.py                 # Duke 5-fold CV + final model + OLIVES inference
  python scripts/train_layers.py --skip-cv       # final model + OLIVES inference only

Duke CV (subject-level, same folds as the fluid model): per-column thickness error vs each
grader, compared with the inter-grader difference. The final model is trained on all 10 Duke
subjects; OLIVES is never used for training, so OLIVES thickness is out-of-sample everywhere
(only the linear µm calibration in scripts/eval_cst_sources.py uses OLIVES CST, fold-wise).

Outputs:
  checkpoints/layers_final.pt
  eval/results/layers_duke_cv.json
  data/processed/geo_cst_visits.csv     study_id, visit, geo_csf_rows, geo_csf_um, ... per visit
  data/processed/geo_profiles.npz       per-visit central-slice ILM/BM rows (for the UI)
  eval/figures/layers_on_olives.png
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

from visentia_te.data.duke import load_all, make_folds  # noqa: E402
from visentia_te.data.olives_images import CENTER, load_index, read_bscan  # noqa: E402
from visentia_te.imaging.layers import (  # noqa: E402
    UM_PER_ROW, csf_weights, duke_item, masked_loss, olives_canvas, predict_prob, thickness_profile,
)
from visentia_te.imaging.seg import build_model  # noqa: E402

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CSF_HALF = 4  # slices 20..28 cover the 1 mm circle at ~125 µm spacing


def train(samples, epochs: int, seed: int, bs: int = 8, lr: float = 3e-4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, _ = build_model(pretrained=True)
    model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((2 * len(samples) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    pool = [(s, g) for s in samples for g in (1, 2)]  # both graders as label noise
    for ep in range(epochs):
        model.train()
        order, tot = rng.permutation(len(pool)), 0.0
        for i in range(0, len(order), bs):
            items = [duke_item(*pool[j], rng=rng) for j in order[i : i + bs]]
            x, y, w = (torch.from_numpy(np.stack(a))[:, None].to(DEV) for a in zip(*items))
            loss = masked_loss(model(x), y, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(items)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  ep {ep:3d} loss {tot / len(pool):.4f}", flush=True)
    return model


def duke_eval(model, samples) -> dict:
    """Per-column thickness / boundary errors (µm, OLIVES scale) vs both graders."""
    e1, e2, eg, eilm, ebm, fm1, fmg = [], [], [], [], [], [], []
    x = np.stack([duke_item(s, 1)[0] for s in samples])
    probs = predict_prob(model, x, DEV)
    for s, p in zip(samples, probs):
        _, m1, w1 = duke_item(s, 1)
        _, m2, w2 = duke_item(s, 2)
        top, bot, th = thickness_profile(p)
        cols = (w1.max(0) > 0) & (w2.max(0) > 0) & np.isfinite(th)
        t1, t2 = m1.sum(0), m2.sum(0)
        top1 = m1.argmax(0).astype(float)
        bot1 = m1.shape[0] - m1[::-1].argmax(0)
        e1.append(np.abs(th - t1)[cols]); e2.append(np.abs(th - t2)[cols]); eg.append(np.abs(t1 - t2)[cols])
        eilm.append(np.abs(top - top1)[cols]); ebm.append(np.abs(bot - bot1)[cols])
        fm1.append(abs(th[cols].mean() - t1[cols].mean())); fmg.append(abs(t1[cols].mean() - t2[cols].mean()))
    um = lambda a: float(np.concatenate(a).mean() * UM_PER_ROW)  # noqa: E731
    return dict(col_thickness_mae_um_vs_g1=um(e1), col_thickness_mae_um_vs_g2=um(e2), col_thickness_mae_um_g1_vs_g2=um(eg),
                ilm_mae_um=um(eilm), bm_mae_um=um(ebm),
                frame_mean_thickness_ae_um_vs_g1=float(np.mean(fm1) * UM_PER_ROW),
                frame_mean_thickness_ae_um_g1_vs_g2=float(np.mean(fmg) * UM_PER_ROW), n_frames=len(samples))


def olives_infer(model) -> None:
    idx = load_index()
    idx = idx[(idx.slice - CENTER).abs() <= CSF_HALF].drop_duplicates(["study_id", "visit", "slice"])
    rows, prof = [], {}
    w_csf = csf_weights(np.arange(-CSF_HALF, CSF_HALF + 1))
    for (sid, v), g in idx.groupby(["study_id", "visit"]):
        g = g.sort_values("slice")
        x = np.stack([olives_canvas(read_bscan(p, size=(252, 256))) for p in g.png])
        probs = predict_prob(model, x, DEV)
        tops, bots, ths = (np.full((2 * CSF_HALF + 1, x.shape[2]), np.nan, np.float32) for _ in range(3))
        for s, p in zip(g.slice.values, probs):
            tops[s - CENTER + CSF_HALF], bots[s - CENTER + CSF_HALF], ths[s - CENTER + CSF_HALF] = thickness_profile(p)
        inside = w_csf & np.isfinite(ths)
        prof[f"{sid}|{v}"] = np.stack([tops, bots])
        rows.append(dict(study_id=sid, visit=int(v), n_slices=len(g), geo_csf_rows=float(np.nanmean(ths[inside])) if inside.any() else np.nan,
                         geo_center_rows=float(np.nanmean(ths[CSF_HALF, 118:139])),  # foveal B-scan, central ~0.5 mm
                         geo_csf_sd_rows=float(np.nanstd(ths[inside])) if inside.any() else np.nan,
                         CST=float(g.CST.iloc[0]) if g.CST.notna().any() else np.nan))
        if len(rows) % 100 == 0:
            print(f"  olives {len(rows)} visits", flush=True)
    df = pd.DataFrame(rows)
    df["geo_csf_um"] = df.geo_csf_rows * UM_PER_ROW
    df.to_csv("data/processed/geo_cst_visits.csv", index=False)
    np.savez_compressed("data/processed/geo_profiles.npz", **prof)
    d = df.dropna(subset=["CST", "geo_csf_um"])
    print(f"OLIVES: {len(df)} visits; raw geo vs CST r={d[['geo_csf_um', 'CST']].corr().iloc[0, 1]:.3f}, "
          f"bias={float((d.geo_csf_um - d.CST).mean()):.1f} µm", flush=True)
    figure(model, idx)


def figure(model, idx) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = idx[idx.slice == CENTER].sample(6, random_state=1)
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    x = np.stack([olives_canvas(read_bscan(p, size=(252, 256))) for p in c.png])
    for a, img, p, (_, r) in zip(ax.flat, x, predict_prob(model, x, DEV), c.iterrows()):
        top, bot, th = thickness_profile(p)
        a.imshow(img, cmap="gray"); a.plot(top, "y-", lw=1); a.plot(bot, "c-", lw=1)
        a.set_title(f"{r.study_id} V{r.visit}  CST {r.CST:.0f} / geo {np.nanmean(th[107:150]) * UM_PER_ROW:.0f} µm", fontsize=9)
        a.axis("off")
    fig.tight_layout()
    Path("eval/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("eval/figures/layers_on_olives.png", dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    samples = load_all()
    res: dict = {"epochs": args.epochs, "um_per_row": UM_PER_ROW}
    if not args.skip_cv:
        per_fold = []
        for f in make_folds(path=None):
            tr = [s for s in samples if s.subject in f["train"]]
            te = [s for s in samples if s.subject in f["test"]]
            print(f"[layers] fold {f['fold']} test subjects {f['test']}", flush=True)
            m = train(tr, args.epochs, seed=f["fold"])
            per_fold.append(dict(fold=f["fold"], **duke_eval(m, te)))
            print(f"  {per_fold[-1]}", flush=True)
        keys = [k for k in per_fold[0] if k not in ("fold", "n_frames")]
        res["cv_mean"] = {k: float(np.average([p[k] for p in per_fold], weights=[p["n_frames"] for p in per_fold])) for k in keys}
        res["folds"] = per_fold
        print(json.dumps(res["cv_mean"], indent=1), flush=True)
    print("[layers] final model on all Duke subjects", flush=True)
    model = train(samples, args.epochs, seed=99)
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), epochs=args.epochs, um_per_row=UM_PER_ROW), "checkpoints/layers_final.pt")
    if not args.skip_cv:
        Path("eval/results/layers_duke_cv.json").write_text(json.dumps(res, indent=1))
    olives_infer(model)
    print(f"[layers] total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
