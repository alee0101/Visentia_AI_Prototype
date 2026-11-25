"""5-fold patient-level CV + final model for Duke DME fluid segmentation.

Per fold (data/processed/duke_folds.json): 2 test subjects; of the 8 training
subjects one is held out as validation (early stopping on val pooled Dice, and
threshold / min-component-size tuning). Test subjects are never used for tuning.
The final model trains on all 10 subjects for the median best epoch count and
uses the median of the per-fold post-processing parameters.

Usage: .venv/bin/python scripts/train_seg.py [--max-epochs 80] [--patience 15]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visentia_te.data.duke import DukeFluidDataset, DukeSample, load_all  # noqa: E402
from visentia_te.imaging.seg import (  # noqa: E402
    DiceBCELoss, build_model, dice, load_checkpoint, postprocess, predict, save_checkpoint,
)

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
MIN_SIZES = [0, 10, 25, 50, 100, 200]
CKPT, RES, FIG = ROOT / "checkpoints", ROOT / "eval" / "results", ROOT / "eval" / "figures"


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def pooled_dice(preds: list[np.ndarray], gts: list[np.ndarray]) -> float:
    return dice(np.concatenate([p.ravel() for p in preds]), np.concatenate([g.ravel() for g in gts]))


def train(samples: list[DukeSample], train_ids: list[int], val: list[DukeSample] | None,
          epochs: int, patience: int, seed: int, bs: int, lr: float):
    """Train; if ``val`` is given, early-stop on val pooled Dice @0.5. Returns (model, best_epoch, log)."""
    seed_all(seed)
    ds = DukeFluidDataset(train_ids, augment=True, seed=seed, samples=samples)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True,
                    generator=torch.Generator().manual_seed(seed), collate_fn=_collate)
    model, pretrained = build_model()
    model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = DiceBCELoss()
    best, best_ep, best_state, log, bad = -1.0, epochs, None, [], 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
        rec = {"epoch": ep, "loss": tot / len(dl), "sec": time.time() - t0}
        if val is not None:
            probs = [predict(model, s.image, tta_flip=False) for s in val]
            rec["val_dice"] = pooled_dice([p > 0.5 for p in probs], [s.fluid1 for s in val])
            if rec["val_dice"] > best:
                best, best_ep, bad = rec["val_dice"], ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        log.append(rec)
        print("   ", {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}, flush=True)
        if val is not None and bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, log, pretrained


def _collate(batch):
    xs, ys, _ = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


def tune_postproc(probs: list[np.ndarray], gts: list[np.ndarray]) -> tuple[float, int, float]:
    best = (0.5, 0, -1.0)
    for t in THRESHOLDS:
        for m in MIN_SIZES:
            d = pooled_dice([postprocess(p, t, m) for p in probs], gts)
            if d > best[2] + 1e-6:
                best = (t, m, d)
    return best


def frame_metrics(preds: list[np.ndarray], samples: list[DukeSample]) -> dict:
    g1, g2 = [s.fluid1 for s in samples], [s.fluid2 for s in samples]
    pf = [d for p, g in zip(preds, g1) if not np.isnan(d := dice(p, g))]
    gt_pos = np.array([g.any() for g in g1])
    pr_pos = np.array([p.any() for p in preds])
    return {
        "n_frames": len(samples),
        "pooled_dice_g1": pooled_dice(preds, g1),
        "pooled_dice_g2": pooled_dice(preds, g2),
        "per_frame_dice_g1": float(np.mean(pf)) if pf else float("nan"),
        "n_per_frame": len(pf),
        "inter_grader_pooled_dice": pooled_dice(g1, g2),
        "det_sensitivity": float(pr_pos[gt_pos].mean()) if gt_pos.any() else float("nan"),
        "det_specificity": float((~pr_pos[~gt_pos]).mean()) if (~gt_pos).any() else float("nan"),
        "n_gt_pos": int(gt_pos.sum()), "n_gt_neg": int((~gt_pos).sum()),
    }


def overlay_fig(rows: list[tuple[DukeSample, np.ndarray, float]], out: Path) -> None:
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 2.9 * len(rows)))
    for (s, pred, d), (a0, a1) in zip(rows, axes):
        for ax, m, c, ttl in ((a0, s.fluid1, (1, .2, .2), "grader 1"), (a1, pred, (.2, .9, .3), "model")):
            ax.imshow(s.image, cmap="gray", aspect="auto")
            ov = np.zeros((*m.shape, 4), np.float32)
            ov[m > 0] = (*c, 0.5)
            ax.imshow(ov, aspect="auto")
            ax.set_title(f"S{s.subject:02d} f{s.frame}  {ttl}" + (f"  Dice={d:.2f}" if ttl == "model" else ""), fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def write_markdown(out: dict, path: Path) -> None:
    """Short markdown summary table of seg_cv.json."""
    f = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    lines = ["# Duke DME fluid segmentation: 5-fold patient-level CV", "",
             "| fold | test subj | val subj | best ep | thr | min px | pooled Dice G1 | per-frame Dice G1 | pooled Dice G2 | "
             "G1 vs G2 (human) | det sens | det spec |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["folds"]:
        lines.append(f"| {r['fold']} | {r['test']} | {r['val']} | {r['best_epoch']} | {r['threshold']} | {r['min_size']} | "
                     f"{f(r['pooled_dice_g1'])} | {f(r['per_frame_dice_g1'])} (n={r['n_per_frame']}) | {f(r['pooled_dice_g2'])} | "
                     f"{f(r['inter_grader_pooled_dice'])} | {f(r['det_sensitivity'])} ({r['n_gt_pos']}) | "
                     f"{f(r['det_specificity'])} ({r['n_gt_neg']}) |")
    o = out["overall"]
    lines.append(f"| **all** | 10 | | | | | **{f(o['pooled_dice_g1'])}** | {f(o['per_frame_dice_g1'])} (n={o['n_per_frame']}) | "
                 f"{f(o['pooled_dice_g2'])} | {f(o['inter_grader_pooled_dice'])} | {f(o['det_sensitivity'])} ({o['n_gt_pos']}) | "
                 f"{f(o['det_specificity'])} ({o['n_gt_neg']}) |")
    lines += ["", "Pooled Dice = Dice over all pixels of all test frames concatenated. Per-frame Dice = mean over "
              "frames where GT or prediction has fluid. Detection = frame has any fluid (GT grader 1). "
              "Thresholds/min-size tuned on the validation subject only.",
              f"Final model: {out.get('final')}", f"CV wall time: {out['cv_wall_sec'] / 60:.1f} min"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--skip-final", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse existing checkpoints/seg_fold{k}.pt")
    args = ap.parse_args()
    for d in (CKPT, RES, FIG):
        d.mkdir(parents=True, exist_ok=True)
    samples = load_all()
    folds = json.loads((ROOT / "data/processed/duke_folds.json").read_text())["folds"]
    t_start = time.time()
    results, all_preds, all_samples, pretrained = [], [], [], True
    for f in folds:
        k = f["fold"]
        if args.folds is not None and k not in args.folds:
            continue
        val_id = f["train"][k % len(f["train"])]
        tr_ids = [s for s in f["train"] if s != val_id]
        val = [s for s in samples if s.subject == val_id]
        test = [s for s in samples if s.subject in f["test"]]
        print(f"fold {k}: train {tr_ids} val {val_id} test {f['test']}", flush=True)
        t0 = time.time()
        ck = CKPT / f"seg_fold{k}.pt"
        if args.resume and ck.exists():  # reuse a finished fold; test metrics are recomputed
            model, cm = load_checkpoint(ck, DEV)
            best_ep, log = cm["best_epoch"], [None] * cm.get("epochs_run", 0)
            print(f"  resumed {ck.name} (best_epoch {best_ep})", flush=True)
        else:
            model, best_ep, log, pretrained = train(samples, tr_ids, val, args.max_epochs, args.patience,
                                                    args.seed + k, args.bs, args.lr)
        thr, ms, vd = tune_postproc([predict(model, s.image) for s in val], [s.fluid1 for s in val])
        preds = [postprocess(predict(model, s.image), thr, ms) for s in test]
        m = frame_metrics(preds, test)
        m.update(fold=k, test=f["test"], val=val_id, best_epoch=best_ep, threshold=thr, min_size=ms,
                 val_dice_tuned=vd, train_sec=time.time() - t0, epochs_run=len(log))
        print(f"  fold {k} -> {json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in m.items()})}", flush=True)
        save_checkpoint(ck, model, fold=k, threshold=thr, min_size=ms, best_epoch=best_ep,
                        epochs_run=len(log), max_epochs=args.max_epochs, train_size=(256, 256))
        results.append(m)
        all_preds += preds
        all_samples += test
    overall = frame_metrics(all_preds, all_samples)
    cv_sec = time.time() - t_start

    final = None
    if not args.skip_final:
        n_ep = int(np.median([r["best_epoch"] for r in results]))
        thr = float(np.median([r["threshold"] for r in results]))
        ms = int(np.median([r["min_size"] for r in results]))
        print(f"final: all 10 subjects, {n_ep} epochs, threshold {thr}, min_size {ms}", flush=True)
        t0 = time.time()
        model, _, _, _ = train(samples, list(range(1, 11)), None, n_ep, 0, args.seed + 100, args.bs, args.lr)
        save_checkpoint(CKPT / "seg_final.pt", model, threshold=thr, min_size=ms, epochs=n_ep, train_size=(256, 256))
        final = {"epochs": n_ep, "threshold": thr, "min_size": ms, "train_sec": time.time() - t0}

    out = {"config": vars(args) | {"device": str(DEV), "imagenet_pretrained": pretrained,
                                   "loss": "BCE + batch soft Dice", "thresholds": THRESHOLDS, "min_sizes": MIN_SIZES},
           "folds": results, "overall": overall, "cv_wall_sec": cv_sec, "final": final}
    (RES / "seg_cv.json").write_text(json.dumps(out, indent=2))
    write_markdown(out, RES / "seg_cv.md")

    rows = [(s, p, dice(p, s.fluid1)) for s, p in zip(all_samples, all_preds)]
    scored = sorted([r for r in rows if not np.isnan(r[2]) and r[0].fluid1.sum() > 500], key=lambda r: r[2])
    overlay_fig(scored[-2:][::-1] + scored[len(scored) // 2: len(scored) // 2 + 2] + scored[:2], FIG / "seg_examples.png")
    print(json.dumps(overall, indent=2), f"\nCV wall {cv_sec / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
