"""Sanity check for the Duke DME dataset: counts, inter-grader Dice, sample overlays.

Usage: .venv/bin/python scripts/duke_sanity.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visentia_te.data.duke import (  # noqa: E402
    AXIAL_UM_PER_PIXEL, DukeSample, load_all, make_folds, retinal_thickness,
)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice of two binary masks; NaN if both empty."""
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else float("nan")


def plot_grid(samples: list[DukeSample], out: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for ax, s in zip(axes.flat, samples):
        ax.imshow(s.image, cmap="gray", aspect="auto")
        ov = np.zeros((*s.image.shape, 4), np.float32)
        ov[s.fluid1 > 0] = (1.0, 0.2, 0.2, 0.45)
        ov[(s.fluid2 > 0) & (s.fluid1 == 0)] = (0.2, 0.6, 1.0, 0.45)
        ax.imshow(ov, aspect="auto")
        x = np.arange(s.ilm.size)
        ax.plot(x, s.ilm, c="yellow", lw=0.8, label="ILM")
        ax.plot(x, s.bm, c="lime", lw=0.8, label="BM")
        ax.set_title(f"S{s.subject:02d} frame {s.frame}  cols {s.col_range}  "
                     f"Dice G1/G2={dice(s.fluid1, s.fluid2):.2f}", fontsize=9)
        ax.axis("off")
    axes.flat[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("Duke DME: red = grader-1 fluid, blue = grader-2 only", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    samples = load_all()
    subjects = sorted({s.subject for s in samples})
    print(f"subjects: {len(subjects)}  annotated frames: {len(samples)}")
    shapes = sorted({s.image.shape for s in samples})
    print(f"cropped shapes (H, W): min W={min(w for _, w in shapes)} max W={max(w for _, w in shapes)}, H={shapes[0][0]}")
    for g, key in ((1, "fluid1"), (2, "fluid2")):
        m = [getattr(s, key) for s in samples]
        print(f"grader {g}: frames with fluid {np.mean([x.any() for x in m]):.3f}, "
              f"mean fluid pixel fraction {np.mean([x.mean() for x in m]):.4f}")
    th = np.concatenate([retinal_thickness(s.ilm, s.bm) for s in samples])
    print(f"retinal thickness px: median {np.nanmedian(th):.1f} (~{np.nanmedian(th) * AXIAL_UM_PER_PIXEL:.0f} um "
          f"@ {AXIAL_UM_PER_PIXEL} um/px assumed), NaN cols {np.isnan(th).mean():.4f}")

    # Inter-grader Dice: pooled over all pixels, and mean per-frame (frames where either grader marks fluid).
    per_subj: dict[int, list[DukeSample]] = defaultdict(list)
    for s in samples:
        per_subj[s.subject].append(s)
    pooled = lambda ss: dice(np.concatenate([s.fluid1.ravel() for s in ss]),
                             np.concatenate([s.fluid2.ravel() for s in ss]))
    frame_d = [d for s in samples if not np.isnan(d := dice(s.fluid1, s.fluid2))]
    print(f"\nINTER-GRADER DICE  pooled: {pooled(samples):.3f}   mean per-frame (n={len(frame_d)}): "
          f"{np.mean(frame_d):.3f}")
    for sid in subjects:
        ss = per_subj[sid]
        print(f"  S{sid:02d}: pooled Dice {pooled(ss):.3f}  frames {len(ss)}  "
              f"fluid px G1 {sum(int(s.fluid1.sum()) for s in ss):>7d} G2 {sum(int(s.fluid2.sum()) for s in ss):>7d}")

    folds = make_folds()
    print("\nfolds:", [f["test"] for f in folds])

    fluid = sorted(samples, key=lambda s: -int(s.fluid1.sum()))
    pick = [fluid[i] for i in (0, 5, 15, 30)] + [s for s in samples if not s.fluid1.any()][:2]
    plot_grid(pick[:6], ROOT / "eval" / "figures" / "duke_samples.png")
    print("saved eval/figures/duke_samples.png")


if __name__ == "__main__":
    main()
