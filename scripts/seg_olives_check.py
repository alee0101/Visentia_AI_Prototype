"""Domain-shift sanity check: Duke-trained seg_final on OLIVES/TREX B-scans.

1. Overlay figure of 12 central (slice 24) labelled B-scans, 6 with IRF and 6 without.
2. Crude agreement: AUROC of predicted fluid area vs the expert IRF label on ~200
   labelled train-split B-scans (slices 12-36), plus central slices only.

The checkpoint's min_size was tuned at Duke native resolution (~496 x 520 px);
OLIVES PNGs are 252 x 256, so each OLIVES pixel covers ~3.9 Duke pixels and
min_size is rescaled by that area ratio.

Usage: .venv/bin/python scripts/seg_olives_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visentia_te.data.olives_images import load_index  # noqa: E402
from visentia_te.imaging.seg import load_checkpoint, postprocess, predict  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
DUKE_PX_PER_OLIVES_PX = (496 / 252) * (512 / 256)


def read_native(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label == 1], score[label == 0]
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "seg_final.pt"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default=DEV)
    args = ap.parse_args()
    model, meta = load_checkpoint(args.ckpt, args.device)
    thr = meta["threshold"]
    ms = int(round(meta["min_size"] / DUKE_PX_PER_OLIVES_PX))
    df = load_index()
    lab = df[df.has_biomarkers & (df.split == "train") & df["Fluid (IRF)"].notna()]
    rng = np.random.default_rng(0)

    # 1) figure
    cen = lab[lab.slice == 24]
    pick = [cen[cen["Fluid (IRF)"] == v].sample(6, random_state=0) for v in (1, 0)]
    fig, axes = plt.subplots(4, 6, figsize=(18, 10))
    for (_, r), ax_pair in zip(list(pick[0].iterrows()) + list(pick[1].iterrows()),
                               [(axes[2 * (i // 6), i % 6], axes[2 * (i // 6) + 1, i % 6]) for i in range(12)]):
        img = read_native(r.png)
        m = postprocess(predict(model, img), thr, ms)
        ax_pair[0].imshow(img, cmap="gray")
        ax_pair[0].set_title(f"{r.study_id} V{r.visit} IRF={int(r['Fluid (IRF)'])} SRF={int(r['Fluid (SRF)'])}", fontsize=8)
        ax_pair[1].imshow(img, cmap="gray")
        ov = np.zeros((*m.shape, 4), np.float32)
        ov[m > 0] = (0.2, 0.9, 0.3, 0.55)
        ax_pair[1].imshow(ov)
        ax_pair[1].set_title(f"pred fluid px={int(m.sum())}", fontsize=8)
        for a in ax_pair:
            a.axis("off")
    fig.suptitle("seg_final (Duke-trained) on OLIVES central B-scans: rows 1-2 IRF=1, rows 3-4 IRF=0", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "eval" / "figures" / "seg_on_olives.png", dpi=80)
    plt.close(fig)

    # 2) AUROC on ~200 labelled scans
    sub = lab[(lab.slice - 24).abs() <= 12]
    sub = sub.iloc[rng.choice(len(sub), size=min(args.n, len(sub)), replace=False)]
    sets = {f"random{len(sub)}_slices12-36": sub, "central_slice24_all_train": cen}
    out = {"ckpt": Path(args.ckpt).name, "threshold": thr, "min_size_olives_px": ms, "split": "train"}
    for name, d in sets.items():
        area, psum = [], []
        for p in d.png:
            prob = predict(model, read_native(p))
            area.append(postprocess(prob, thr, ms).sum())
            psum.append(prob.sum())
        y = d["Fluid (IRF)"].to_numpy().astype(int)
        area, psum = np.array(area, float), np.array(psum, float)
        srf = d["Fluid (SRF)"].to_numpy().astype(int)
        anyf = ((y + srf) > 0).astype(int)
        out[name] = {
            "n": len(d), "n_irf_pos": int(y.sum()), "n_studies": int(d.study_id.nunique()),
            "auroc_irf_area": auroc(area, y), "auroc_irf_probsum": auroc(psum, y),
            "auroc_irf_or_srf_area": auroc(area, anyf) if 0 < anyf.sum() < len(anyf) else None,
            "median_area_irf1": float(np.median(area[y == 1])), "median_area_irf0": float(np.median(area[y == 0])),
            "frac_any_pred_irf1": float((area[y == 1] > 0).mean()), "frac_any_pred_irf0": float((area[y == 0] > 0).mean()),
        }
        print(name, json.dumps(out[name], indent=1), flush=True)
    (ROOT / "eval" / "results" / "seg_olives_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
