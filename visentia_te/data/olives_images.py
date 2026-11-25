"""PyTorch datasets over the locally extracted TREX-DME OCT B-scans.

Images come from scripts/fetch_olives_trex.py (PNG, width 256) under
data/raw/olives_trex/TREX DME/<arm>/<study_id>/V<n>/<eye>/<name>.png and are indexed by
data/processed/trex_images.csv (built by scripts/build_visit_table.py).

B-scan position: each visit holds a 49-slice raster volume; slice index is
(file number - min file number in that visit/prefix) mod 49, which matches the
labeled 'Scan (n/49)' column for ~96% of labeled images (the rest are visits that
contain two volumes). The fovea is the central slice (index 24, 0-based).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_ROOT = Path("data/raw/olives_trex")
INDEX_CSV = Path("data/processed/trex_images.csv")
N_SLICES, CENTER = 49, 24
FLUID_COLS = ["Fluid (IRF)", "Fluid (SRF)", "DRT/ME"]


def load_index(index_csv: Path = INDEX_CSV, img_root: Path = IMG_ROOT, require_files: bool = True) -> pd.DataFrame:
    df = pd.read_csv(index_csv, low_memory=False)
    df["num"] = df.fname.str.extract(r"_(\d+)\.\w+$")[0].astype(float)
    df["prefix"] = df.fname.str.extract(r"^(.*)_\d+\.\w+$")[0]
    df = df[df.num.notna() & ~df.fname.str.contains("fundus", case=False)].copy()
    base = df.groupby(["study_id", "visit", "eye", "prefix"]).num.transform("min")
    df["slice"] = ((df.num - base) % N_SLICES).astype(int)
    labeled = df["Scan (n/49)"].notna()
    df.loc[labeled, "slice"] = df.loc[labeled, "Scan (n/49)"].astype(int) - 1  # trust labels when present
    df["png"] = [
        img_root / "TREX DME" / a / s / f"V{v}" / e / Path(f).with_suffix(".png")
        for a, s, v, e, f in zip(df.arm, df.study_id, df.visit, df.eye, df.fname)
    ]
    if require_files:
        df = df[df.png.map(Path.exists)]
    return df.reset_index(drop=True)


def read_bscan(path: Path, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Grayscale float32 in [0, 1], resized to (H, W)."""
    im = Image.open(path).convert("L").resize((size[1], size[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


class OlivesBscanDataset(Dataset):
    """Per-B-scan samples for biomarker classification and/or CST regression.

    task="biomarkers": only expert-labeled slices; target = multi-hot over FLUID_COLS.
    task="cst": central slices (|slice-24| <= central_halfwidth); target = visit CST (µm).
    """

    def __init__(
        self,
        index: pd.DataFrame,
        split: str,
        task: str = "biomarkers",
        size: tuple[int, int] = (224, 224),
        central_halfwidth: int = 3,
        augment: bool = False,
    ):
        df = index[index.split == split]
        if task == "biomarkers":
            df = df[df.has_biomarkers]
        elif task == "cst":
            df = df[((df.slice - CENTER).abs() <= central_halfwidth) & df.CST.notna()]
        else:
            raise ValueError(task)
        self.df, self.task, self.size, self.augment = df.reset_index(drop=True), task, size, augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        x = read_bscan(r.png, self.size)
        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, ::-1]
            x = np.clip(x ** np.random.uniform(0.8, 1.25) + np.random.normal(0, 0.02, x.shape), 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(x))[None]
        if self.task == "biomarkers":
            y = torch.tensor(r[FLUID_COLS].astype(float).values, dtype=torch.float32)
        else:
            y = torch.tensor([r.CST], dtype=torch.float32)
        meta = dict(study_id=r.study_id, visit=int(r.visit), slice=int(r.slice))
        return x, y, meta


def visit_stack(index: pd.DataFrame, study_id: str, visit: int, halfwidth: int = 3, size=(224, 224)) -> tuple[np.ndarray, list[int]]:
    """Central B-scans of one visit as (K, H, W) for inference, with their slice indices."""
    g = index[(index.study_id == study_id) & (index.visit == visit)]
    g = g[(g.slice - CENTER).abs() <= halfwidth].drop_duplicates("slice").sort_values("slice")
    return np.stack([read_bscan(p, size) for p in g.png]), g.slice.tolist()
