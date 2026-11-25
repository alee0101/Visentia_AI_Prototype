"""B-scan models trained on OLIVES / TREX-DME.

Two tasks share one backbone recipe (timm ResNet, ImageNet init, 1-channel input):
  * biomarkers: multi-label sigmoid over FLUID_COLS (IRF, SRF, DRT/ME), on expert-labeled slices
  * cst: regression of the visit's central subfield thickness (µm) from central slices;
         a visit's CST estimate is the mean over its central slices

Evaluation is 5-fold patient-level cross-validation so that *every* visit gets an
out-of-fold (OOF) prediction; the decision engine is later scored on those OOF outputs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from visentia_te.data.olives import patient_of
from visentia_te.data.olives_images import CENTER, FLUID_COLS, read_bscan

SIZE = (256, 256)
N_FOLDS = 5


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def assign_folds(study_ids, n_folds: int = N_FOLDS, seed: int = 7) -> dict[str, int]:
    """Patient-level folds (fellow eyes share a fold)."""
    pats = sorted({patient_of(s) for s in study_ids})
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    pf = {p: i % n_folds for i, p in enumerate(pats)}
    return {s: pf[patient_of(s)] for s in study_ids}


def make_model(n_out: int, arch: str = "resnet18") -> nn.Module:
    try:
        return timm.create_model(arch, pretrained=True, in_chans=1, num_classes=n_out)
    except Exception as e:  # offline fallback
        print(f"[warn] pretrained weights unavailable ({e}); training from scratch")
        return timm.create_model(arch, pretrained=False, in_chans=1, num_classes=n_out)


def preload(paths) -> np.ndarray:
    """uint8 (N, H, W) cache; B-scans are tiny, so RAM beats a DataLoader on macOS."""
    return np.stack([(read_bscan(p, SIZE) * 255).astype(np.uint8) for p in paths])


def _augment(x: torch.Tensor) -> torch.Tensor:
    """x: (B,1,H,W) float on device. Flip, gamma, gain, speckle, small vertical shift."""
    b = x.shape[0]
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    gamma = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.75, 1.33)
    gain = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
    x = (x.clamp_min(1e-4) ** gamma) * gain
    x = x * (1 + 0.08 * torch.randn_like(x)) + 0.02 * torch.randn_like(x)
    shift = int(torch.randint(-12, 13, (1,)))
    x = torch.roll(x, shifts=shift, dims=-2)  # retina position varies in the scan window
    return x.clamp(0, 1)


def _norm(x_u8: np.ndarray, dev) -> torch.Tensor:
    x = torch.from_numpy(x_u8).to(dev).float().div_(255.0)[:, None]
    return x


@dataclass
class TrainCfg:
    task: str  # "biomarkers" | "cst"
    epochs: int = 12
    batch: int = 48
    lr: float = 3e-4
    wd: float = 1e-4
    arch: str = "resnet18"
    seed: int = 0


def train_one(cfg: TrainCfg, Xtr, Ytr, Xva, Yva, log_prefix: str = "") -> tuple[nn.Module, dict]:
    """Train with AdamW + cosine; keep the best epoch on the validation fold."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = device()
    n_out = Ytr.shape[1]
    model = make_model(n_out, cfg.arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    steps = cfg.epochs * int(np.ceil(len(Xtr) / cfg.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.15)
    if cfg.task == "biomarkers":
        pos = Ytr.mean(0).clip(0.02, 0.98)
        pos_weight = torch.tensor((1 - pos) / pos, dtype=torch.float32, device=dev).clamp(max=10)
    best, best_state, hist = np.inf, None, []
    for ep in range(cfg.epochs):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(perm), cfg.batch):
            j = perm[i : i + cfg.batch]
            x = _augment(_norm(Xtr[j], dev))
            y = torch.from_numpy(Ytr[j]).to(dev)
            out = model(x)
            if cfg.task == "biomarkers":
                loss = F.binary_cross_entropy_with_logits(out, y, pos_weight=pos_weight)
            else:
                loss = F.smooth_l1_loss(out, y, beta=0.5)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(j)
        va_out = predict_raw(model, Xva)
        if cfg.task == "biomarkers":
            va_loss = float(F.binary_cross_entropy_with_logits(torch.from_numpy(va_out), torch.from_numpy(Yva)))
        else:
            va_loss = float(np.abs(va_out - Yva).mean())
        hist.append(dict(epoch=ep, train_loss=tot / len(Xtr), val_loss=va_loss, sec=time.time() - t0))
        print(f"{log_prefix} ep{ep:02d} train {tot / len(Xtr):.4f} val {va_loss:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if va_loss < best:
            best, best_ep = va_loss, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, dict(best_val=best, best_epoch=best_ep, history=hist)


@torch.no_grad()
def predict_raw(model: nn.Module, X: np.ndarray, batch: int = 128, tta: bool = True) -> np.ndarray:
    """Logits (biomarkers) or standardized values (cst); horizontal-flip TTA."""
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    for i in range(0, len(X), batch):
        x = _norm(X[i : i + batch], dev)
        o = model(x)
        if tta:
            o = (o + model(x.flip(-1))) / 2
        outs.append(o.float().cpu().numpy())
    return np.concatenate(outs)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-output temperature scaling on validation logits (calibration for the rule engine)."""
    temps = []
    for k in range(logits.shape[1]):
        lg, yy = torch.tensor(logits[:, k]), torch.tensor(y[:, k])
        t = torch.ones(1, requires_grad=True)
        opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(lg / t.clamp_min(0.05), yy)
            loss.backward()
            return loss

        opt.step(closure)
        temps.append(float(t.clamp_min(0.05)))
    return np.array(temps, dtype=np.float32)


def central_mask(df: pd.DataFrame, halfwidth: int) -> pd.Series:
    return (df.slice - CENTER).abs() <= halfwidth


__all__ = ["FLUID_COLS", "TrainCfg", "assign_folds", "fit_temperature", "make_model", "predict_raw", "preload", "train_one"]
