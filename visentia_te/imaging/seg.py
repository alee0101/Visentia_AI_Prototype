"""Binary retinal-fluid segmentation (U-Net / ResNet34) for Spectralis B-scans.

Trained on Duke DME (see visentia_te/data/duke.py, scripts/train_seg.py).

Scale note: Duke crops are ~496 x 520 px (3.87 um axial, ~11.5 um lateral, so
roughly 1.9 x 6 mm) and OLIVES/TREX B-scans are 496 x 512 originals covering
~6 mm. Both are resized to 256 x 256 here, so the physical scale seen by the
network is comparable across the two sources.

Loss choice: BCE (no pos_weight) + soft Dice computed over the whole batch.
Fluid is ~1.2% of pixels; the Dice term already handles the imbalance, and a
large pos_weight makes the BCE favour false positives and skews probabilities.
The decision threshold is tuned on a validation subject afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN_SIZE: tuple[int, int] = (256, 256)


def build_model(pretrained: bool = True) -> tuple[nn.Module, bool]:
    """U-Net with a ResNet34 encoder, 1 input channel and 1 logit output.

    Returns (model, used_pretrained). Falls back to random init if the ImageNet
    weights cannot be downloaded.
    """
    if pretrained:
        try:
            return smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1), True
        except Exception as e:  # network / hub failure
            print(f"[seg] ImageNet weights unavailable ({e!r}); using random init")
    return smp.Unet("resnet34", encoder_weights=None, in_channels=1, classes=1), False


class DiceBCELoss(nn.Module):
    """BCE-with-logits + (1 - batch soft Dice)."""

    def __init__(self, bce_weight: float = 1.0, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce_weight, self.smooth = bce_weight, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum()
        dice = (2 * inter + self.smooth) / (p.sum() + target.sum() + self.smooth)
        return self.bce_weight * bce + (1 - dice)


@torch.no_grad()
def predict(model: nn.Module, image: np.ndarray, size: tuple[int, int] = TRAIN_SIZE,
            tta_flip: bool = True) -> np.ndarray:
    """Fluid probability map (H, W) for a grayscale float [0, 1] B-scan of any size.

    Resizes to the training size, runs the model (optionally averaging with a
    horizontal flip) and resizes the probabilities back to the input shape.
    """
    model.eval()
    dev = next(model.parameters()).device
    x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None].to(dev)
    h, w = x.shape[-2:]
    xr = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    logit = model(xr)
    if tta_flip:
        logit = (logit + model(xr.flip(-1)).flip(-1)) / 2
    prob = F.interpolate(torch.sigmoid(logit), size=(h, w), mode="bilinear", align_corners=False)
    return prob[0, 0].float().cpu().numpy()


@dataclass(frozen=True)
class PostProc:
    """Threshold plus removal of connected components smaller than ``min_size`` pixels.

    ``min_size`` is in pixels at the resolution the probability map is given in
    (Duke native crop resolution during tuning).
    """

    threshold: float = 0.5
    min_size: int = 0

    def __call__(self, prob: np.ndarray) -> np.ndarray:
        return postprocess(prob, self.threshold, self.min_size)


def postprocess(prob: np.ndarray, threshold: float = 0.5, min_size: int = 0) -> np.ndarray:
    """Binary uint8 mask: ``prob > threshold``, dropping 8-connected blobs < min_size px."""
    mask = prob > threshold
    if min_size > 0 and mask.any():
        lab, n = ndi.label(mask, structure=np.ones((3, 3), bool))
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        keep = sizes >= min_size
        keep[0] = False
        mask = keep[lab]
    return mask.astype(np.uint8)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice of two binary masks; NaN if both are empty."""
    s = int(a.sum()) + int(b.sum())
    return float(2 * np.logical_and(a, b).sum() / s) if s else float("nan")


def save_checkpoint(path, model: nn.Module, **meta) -> None:
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def load_checkpoint(path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Load a checkpoint written by ``save_checkpoint``; returns (model, meta)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model, _ = build_model(pretrained=False)
    model.load_state_dict(ck.pop("state_dict"))
    return model.to(device).eval(), ck
