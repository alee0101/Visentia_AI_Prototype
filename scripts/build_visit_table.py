"""Build processed TREX-DME tables + patient-level splits, and print an EDA summary.

Outputs (data/processed/):
  trex_visits.csv     one row per eye-visit with CST, ETDRS, week, interval, trial action, split
  trex_images.csv     one row per OCT/fundus image with visit key, BCVA, CST, biomarkers, split
  splits.json         study_id -> train/val/test (patient-level; fellow eyes share a split)
  align_report.csv    quality of the week-recovery alignment per eye
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])

import numpy as np  # noqa: E402

from visentia_te.data.olives import load_image_labels, load_trex_visits, make_patient_splits  # noqa: E402

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

visits, rep = load_trex_visits()
images = load_image_labels()
splits = make_patient_splits(sorted(set(visits.study_id) | set(images.study_id)))
visits["split"] = visits.study_id.map(splits)
images["split"] = images.study_id.map(splits)
has_img = images.groupby(["study_id", "visit"]).size().rename("n_images")
visits = visits.merge(has_img, left_on=["study_id", "visit"], right_index=True, how="left")
visits["n_images"] = visits.n_images.fillna(0).astype(int)
visits["cst_ratio"] = visits.cst / visits.baseline_cst

visits.to_csv(OUT / "trex_visits.csv", index=False)
images.to_csv(OUT / "trex_images.csv", index=False)
rep.to_csv(OUT / "align_report.csv", index=False)
(OUT / "splits.json").write_text(json.dumps(splits, indent=1, sort_keys=True))

te = visits[visits.arm.isin(["TREX", "GILA"])]
dec = te[te.action.notna()]
print(f"eyes: {visits.study_id.nunique()} ({visits.groupby('arm').study_id.nunique().to_dict()})")
print(f"T&E visits: {len(te)}, with recovered week: {te.week.notna().sum()}, decision visits: {len(dec)}")
print(f"week alignment: median match {rep.match_ratio.median():.3f}, min {rep.match_ratio.min():.3f}")
print("trial actions:", dec.action.value_counts().to_dict())
print("decision visits with OCT images, by split:")
print(dec[dec.n_images > 0].groupby("split").action.value_counts().unstack().fillna(0).astype(int))
print(f"images: {len(images)} ({images.has_biomarkers.sum()} with biomarker labels), eyes with images: {images.study_id.nunique()}")
print("split sizes (patients):", {s: len({k[:4] for k, v in splits.items() if v == s}) for s in ("train", "val", "test")})
print("CST ratio-to-baseline by trial action (median):", dec.groupby("action").cst_ratio.median().round(3).to_dict())
print("interval distribution (wk):", np.histogram(te.interval_wk.dropna(), bins=[2, 5, 7, 9, 11, 13, 17, 21])[0].tolist())
