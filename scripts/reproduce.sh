#!/usr/bin/env bash
# Rebuild every result from raw data (≈2.5 h on an Apple M4; downloads ≈4 GB).
# Assumes the arm64 venv from README "Setup" and the OLIVES labels unzipped in data/raw/olives_labels.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -d "data/raw/olives_trex/TREX DME" ] || $PY scripts/fetch_olives_trex.py --out data/raw/olives_trex --max-width 256
$PY scripts/build_visit_table.py
$PY scripts/train_seg.py                       # Duke fluid segmentation (5-fold + final)
$PY scripts/train_oct.py --task biomarkers     # OLIVES IRF/SRF/DRT
$PY scripts/train_oct.py --task cst --epochs 10
$PY scripts/train_layers.py --epochs 35        # Duke retina seg -> geometric CST on OLIVES
$PY scripts/eval_cst_sources.py
$PY scripts/eval_engine.py
$PY scripts/eval_engine.py --inputs model
$PY scripts/eval_engine.py --inputs model --cst-file data/processed/oof_cst_geo_visits.csv --tag _geo
$PY scripts/eval_engine.py --inputs model --cst-file data/processed/oof_cst_stack_visits.csv --tag _stack
$PY scripts/learned_policy.py
$PY -m pytest -q
