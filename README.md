# Treat-and-Extend Copilot (Visentia prototype)

An OCT-driven decision-support prototype for **anti-VEGF treat-and-extend (T&E) dosing in diabetic macular edema**.
It measures retinal fluid and thickness from OCT, tracks each eye over time, recommends
**EXTEND / MAINTAIN / SHORTEN / FLAG** with an auditable rule trace, then drafts the note, proposes the next
appointment and emits FHIR resources. The physician approves or overrides every recommendation.

> Models *measure*, a transparent configurable protocol *decides*, an LLM only *writes*.

## Data

| Source | Use | Notes |
|---|---|---|
| [OLIVES](https://zenodo.org/records/7105232) (CC-BY-4.0), TREX-DME trial | longitudinal ground truth + OCT | 150 eyes, 3 arms (Monthly / TREX / GILA). 56 eyes have OCT volumes (49 B-scans/visit). |
| [Duke DME (Chiu 2015)](https://people.duke.edu/~sf59/software.html) | fluid segmentation | 110 expert-annotated B-scans, 10 patients, 2 graders |

### Key data finding: real clinician decisions are recoverable
The TREX-DME spreadsheets blank visit dates, but the per-arm "Vision" sheets carry fractional weeks per visit
without IDs. Aligning ETDRS sequences recovers weeks for **1,771 / 1,816** T&E visits (median sequence match 0.98).
Because T&E injects at every visit, the inter-visit gap *is* the chosen treatment interval, so the investigator's
action at each visit is Δinterval (±1.25 wk tolerance for scheduling jitter; loading phase excluded):

| Trial action | n | median CST / baseline |
|---|---|---|
| EXTEND | 393 | 0.69 |
| MAINTAIN | 602 | 0.75 |
| SHORTEN | 174 | 0.84 |

407 decision visits have OCT images (patient-level split: 217 train / 81 val / 109 test).

## Results so far (all out-of-fold, patient-level 5-fold CV)

**Image models**

| Model | Metric | Result | Reference |
|---|---|---|---|
| Fluid segmentation (Duke) | pooled Dice vs grader 1 | 0.72 | grader 1 vs 2: 0.69 |
| Retina ILM→BM segmentation (Duke) | per-column thickness MAE | 6.0 µm | grader 1 vs 2: 7.1 µm |
| Biomarkers (OLIVES) | slice AUROC IRF / SRF / DRT | 0.96 / 0.94 / 0.97 | — |
| CST, CNN regression | MAE / within-eye SD | 29.7 / 24.3 µm | predict-mean 68.7 µm |
| **CST, geometric (Duke-trained, never saw OLIVES)** | MAE / within-eye SD | **11.9 / 18.3 µm** | r = 0.97 |
| CST, geometric, image-QC pass (98.6%) | MAE / within-eye SD / band agreement | **10.4 / 11.6 µm / 0.91** | CNN band agreement 0.77 |

Image QC: thickness spread inside the 1 mm central circle above 15 rows means the retina mask
failed (median error ~100 µm vs 3-15 µm otherwise); such visits are flagged and never set or
move the T&E reference. Per-visit CST uncertainty is fitted out-of-fold from the same spread.

**Engine replay vs investigators** (449 imaged decisions, teacher-forced intervals)

| Engine inputs | Accuracy | Linear κ | Same call as engine on trial CST |
|---|---|---|---|
| Trial-measured CST (oracle) | 0.65 | 0.45 | — |
| CNN CST | 0.56 | 0.26 | 0.81 (κ 0.68) |
| **Geometric CST** | **0.60** | **0.35** | **0.91 (κ 0.87)** |
| Geometric CST + uncertainty flag (12% flagged) | 0.60 unflagged | — | 0.93 unflagged / 0.40 flagged |

Reading: geometric thickness removes most of the measurement-attributable error, and the flag
concentrates what remains (flagged decisions are the ones that disagree). The remaining gap to
investigators is protocol vs investigator judgment, which exists even with perfect CST.

Reproduce: `scripts/train_seg.py`, `scripts/train_oct.py --task biomarkers|cst`,
`scripts/train_layers.py`, `scripts/eval_cst_sources.py`, `scripts/eval_engine.py [--inputs model --cst-file ... --tag ...]`.

## Layout
```
visentia_te/data/     olives.py (trial tables, splits), olives_images.py (B-scan datasets), duke.py
visentia_te/imaging/  fluid segmentation, biomarker classifier, CST regression, retina layers (geometric CST)
visentia_te/engine/   EyeState + T&E rule engine (configs/protocol.yaml)
visentia_te/agent/    visit packet, note + patient message (Claude, numeric grounding check, template fallback),
                      scheduler, FHIR R4 bundle, physician approve/override -> feedback.jsonl
app/                  Streamlit UI
scripts/              fetch_olives_trex.py, build_visit_table.py, duke_sanity.py
eval/                 metrics + figures
```

## Setup
```bash
# Apple Silicon: use a native arm64 Python (an x86/Rosetta Python has no torch wheels and no MPS)
/opt/homebrew/bin/python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[fetch,dev]"
# labels (10 MB)
curl -L -o data/raw/olives_labels.zip "https://zenodo.org/api/records/7105232/files/OLIVES_Dataset_Labels.zip/content"
unzip data/raw/olives_labels.zip -d data/raw/olives_labels
# TREX-DME B-scans only, streamed from the 34 GB archive (~25 min, ~4 GB on disk)
python scripts/fetch_olives_trex.py --out data/raw/olives_trex --max-width 256
python scripts/build_visit_table.py
pytest
```
macOS note: if the repo lives in an iCloud-synced folder (e.g. Desktop), the editable-install
`.pth` file can get the `hidden` flag, which Python 3.13 skips. Scripts and pytest add the repo
root to `sys.path` themselves, so this only affects ad-hoc `import visentia_te` from elsewhere
(`chflags nohidden .venv/lib/python3.13/site-packages/*.pth` fixes it).

## Limitations (running list)
- DME only (no wet AMD / PED); single trial, 3 US sites.
- Trial actions are noisy labels (investigator discretion, missed visits, re-baselining not recorded).
- OLIVES images are downsampled to 256 px wide for disk reasons.
- Duke axial spacing (3.87 µm/px) and the OLIVES 6 mm raster geometry are nominal assumptions;
  geometric CST is linearly calibrated out-of-fold, and the raw (uncalibrated) value is already within 8.5 µm bias.
- Fovea assumed at the scan centre (fixation-centred raster); no explicit fovea detection.
- Not a medical device; decision support with a physician in the loop.
