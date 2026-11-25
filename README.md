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

## Layout
```
visentia_te/data/     olives.py (trial tables, splits), olives_images.py (B-scan datasets), duke.py
visentia_te/imaging/  fluid segmentation, biomarker classifier, CST regression
visentia_te/engine/   EyeState + T&E rule engine (configs/protocol.yaml)
visentia_te/agent/    note drafting w/ numeric grounding check, scheduler, FHIR
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

## Limitations (running list)
- DME only (no wet AMD / PED); single trial, 3 US sites.
- Trial actions are noisy labels (investigator discretion, missed visits, re-baselining not recorded).
- OLIVES images are downsampled to 256 px wide for disk reasons.
- Not a medical device; decision support with a physician in the loop.
