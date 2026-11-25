# Treat-and-Extend Copilot: evaluation report

**What it is.** A copilot for the most common repeated decision in a retina clinic: at each
anti-VEGF injection visit for diabetic macular edema (DME), should the next interval be extended,
kept, or shortened? It reads the OCT, applies the treat-and-extend protocol, and says why. It flags
cases it can't call reliably, drafts the note, patient message, schedule and FHIR record, and waits
for the physician to approve or override.

**How it was evaluated.** The copilot is replayed against **real investigator decisions** from
the TREX-DME trial (OLIVES dataset, 115 eyes, 1,284 decisions). Every model output is
out-of-fold with patient-level 5-fold CV, so fellow eyes share a fold. The geometric CST model was
trained only on a separate dataset (Duke DME) and never saw a trial image.

## Headline results

| | Result |
|---|---|
| Retina thickness, from Duke-trained ILM→BM segmentation | 6.0 µm per-column error (inter-grader: 7.1 µm) |
| CST on trial scans (out-of-domain) | MAE 11.9 µm, r = 0.97; within-eye noise 18.3 µm (11.6 µm on QC-passing scans) |
| Engine on image-derived CST vs engine on device CST | **same decision 91%** (κ 0.87); a CNN regressor gets 81% |
| Uncertainty flag (12% of decisions) | unflagged agree 93%, flagged only 40%, so the flag catches the measurement errors |
| Written protocol vs investigators, even with perfect CST | 64% accuracy, κ 0.44 |
| Learned policy vs investigators | 69% accuracy, κ 0.52 (+0.035 to +0.129 κ, eye-level 95% CI) |

The story in one line: **image-derived measurement is no longer the bottleneck. The remaining gap
is the written protocol versus clinician judgment, and that gap is partly learnable.**

## 1. Data: recovering real clinician decisions

OLIVES releases the TREX-DME trial spreadsheets without visit dates. Its "Vision" sheets carry
fractional trial weeks but no patient IDs, so I aligned each eye's ETDRS sequence to the Vision
rows with sequence matching. 1,771 of 1,816 T&E visits got a week back, with a median match ratio of 0.977.

In a T&E arm the eye is injected at every visit, so the gap between visits *is* the investigator's
chosen interval, and the change in gap is their decision. Labels use a ±1.25 wk tolerance to
absorb scheduling jitter. They start at visit 4 (week 12), the first protocol decision.
Implausible gaps from missed visits are dropped. Result: SHORTEN 180, MAINTAIN 656, EXTEND 448.
As a sanity check, median CST/baseline falls in order: SHORTEN 0.84, MAINTAIN 0.75, EXTEND 0.69.

Images: only the TREX-DME B-scans were streamed out of the 34 GB nested, deflated archive, using
parallel HTTP range reads and inflate-on-the-fly. That gave 45,848 B-scans across 934 visits of 56 eyes.

## 2. Imaging

| Model | Data | Out-of-fold result | Reference |
|---|---|---|---|
| Fluid segmentation (U-Net/ResNet34) | Duke, 110 frames | pooled Dice 0.72 vs grader 1; detection sens. 0.90 / spec. 0.81 | inter-grader Dice 0.69 |
| Biomarker classifier (ResNet18, temperature-scaled) | OLIVES | slice AUROC IRF 0.96 / SRF 0.94 / DRT 0.97, IRF ECE 0.03 | — |
| CST regression (ResNet18) | OLIVES | MAE 29.7 µm, r 0.94 | predict-the-mean 68.7 µm |
| **Retina segmentation → geometric CST** | Duke → OLIVES | ILM/BM error 4 µm; CST MAE 11.9 µm, r 0.97 | inter-grader 7.1 µm |

**Why geometric CST.** Decisions compare CST with the eye's own reference, so a constant per-eye
bias cancels. Visit-to-visit noise does not. The CNN regressor's within-eye noise (24 µm) was about
as wide as the protocol's 10% band (~30 µm), and replaying the engine on it dropped κ from 0.45 to
0.26. Measuring thickness the way the device does fixes this. The model finds the ILM and Bruch's
membrane and averages their distance over the ETDRS 1 mm central circle. It is trained on Duke
frames resampled to the trial images' pixel scale. Uncalibrated, it is already within 8.5 µm of
device CST.

**Image QC, found by error analysis.** The median within-eye error is 4 µm, but there is a tail.
That tail is predicted by the thickness spread inside the central circle. Above 15 rows (1.4% of visits) the retina
mask has failed and the median error is ~100 µm; below it, errors are 3–15 µm. Failed-QC visits are flagged, and they never
set or move the T&E reference. Per-visit CST uncertainty is fitted out-of-fold from the same
spread.

| CST source | MAE | within-eye SD | visit-to-visit change MAE | protocol band agreement |
|---|---|---|---|---|
| CNN regression | 29.7 µm | 24.3 µm | 15.9 µm | 77% |
| Geometric | 11.9 µm | 18.3 µm | 9.0 µm | 88% |
| Geometric, QC pass (98.6%) | 10.4 µm | 11.6 µm | 6.3 µm | 91% |

## 3. Decision engine: replay against investigators

The TREX protocol is a deterministic, configurable rule engine (`configs/protocol.yaml`):
- 4 monthly loading doses, then q4w until CST ≤ 325 µm.
- Then compare CST with the reference: within +10% extend 2 wk, within +20% maintain, above that shorten 2 wk.
- Intervals run from 4 to 16 wk, and the reference re-baselines after 3 stable visits.

Copilot-only safeguards sit on top:
- A BCVA loss of 5 or more letters blocks extension.
- A flag fires when CST ± k·SD crosses a rule boundary, when P(foveal fluid) is ambiguous, or when image QC fails.
- A flagged extension is capped at MAINTAIN.

Every decision carries a plain-language trace.

Replay is teacher-forced: the interval in force before each visit is the one the trial actually
used, so every decision is scored independently. Results are on 449 decisions in imaged eyes.

| Engine inputs | Accuracy | Linear κ | Unsafe extension* | Same call as engine on device CST |
|---|---|---|---|---|
| Device CST (oracle) | 0.65 | 0.45 | 12% (7/58) | — |
| CNN CST | 0.56 | 0.26 | 14% (8/58) | 81% |
| Geometric CST | 0.60 | 0.35 | 17% (10/58) | **91%** |
| Always MAINTAIN | 0.50 | 0 | 0% | — |

\*Unsafe extension = engine EXTEND when the investigator shortened. It is a disagreement metric,
not observed harm, and with 58 shortening events, differences of 1–3 cases are noise.

**Abstention.** Widening the uncertainty band trades physician review load against accuracy on
what is left. For geometric CST, flagging 20% of decisions lifts unflagged accuracy from 0.60 to
0.62. The CNN needs about 40% flagged to reach the same 0.62. Flagging
helps less than better measurement did, because most of what remains is protocol vs
investigator, not noise.

## 4. Where the protocol and investigators disagree

Even with perfect CST, the written protocol matches investigators on 64% of all 1,284 decisions.
The two largest disagreement cells are protocol EXTEND / investigator MAINTAIN (158) and protocol
MAINTAIN / investigator EXTEND (150), so the gap runs both ways.

To ask whether that gap is predictable, I trained a gradient-boosted policy to imitate
investigators. It used only what the clinician saw: CST and its changes, BCVA and its changes, the
current interval and the visit number. Evaluation used patient-level 5-fold CV.

| Policy | Accuracy | Linear κ | Unsafe extension |
|---|---|---|---|
| Always MAINTAIN | 0.51 | 0 | 0% |
| Written TREX protocol | 0.64 | 0.44 | 10.6% |
| Logistic regression | 0.63 | 0.44 | 13.3% |
| **Gradient boosting** | **0.69** | **0.52** | 12.2% |

The gain over the protocol is real: Δaccuracy is +1.9 to +7.7 points and Δκ is +0.035 to +0.129
(95% CI, eye-level paired bootstrap). The learned policy fixes 39% of protocol disagreements but
breaks 14% of protocol agreements.

Investigators respond to two things the rules ignore (`eval/figures/learned_policy_importance.png`):
- **Absolute thickness**, not just change against the reference.
- **Interval length.** When the protocol says extend, investigators do so 70% of the time at
  4–6 wk but only 43–54% at 8 wk or more. They get more cautious as intervals lengthen.

The model is non-linear: logistic regression does no better than the rules.

**How to read this.** This is imitation, not optimisation. Agreeing with investigators is not
better vision. The learned policy does not lower the unsafe-extension rate, so it should not
replace the protocol. It is better used to explain disagreement ("investigators usually hold at
12 wk in eyes like this"), and as the starting point for a policy learned from outcomes, using
visual-acuity trajectories and treatment burden, once longitudinal outcome data is available.

## 5. Agent layer

- **One source of truth.** Every fact the LLM may state lives in a `VisitPacket` built from the
  engine. Claude drafts the Assessment & Plan and a patient message at grade 6–8 reading level.
- **Numeric grounding check.** Every number in a draft must trace to the packet, the trace or the
  schedule, and rounding is respected: 296 is accepted for 295.6, 300 is not. An ungrounded draft
  is retried once with the offending spans, then replaced by a deterministic template. The
  patient message may not claim a decision before sign-off.
- **Scheduler and FHIR.** The scheduler gives a target date ± 3 days and skips weekends and
  holidays; when no slot fits, the result says manual booking is needed. The FHIR R4 bundle holds
  Observations, a ServiceRequest, an Appointment and a Provenance. Its statuses flip from draft
  to final on physician sign-off, and the Provenance records the AI as author and the physician
  as verifier.
- **Feedback.** Approvals and overrides are logged with the engine's proposal and the reason.
  That log is the training signal for the learned policy above.

## 6. Limitations

- **Scope:** DME only (no nAMD or PED), one trial at three US sites, retrospective, 56 imaged eyes.
- **Label noise:** trial actions are noisy. They include investigator discretion and missed
  visits, and re-baselining is not recorded.
- **Geometry assumptions:** the OLIVES raster geometry (6 mm, 49 B-scans) and Duke's axial spacing
  are nominal. The fovea is assumed to be at the scan centre, with no fovea detection.
- **Fluid signal:** 93% of visits have fluid, so fluid is a weak decision signal here. The fluid
  gate only flags ambiguous probabilities.
- **Uncertainty calibration:** the band is a heuristic (±k·SD), not calibrated against decision outcomes.
- **Learned policy:** it imitates investigators, so its feature importances are descriptive and computed in-sample.
- **Not a medical device.** The physician is always the decision-maker.

## 7. What I'd do next

1. Run on full-resolution device exports and use vendor-reported CST as a QC cross-check.
2. Add nAMD with PED/SRF-specific rules: the same engine with a different `protocol.yaml`.
3. Calibrate the flag against decision flips (conformal prediction over the CST→decision map).
4. Learn the interval policy from outcomes rather than imitation: offline RL or causal methods
   over BCVA and injection burden, with the written protocol as a safe default.
5. Run a prospective silent pilot: log copilot proposals next to physician decisions before any
   recommendation is shown.

## Reproduce

```bash
python scripts/build_visit_table.py
python scripts/train_seg.py                      # Duke fluid segmentation
python scripts/train_oct.py --task biomarkers    # OLIVES biomarkers
python scripts/train_oct.py --task cst           # CNN CST
python scripts/train_layers.py                   # Duke retina seg -> geometric CST on OLIVES
python scripts/eval_cst_sources.py
python scripts/eval_engine.py && python scripts/eval_engine.py --inputs model \
  && python scripts/eval_engine.py --inputs model --cst-file data/processed/oof_cst_geo_visits.csv --tag _geo
python scripts/learned_policy.py
```
