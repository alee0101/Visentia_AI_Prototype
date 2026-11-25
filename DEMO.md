# Demo script (~7 minutes)

```bash
.venv/bin/streamlit run app/streamlit_app.py      # http://localhost:8501
```
Keep "Draft with Claude" off unless `.env` has a key. The templates are instant and need no network.

## 0. Framing (30 s)
"Treat-and-extend is the decision a retina clinic makes dozens of times a day: extend, keep, or
shorten the injection interval. It's mostly measurement plus a rule, which makes it a good place
for an AI that measures, explains and drafts, while the physician decides. I validated it against
real investigator decisions from a clinical trial, not synthetic labels."

## 1. A flagged visit (2 min), eye **0232TOS**, opens on **V9**
- Top row: CST is +20% vs the reference, BCVA is 1 letter below best, the current interval is 4 wk.
- The card reads **⚑ Physician review, proposal SHORTEN**. Read the trace: CST ±1.5·SD spans
  +18%..+22%, right on the 20% line, so the copilot will not pretend to be sure.
- The investigator actually chose MAINTAIN, which is exactly a judgment call.
- The scan shows yellow ILM, aqua Bruch's membrane and red fluid. Move the slice slider. "Thickness
  is measured the way the device measures it, not guessed by a black box."
- Drafts: show the clinic note, then **Grounding check**: every number in the draft traces to the
  engine's facts. Then the patient message, which never says "scheduled" before sign-off.
- Sign-off: Override → MAINTAIN, 4 wk, reason "stable VA, borderline thickening". It shows
  OVERRIDDEN, FHIR flips to final/booked, and the patient message is regenerated. "The override,
  with its reason, is logged: that's the training data for learning what clinicians do differently."

## 2. Why measurement matters (1.5 min), eye **0208TOD**, **V7**
- Geometric source: CST 235 vs reference 235, so EXTEND, which matches the investigator.
- Switch the sidebar to **CNN regression**. CST reads 328 (a ~90 µm overestimate), and the
  proposal becomes a flag.
- "The first model I trained regressed thickness directly. 30 µm error sounds fine, but the
  decision band is only 30 µm wide. So I trained a retina segmentation on a different public
  dataset and measured the distance between the boundaries instead. Visit-to-visit noise halved,
  and the engine now makes the same call as with device CST 91% of the time instead of 81%."
- Also on this eye, **V4**: flagged because P(foveal fluid) = 0.46 is ambiguous, an example of a
  non-thickness flag.

## 3. Eye timeline tab (1 min)
- Blue is device CST and orange is the copilot input. They track closely. The dashed line is the
  T&E reference, with the +10% / +20% protocol lines.
- The strip below compares copilot and investigator decisions, with ⚑ marking flags. "Agreement
  for this eye is 53%. Even with perfect measurement the written protocol only matches
  investigators 65% of the time."

## 4. Validation tab (1.5 min)
- Imaging: all models are at or near inter-grader agreement.
- CST table: within-eye SD drops from 24 µm (CNN) to 12 µm (geometric with QC), and that is the
  number that matters.
- Engine replay: same call as the device-CST engine, 81% → 91%.
- The abstention curve shows physician review load against accuracy.
- Finding to close on (`eval/REPORT.md` §4): a learned policy predicts investigators better than
  the protocol (κ 0.44 → 0.52, CI excludes 0). Investigators react to absolute thickness and get
  cautious at long intervals, which the rules ignore. "That's imitation, not better care. The next
  step is to learn from outcomes, and the feedback log is where that data comes from."

## Likely questions
- **Why not end-to-end OCT → action?** There are 1,284 noisy labels, and clinicians need a
  reason. Measurement plus a rule is auditable and swappable per protocol (`configs/protocol.yaml`),
  and the error analysis can say exactly which part failed.
- **Leakage?** All metrics use patient-level 5-fold CV with fellow eyes together. The geometric CST
  never saw OLIVES; only its linear µm calibration is fitted, out-of-fold.
- **Unsafe extensions?** 10/58 vs 7/58 for device CST. The sample is too small to separate the two,
  and it's a disagreement metric, not harm. Flagged extensions are always capped at MAINTAIN.
- **nAMD?** Same pipeline: add a PED/SRF-aware rule set and fluid-type-specific gates.
