"""Auditable treat-and-extend decision engine.

Implements the TREX-DME T&E protocol (configs/protocol.yaml) plus copilot safety rules.
Every decision returns a trace of the rules that fired and the measurements they used,
so a physician can see *why* -- the engine never calls an ML model or an LLM itself;
it consumes measurements (true or model-estimated) and their uncertainty.

Protocol interpretation (TREX-DME):
  * 4 monthly loading injections; then continue q4w until CST <= 325 µm.
  * The *reference* CST is set when T&E starts. Each visit compares CST to it:
      CST <= ref * 1.10 -> extend +2 wk; <= ref * 1.20 -> maintain; else shorten -2 wk.
  * Re-baseline: if CST stays >10% below ref for 3 consecutive visits with <= 50 µm
    spread, their mean becomes the new reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

EXTEND, MAINTAIN, SHORTEN, FLAG, LOADING = "EXTEND", "MAINTAIN", "SHORTEN", "FLAG", "LOADING"


def load_protocol(path: str | Path = "configs/protocol.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


@dataclass
class Observation:
    visit: int
    cst: float  # µm (measured or model estimate)
    bcva: float | None = None  # ETDRS letters
    cst_sd: float = 0.0  # uncertainty of the CST estimate (µm); 0 for ground truth
    p_fluid: float | None = None  # P(fluid present), calibrated
    image_quality_ok: bool = True


@dataclass
class Decision:
    action: str  # protocol action (EXTEND/MAINTAIN/SHORTEN/LOADING)
    next_interval_wk: int
    flagged: bool
    trace: list[str]
    recommended: str  # what the copilot shows: action, or FLAG when physician review is required
    ratio: float | None = None


@dataclass
class EyeState:
    protocol: dict
    interval_wk: int = 4
    ref_cst: float | None = None
    te_started: bool = False
    n_visits: int = 0
    best_bcva: float | None = None
    low_run: list[float] = field(default_factory=list)  # CSTs >10% below ref, for re-baselining

    def step(self, obs: Observation) -> Decision:
        p, iv = self.protocol, self.protocol["interval"]
        r, s = p["rules"], p["safety"]
        self.n_visits += 1
        trace: list[str] = []
        if obs.bcva is not None:
            self.best_bcva = obs.bcva if self.best_bcva is None else max(self.best_bcva, obs.bcva)

        # 1. loading phase / CST gate --------------------------------------------------------
        if obs.visit < p["loading_doses"]:  # the 4th dose visit is the first decision (week-12 CST gate)
            trace.append(f"loading dose {obs.visit}/{p['loading_doses']} -> q4w")
            return Decision(LOADING, 4, False, trace, LOADING)
        if not self.te_started:
            if not obs.image_quality_ok:  # never anchor the reference on an unreliable measurement
                trace.append("image quality insufficient: cannot set T&E reference -> continue q4w")
                return Decision(MAINTAIN, 4, True, trace, FLAG)
            if obs.cst > p["loading_cst_gate_um"]:
                trace.append(f"CST {obs.cst:.0f} µm > {p['loading_cst_gate_um']} gate -> continue q4w")
                return Decision(MAINTAIN, 4, self._uncertain_gate(obs, trace), trace, MAINTAIN)
            self.te_started, self.ref_cst = True, obs.cst
            trace.append(f"T&E start: reference CST = {obs.cst:.0f} µm")

        # 2. protocol rule on CST change vs reference ---------------------------------------
        ratio = obs.cst / self.ref_cst
        change = ratio - 1
        if change <= r["extend_if_within"]:
            action = EXTEND
            trace.append(f"CST {obs.cst:.0f} vs ref {self.ref_cst:.0f} ({change:+.1%}) <= +{r["extend_if_within"]:.0%} -> extend")
        elif change <= r["maintain_if_within"]:
            action = MAINTAIN
            trace.append(f"CST {change:+.1%} vs ref within +{r['maintain_if_within']:.0%} -> maintain")
        else:
            action = SHORTEN
            trace.append(f"CST {change:+.1%} vs ref > +{r['shorten_if_above']:.0%} -> shorten")

        # 3. copilot safety overrides (not in trial protocol) --------------------------------
        flagged = False
        if obs.bcva is not None and self.best_bcva is not None:
            loss = self.best_bcva - obs.bcva
            if loss >= s["bcva_loss_letters_shorten"] and action == EXTEND:
                action = MAINTAIN
                trace.append(f"safety: BCVA {loss:.0f} letters below best -> do not extend")
        # uncertainty: would a ±1.5 sd CST error move us across a rule boundary?
        if obs.cst_sd > 0:
            lo, hi = (obs.cst - 1.5 * obs.cst_sd) / self.ref_cst - 1, (obs.cst + 1.5 * obs.cst_sd) / self.ref_cst - 1
            bands = {self._band(lo), self._band(hi)}
            if len(bands) > 1:
                flagged = True
                trace.append(f"uncertain: CST ±1.5sd spans {lo:+.0%}..{hi:+.0%}, crosses a rule boundary")
        if obs.p_fluid is not None:
            a, b = s["fluid_uncertain_band"]
            if a <= obs.p_fluid <= b and action == EXTEND:
                flagged = True
                trace.append(f"uncertain: P(foveal fluid) = {obs.p_fluid:.2f} in [{a}, {b}]")
        if not obs.image_quality_ok:
            flagged = True
            trace.append("image quality insufficient")
        if flagged and s.get("never_extend_when_flagged") and action == EXTEND:
            trace.append("flagged -> proposal capped at MAINTAIN pending physician review")
            action = MAINTAIN

        # 4. interval arithmetic + re-baselining ---------------------------------------------
        step = iv["step_wk"]
        nxt = {EXTEND: self.interval_wk + step, MAINTAIN: self.interval_wk, SHORTEN: self.interval_wk - step}[action]
        nxt = int(min(iv["max_wk"], max(iv["min_wk"], nxt)))
        if nxt == self.interval_wk and action != MAINTAIN:
            trace.append(f"interval at {'max' if action == EXTEND else 'min'} ({nxt} wk) -> held")
        self.interval_wk = nxt
        if obs.image_quality_ok:  # QC-failed measurements never move the reference
            self._rebaseline(obs.cst, trace)
        return Decision(action, nxt, flagged, trace, FLAG if flagged else action, ratio)

    def _band(self, change: float) -> str:
        r = self.protocol["rules"]
        return EXTEND if change <= r["extend_if_within"] else MAINTAIN if change <= r["maintain_if_within"] else SHORTEN

    def _uncertain_gate(self, obs: Observation, trace: list[str]) -> bool:
        if obs.cst_sd > 0 and obs.cst - 1.5 * obs.cst_sd <= self.protocol["loading_cst_gate_um"]:
            trace.append("uncertain: CST estimate near the 325 µm gate")
            return True
        return False

    def _rebaseline(self, cst: float, trace: list[str]) -> None:
        rb = self.protocol["rebaseline"]
        if cst < self.ref_cst * (1 - self.protocol["rules"]["extend_if_within"]):
            self.low_run.append(cst)
            run = self.low_run[-rb["stable_visits"]:]
            if len(run) == rb["stable_visits"] and max(run) - min(run) <= rb["max_variability_um"]:
                new = sum(run) / len(run)
                trace.append(f"re-baseline: reference {self.ref_cst:.0f} -> {new:.0f} µm (3 stable visits)")
                self.ref_cst, self.low_run = new, []
        else:
            self.low_run = []
