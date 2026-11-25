"""VisitPacket: the single source of truth for every fact the agent layer may use.

The LLM never sees raw images, raw tables or the engine object -- only a VisitPacket (plus the
proposed schedule). The numeric grounding check (grounding.py) verifies generated text against
exactly these facts, so anything not in the packet is, by construction, an invented number.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Sequence

from visentia_te.engine.protocol import EXTEND, FLAG, LOADING, MAINTAIN, SHORTEN, Decision, EyeState, Observation

ACTIONS = (EXTEND, MAINTAIN, SHORTEN, LOADING)
_REF_RE = re.compile(r"(?:vs ref|reference CST =)\s*(\d+(?:\.\d+)?)")
_REBASE_RE = re.compile(r"re-baseline: reference \d+(?:\.\d+)? -> (\d+(?:\.\d+)?)")


def _is_flag_reason(line: str) -> bool:
    return line.startswith("uncertain") or "image quality" in line


def _eye_from_id(study_id: str) -> str | None:
    """OLIVES study ids end in the eye (e.g. 0201GOD -> OD, 0201TOS -> OS)."""
    s = study_id.upper()
    return "OD" if s.endswith("OD") else "OS" if s.endswith("OS") else None


@dataclass(frozen=True)
class VisitPacket:
    study_id: str
    visit: int
    cst_um: float  # current central subfield thickness (measured or model estimate)
    action: str  # protocol action from the engine (EXTEND/MAINTAIN/SHORTEN/LOADING)
    recommended: str  # what the copilot shows: action, or FLAG when physician review is required
    next_interval_wk: int  # engine-recommended interval to the next injection
    eye: str | None = None  # "OD" / "OS"
    visit_date: date | None = None
    ref_cst_um: float | None = None  # reference CST the rule compared against
    ratio: float | None = None  # cst / ref
    bcva_letters: float | None = None  # ETDRS letters
    best_bcva_letters: float | None = None
    fluid_probs: Mapping[str, float] = field(default_factory=dict)  # e.g. {"any_fluid": 0.42}
    cst_sd_um: float = 0.0  # CST estimate uncertainty (0 = ground truth)
    current_interval_wk: int | None = None  # interval that led to this visit
    flagged: bool = False
    flag_reasons: tuple[str, ...] = ()
    trace: tuple[str, ...] = ()
    protocol: str = "trex_dme"

    # ---- derived facts (still deterministic functions of the packet) ----------------------
    @property
    def cst_change_pct(self) -> float | None:
        return None if self.ratio is None else (self.ratio - 1) * 100

    @property
    def bcva_loss_letters(self) -> float | None:
        if self.bcva_letters is None or self.best_bcva_letters is None:
            return None
        return self.best_bcva_letters - self.bcva_letters

    @property
    def interval_change_wk(self) -> int | None:
        return None if self.current_interval_wk is None else self.next_interval_wk - self.current_interval_wk

    def facts(self) -> dict[str, Any]:
        """Flat, JSON-safe dict of every fact the LLM may cite (None values dropped)."""
        d = {
            "study_id": self.study_id, "eye": self.eye, "visit_number": self.visit,
            "visit_date": self.visit_date.isoformat() if self.visit_date else None,
            "cst_um": _r(self.cst_um, 0), "cst_sd_um": _r(self.cst_sd_um, 0) if self.cst_sd_um else None,
            "reference_cst_um": _r(self.ref_cst_um, 0), "cst_change_vs_reference_pct": _r(self.cst_change_pct, 0),
            "bcva_etdrs_letters": _r(self.bcva_letters, 0), "best_bcva_etdrs_letters": _r(self.best_bcva_letters, 0),
            "bcva_letters_below_best": _r(self.bcva_loss_letters, 0),
            "fluid_probability": {k: round(v, 2) for k, v in self.fluid_probs.items()} or None,
            "current_interval_weeks": self.current_interval_wk, "engine_protocol_action": self.action,
            "copilot_recommendation": self.recommended, "recommended_next_interval_weeks": self.next_interval_wk,
            "flagged_for_physician_review": self.flagged, "flag_reasons": list(self.flag_reasons) or None,
            "protocol": self.protocol, "engine_trace": list(self.trace),
        }
        return {k: v for k, v in d.items() if v is not None}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["visit_date"] = self.visit_date.isoformat() if self.visit_date else None
        d["fluid_probs"] = dict(self.fluid_probs)
        return d


def _r(x: float | None, nd: int) -> float | int | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return int(round(x)) if nd == 0 else round(x, nd)


@dataclass(frozen=True)
class PhysicianDecision:
    """Physician review outcome. `action`/`interval_wk` None on approve = accept the engine's values."""
    physician_id: str
    approved: bool = True  # False = override
    action: str | None = None
    interval_wk: int | None = None
    reason: str = ""
    physician_name: str | None = None
    decided_at: datetime | None = None

    @property
    def is_override(self) -> bool:
        return not self.approved


# ---- builders ---------------------------------------------------------------------------------
def from_engine(state: EyeState, obs: Observation, decision: Decision, *, study_id: str,
                prior_interval_wk: int | None = None, visit_date: date | None = None,
                fluid_probs: Mapping[str, float] | None = None, eye: str | None = None) -> VisitPacket:
    """Build a packet *after* `state.step(obs)` returned `decision`.

    `state.step` mutates `interval_wk`, so pass the interval in force before the step as
    `prior_interval_wk` (or use `step_and_pack`). The reference CST is recovered from the
    decision ratio, so it is the reference the rule actually used even if a re-baseline fired.
    """
    ref = obs.cst / decision.ratio if decision.ratio else state.ref_cst
    fluid = dict(fluid_probs or {})
    if obs.p_fluid is not None:
        fluid.setdefault("any_fluid", obs.p_fluid)
    return VisitPacket(
        study_id=study_id, eye=eye or _eye_from_id(study_id), visit=obs.visit, visit_date=visit_date,
        cst_um=float(obs.cst), ref_cst_um=ref, ratio=decision.ratio, bcva_letters=obs.bcva,
        best_bcva_letters=state.best_bcva, fluid_probs=fluid, cst_sd_um=float(obs.cst_sd),
        current_interval_wk=prior_interval_wk, action=decision.action, recommended=decision.recommended,
        next_interval_wk=int(decision.next_interval_wk), flagged=decision.flagged,
        flag_reasons=tuple(t for t in decision.trace if _is_flag_reason(t)), trace=tuple(decision.trace),
        protocol=state.protocol.get("name", "trex_dme"),
    )


def step_and_pack(state: EyeState, obs: Observation, *, study_id: str, **kw) -> VisitPacket:
    """Run the engine on one visit and pack the result (captures the pre-step interval)."""
    prior = state.interval_wk
    return from_engine(state, obs, state.step(obs), study_id=study_id, prior_interval_wk=prior, **kw)


def _infer_prior_interval(action: str, nxt: int, trace: Sequence[str], step: int = 2) -> int:
    """Invert the interval arithmetic when the replay row doesn't carry the prior interval."""
    if any("-> held" in t for t in trace) or action in (MAINTAIN, LOADING):
        return nxt
    return nxt - step if action == EXTEND else nxt + step


def from_replay_row(row: Mapping[str, Any], *, visit_date: date | None = None, bcva: float | None = None, best_bcva: float | None = None,
                    fluid_probs: Mapping[str, float] | None = None, cst_sd: float = 0.0,
                    ref_cst: float | None = None) -> VisitPacket:
    """Build a packet from one row of eval/results/engine_replay_*.csv.

    Replay columns: study_id, visit, engine, shown, flagged, cst_input, interval_in (= the engine's
    interval *after* the step, i.e. the recommended next interval), trace (' | '-joined). BCVA,
    fluid and CST uncertainty are not in the CSV and can be supplied. The prior interval is recovered
    exactly by inverting the interval arithmetic (replay intervals are even and clipped to [4, 16],
    and the trace records when a cap held the interval). The reference CST is parsed from the
    trace (rounded to 1 µm there, so the derived % change can differ from the trace's by ~1 point).
    """
    trace = tuple(t.strip() for t in str(row.get("trace", "")).split("|") if t.strip())
    action, nxt, cst = str(row["engine"]), int(row["interval_in"]), float(row["cst_input"])
    refs = [float(m.group(1)) for t in trace for m in _REF_RE.finditer(t)]
    ref = refs[0] if refs else ref_cst  # some trace lines omit the reference; caller may carry it
    flagged = str(row.get("flagged", "False")).lower() in ("true", "1")
    prior = _infer_prior_interval(action, nxt, trace)
    sid = str(row["study_id"])
    return VisitPacket(
        study_id=sid, eye=_eye_from_id(sid), visit=int(row["visit"]), visit_date=visit_date, cst_um=cst,
        ref_cst_um=ref, ratio=cst / ref if ref else None, bcva_letters=bcva, best_bcva_letters=best_bcva,
        fluid_probs=dict(fluid_probs or {}), cst_sd_um=cst_sd, current_interval_wk=prior, action=action,
        recommended=str(row.get("shown", FLAG if flagged else action)), next_interval_wk=nxt, flagged=flagged,
        flag_reasons=tuple(t for t in trace if _is_flag_reason(t)), trace=trace,
    )


def packets_from_replay_csv(path: str, study_id: str, **kw) -> list[VisitPacket]:
    """All packets for one eye from a replay CSV, in visit order (reference CST carried forward)."""
    import pandas as pd

    df = pd.read_csv(path, dtype={"study_id": str})
    out, ref = [], None
    for r in df[df.study_id == study_id].sort_values("visit").to_dict("records"):
        pk = from_replay_row(r, ref_cst=ref, **kw)
        out.append(pk)
        ref = pk.ref_cst_um
        for t in pk.trace:
            if m := _REBASE_RE.search(t):
                ref = float(m.group(1))
    return out
