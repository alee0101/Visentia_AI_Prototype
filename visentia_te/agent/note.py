"""Physician-facing Assessment & Plan note, drafted from a VisitPacket.

LLM path: Claude writes the note from FACTS (packet + proposed schedule) only; the numeric
grounding check rejects any number/date not traceable to FACTS, retries once with feedback, then
falls back to the deterministic template. Template path: used when no client is available.
"""
from __future__ import annotations

import json
from typing import Any

from visentia_te.engine.protocol import EXTEND, FLAG, LOADING, MAINTAIN, SHORTEN

from .grounding import FactPool, GroundingReport, build_fact_pool, check_grounding
from .llm import DEFAULT_MODEL, DraftResult, generate_checked
from .packet import VisitPacket
from .scheduler import ScheduleProposal

ACTION_PHRASE = {EXTEND: "extend", MAINTAIN: "maintain", SHORTEN: "shorten", LOADING: "loading phase, continue"}

NOTE_SYSTEM = """You draft the Assessment & Plan section of a retina clinic note for an eye with \
diabetic macular edema on treat-and-extend intravitreal anti-VEGF therapy.

Rules:
- Use ONLY the facts in the FACTS JSON. Do not invent or infer findings, exam results, history, \
medications, drug names, laterality, or measurements that are not in FACTS.
- Every number and date you write must appear in FACTS (you may round to whole numbers or the \
precision given). Write numbers as digits. Do not convert weeks to months or compute new values.
- The interval recommendation comes from a deterministic protocol engine. Present it as a DRAFT \
recommendation pending physician approval, never as a final decision or completed order.
- If flagged_for_physician_review is true, state the flag reasons prominently.
- Ophthalmology register, concise: a header line, "Assessment:" and "Plan:" with short bullet \
lines, at most ~12 lines total. Plain text, no markdown headings, no preamble."""


def schedule_facts(schedule: ScheduleProposal | None) -> dict[str, Any]:
    if schedule is None:
        return {}
    return {"proposed_next_visit": {
        "target_date": schedule.target_date.isoformat(), "window_start": schedule.window_start.isoformat(),
        "window_end": schedule.window_end.isoformat(), "window_days": schedule.window_days,
        "proposed_slot": schedule.slot.strftime("%Y-%m-%d %H:%M") if schedule.slot else None,
        "status": schedule.status}}


def fact_pool_for(packet: VisitPacket, schedule: ScheduleProposal | None) -> FactPool:
    if schedule is None:
        return build_fact_pool(packet)
    return build_fact_pool(packet, dates=schedule.allowed_dates(), times=[schedule.slot] if schedule.slot else [],
                           numbers=[schedule.window_days, schedule.interval_wk])


def template_note(packet: VisitPacket, schedule: ScheduleProposal | None = None) -> str:
    """Deterministic note; every number in it comes straight from the packet/schedule."""
    p = packet
    eye = f" {p.eye}" if p.eye else ""
    head = f"{p.study_id}{eye} | Visit {p.visit}" + (f" | {p.visit_date.isoformat()}" if p.visit_date else "")
    lines = ["ASSESSMENT & PLAN (DRAFT - AI-generated, pending physician approval)", head, "Assessment:",
             f"- Diabetic macular edema{eye}, on treat-and-extend anti-VEGF ({p.protocol} protocol)."]
    cst = f"- OCT CST {p.cst_um:.0f} µm" + (f" (model estimate, SD {p.cst_sd_um:.0f} µm)" if p.cst_sd_um else "")
    if p.ref_cst_um is not None and p.cst_change_pct is not None:
        cst += f" vs reference {p.ref_cst_um:.0f} µm ({p.cst_change_pct:+.0f}%)"
    lines.append(cst + ".")
    if p.bcva_letters is not None:
        b = f"- BCVA {p.bcva_letters:.0f} ETDRS letters"
        if p.best_bcva_letters is not None:
            b += f" (best {p.best_bcva_letters:.0f}; {p.bcva_loss_letters:.0f} below best)"
        lines.append(b + ".")
    if p.fluid_probs:
        lines.append("- Model fluid probability: " + ", ".join(f"{k} {v:.2f}" for k, v in p.fluid_probs.items()) + ".")
    lines.append("Plan:")
    iv = f"interval {p.current_interval_wk} -> {p.next_interval_wk} wk" if p.current_interval_wk else f"next interval {p.next_interval_wk} wk"
    if p.recommended == FLAG:
        lines.append(f"- Copilot: FLAG for physician review (protocol action {p.action}; {iv}).")
        lines += [f"- Review reason: {r}." for r in p.flag_reasons]
    else:
        lines.append(f"- Copilot draft recommendation: {p.action} ({ACTION_PHRASE.get(p.action, p.action)}); {iv}.")
    if schedule is not None:
        when = f"{schedule.slot:%Y-%m-%d %H:%M}" if schedule.slot else "no open slot (manual booking)"
        lines.append(f"- Proposed next injection visit: {when} (window {schedule.window_start.isoformat()} "
                     f"to {schedule.window_end.isoformat()}).")
    lines.append("- Rule trace: " + " | ".join(p.trace) + ".")
    lines.append("- Pending physician approval; physician may approve or override.")
    return "\n".join(lines)


def note_user_prompt(packet: VisitPacket, schedule: ScheduleProposal | None) -> str:
    facts = {**packet.facts(), **schedule_facts(schedule)}
    return f"FACTS:\n{json.dumps(facts, indent=1, ensure_ascii=False)}\n\nDraft the Assessment & Plan now."


def check_note(text: str, pool: FactPool) -> GroundingReport:
    rep = check_grounding(text, pool)
    if "draft" not in text.lower() and "pending" not in text.lower():
        rep.issues.append("The note must state that the recommendation is a draft pending physician approval.")
        rep.ok = False
    return rep


def draft_note(packet: VisitPacket, *, client: Any | None = None, schedule: ScheduleProposal | None = None,
               model: str = DEFAULT_MODEL) -> DraftResult:
    """Draft the A&P note. client=None -> deterministic template (zero network)."""
    pool = fact_pool_for(packet, schedule)
    return generate_checked(client, system=NOTE_SYSTEM, user=note_user_prompt(packet, schedule),
                            check=lambda t: check_note(t, pool), template=lambda: template_note(packet, schedule),
                            model=model)
