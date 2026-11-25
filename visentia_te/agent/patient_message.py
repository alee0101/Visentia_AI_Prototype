"""Plain-language patient message: when the next injection visit is, and why, in simple terms.

Same grounding check as the note, plus two message-specific gates:
  * finality: before approval the message must not read as a final/booked decision;
  * readability: Flesch-Kincaid grade must be <= MAX_GRADE (target 6th-8th grade).
"""
from __future__ import annotations

import json
import re
from typing import Any

from visentia_te.engine.protocol import EXTEND, FLAG, LOADING, MAINTAIN, SHORTEN

from .grounding import FactPool, GroundingReport, check_grounding
from .llm import DEFAULT_MODEL, DraftResult, generate_checked
from .note import fact_pool_for, schedule_facts
from .packet import VisitPacket
from .scheduler import ScheduleProposal

TARGET_GRADE, MAX_GRADE = (6.0, 8.0), 9.0

WHY = {
    EXTEND: "The swelling in the back of your eye looks under control, so we may be able to wait a bit longer between shots.",
    MAINTAIN: "Your eye looks about the same as last time, so we plan to keep the same time between shots.",
    SHORTEN: "There is a bit more swelling in the back of your eye, so we may need to give the next shot sooner.",
    LOADING: "You are in the first set of monthly shots that start your treatment.",
    FLAG: "Your doctor wants to look more closely at your scan before picking the time between shots.",
}

PATIENT_SYSTEM = f"""You write a short message to a patient who gets eye injections for diabetic \
macular edema (swelling in the back of the eye).

Rules:
- 3 to 6 short sentences. Plain words at a 6th to 8th grade reading level. No medical jargon, \
no acronyms (no CST, OCT, BCVA, VEGF, FLAG), no test numbers.
- Say when the next injection visit is likely to be and why, in simple terms.
- Use ONLY the facts in FACTS. Every number and date must appear in FACTS; write numbers as digits.
- If physician_approved is false, this is only a suggestion: say the doctor will review and \
confirm the plan. Never say the visit is booked, scheduled, confirmed, or decided.
- Do not give medical advice beyond the plan. Plain text, no greeting name, no sign-off name."""

_FINAL_RE = re.compile(r"\b(is|has been|are|was|been)\s+(booked|scheduled|confirmed|finalized|decided)\b"
                       r"|\byour doctor (has )?(decided|approved)\b", re.IGNORECASE)


# ---- readability --------------------------------------------------------------------------
def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def flesch_kincaid_grade(text: str) -> float:
    """FK grade = 0.39 * words/sentence + 11.8 * syllables/word - 15.59 (heuristic syllables)."""
    sentences = max(1, len(re.findall(r"[.!?]+(?:\s|$)", text)))
    words = re.findall(r"[A-Za-z][A-Za-z'-]*|\d+", text)
    if not words:
        return 0.0
    syl = sum(_syllables(w) if w[0].isalpha() else 1 for w in words)
    return round(0.39 * len(words) / sentences + 11.8 * syl / len(words) - 15.59, 1)


# ---- template -----------------------------------------------------------------------------
def template_message(packet: VisitPacket, schedule: ScheduleProposal | None, *, approved: bool = False,
                     action: str | None = None, interval_wk: int | None = None) -> str:
    """Deterministic message. `action`/`interval_wk` override the engine values (physician override)."""
    act = action or packet.recommended
    wk = interval_wk or packet.next_interval_wk
    parts = ["We checked your eye with a scan today.", WHY.get(act, WHY[FLAG])]
    when = f"in about {wk} weeks"
    if schedule is not None and schedule.slot is not None:
        when += f", on {schedule.slot:%A, %B} {schedule.slot.day}"
    if approved:
        parts.append(f"Your doctor has set your next shot visit {when}.")
    else:
        parts.append(f"The plan is for your next shot visit {when}.")
        parts.append("Your doctor will look at this plan and let you know if it changes.")
    parts.append("Please call us if your vision gets worse before then.")
    return " ".join(parts)


def message_user_prompt(packet: VisitPacket, schedule: ScheduleProposal | None) -> str:
    facts = {"physician_approved": False, "plan_in_simple_terms": WHY.get(packet.recommended, WHY[FLAG]),
             "next_interval_weeks": packet.next_interval_wk, **schedule_facts(schedule)}
    if schedule is not None and schedule.slot is not None:
        facts["proposed_next_visit"]["proposed_day"] = f"{schedule.slot:%A, %B} {schedule.slot.day}"
    return f"FACTS:\n{json.dumps(facts, indent=1)}\n\nWrite the patient message now."


def check_message(text: str, pool: FactPool, *, approved: bool = False) -> GroundingReport:
    rep = check_grounding(text, pool)
    if not approved and (m := _FINAL_RE.search(text)):
        rep.issues.append(f"'{m.group(0)}' presents the plan as final; it is pending physician approval.")
    grade = flesch_kincaid_grade(text)
    if grade > MAX_GRADE:
        rep.issues.append(f"Reading level is grade {grade}; simplify to grade {TARGET_GRADE[1]:.0f} or below "
                          "(shorter sentences, shorter words).")
    rep.ok = not rep.ungrounded and not rep.issues
    return rep


def draft_patient_message(packet: VisitPacket, *, client: Any | None = None,
                          schedule: ScheduleProposal | None = None, model: str = DEFAULT_MODEL) -> DraftResult:
    """Draft the (pre-approval) patient message. client=None -> template (zero network)."""
    pool = fact_pool_for(packet, schedule)
    res = generate_checked(client, system=PATIENT_SYSTEM, user=message_user_prompt(packet, schedule),
                           check=lambda t: check_message(t, pool), template=lambda: template_message(packet, schedule),
                           model=model)
    res.extra["fk_grade"] = flesch_kincaid_grade(res.text)
    return res
