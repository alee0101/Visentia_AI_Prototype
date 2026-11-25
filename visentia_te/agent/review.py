"""Physician approve / override. The physician is the decision-maker; the AI output is a draft.

Every review appends one JSON line to feedback.jsonl (path configurable; gitignored), keeping
the engine's recommendation next to the physician's choice -- the raw material for measuring
agreement and for revising protocol thresholds later. An override regenerates the schedule,
the patient message and the FHIR bundle with the physician-chosen values.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from visentia_te.engine.protocol import EXTEND, MAINTAIN, SHORTEN

from .fhir import build_bundle
from .note import fact_pool_for
from .packet import PhysicianDecision, VisitPacket
from .patient_message import check_message, flesch_kincaid_grade, template_message
from .scheduler import ClinicCalendar, propose_next_visit

OVERRIDE_ACTIONS = (EXTEND, MAINTAIN, SHORTEN)
PROTOCOL_INTERVAL_RANGE = (4, 16)


def feedback_path(path: str | os.PathLike | None = None) -> Path:
    return Path(path or os.environ.get("VISENTIA_FEEDBACK_PATH", "feedback.jsonl"))


def _log(record: dict, path: str | os.PathLike | None) -> Path:
    p = feedback_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return p


def approve(physician_id: str, *, physician_name: str | None = None) -> PhysicianDecision:
    return PhysicianDecision(physician_id, approved=True, physician_name=physician_name,
                             decided_at=datetime.now(timezone.utc))


def override(physician_id: str, *, action: str, interval_wk: int, reason: str,
             physician_name: str | None = None) -> PhysicianDecision:
    """Physician override. Requires a reason; interval must be a positive whole number of weeks."""
    if action not in OVERRIDE_ACTIONS:
        raise ValueError(f"action must be one of {OVERRIDE_ACTIONS}")
    if int(interval_wk) != interval_wk or interval_wk <= 0:
        raise ValueError("interval_wk must be a positive integer")
    if not reason.strip():
        raise ValueError("an override requires a free-text reason")
    return PhysicianDecision(physician_id, approved=False, action=action, interval_wk=int(interval_wk),
                             reason=reason.strip(), physician_name=physician_name,
                             decided_at=datetime.now(timezone.utc))


def apply_review(packet: VisitPacket, decision: PhysicianDecision, *, visit_date: date,
                 calendar: ClinicCalendar | None = None, window_days: int = 3,
                 feedback: str | os.PathLike | None = None) -> dict[str, Any]:
    """Finalize a visit: (re)schedule with the signed values, build signed FHIR, log feedback."""
    if decision.decided_at is None:
        decision = replace(decision, decided_at=datetime.now(timezone.utc))
    action = decision.action or packet.action
    interval = decision.interval_wk or packet.next_interval_wk
    schedule = propose_next_visit(visit_date, interval, calendar=calendar, window_days=window_days)
    schedule.status = "approved"
    msg = template_message(packet, schedule, approved=True, action=action, interval_wk=interval)
    msg_report = check_message(msg, fact_pool_for(packet, schedule), approved=True)
    bundle = build_bundle(packet, schedule, decision=decision)
    lo, hi = PROTOCOL_INTERVAL_RANGE
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(), "study_id": packet.study_id, "visit": packet.visit,
        "visit_date": visit_date.isoformat(), "type": "override" if decision.is_override else "approve",
        "physician_id": decision.physician_id, "decided_at": decision.decided_at.isoformat(),
        "engine": {"action": packet.action, "recommended": packet.recommended,
                   "next_interval_wk": packet.next_interval_wk, "flagged": packet.flagged},
        "final": {"action": action, "interval_wk": interval}, "reason": decision.reason or None,
        "outside_protocol_range": not (lo <= interval <= hi),
        "schedule_slot": schedule.slot.isoformat() if schedule.slot else None, "trace": list(packet.trace),
    }
    path = _log(record, feedback)
    return {"status": "overridden" if decision.is_override else "approved", "decision": record, "schedule": schedule.to_dict(), "fhir": bundle,
            "patient_message": {"text": msg, "source": "template", "grounding": msg_report.to_dict(),
                                "fk_grade": flesch_kincaid_grade(msg)},
            "feedback_path": str(path)}
