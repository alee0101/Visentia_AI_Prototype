"""One UI-agnostic entry point: engine decision (VisitPacket) -> draft artifacts -> optional sign-off.

    out = run_visit(packet, offline=True)                     # template path, zero network
    out = run_visit(packet, client=anthropic.Anthropic())     # LLM drafting with grounding checks
    out = run_visit(packet, approve=review.approve("dr_lee")) # physician sign-off (logged)
"""
from __future__ import annotations

import os
from datetime import date
from typing import Any

from .fhir import build_bundle
from .llm import DEFAULT_MODEL, default_client
from .note import draft_note
from .packet import PhysicianDecision, VisitPacket
from .patient_message import draft_patient_message
from .review import apply_review
from .scheduler import ClinicCalendar, propose_next_visit


def run_visit(packet: VisitPacket, *, client: Any | None = None, approve: PhysicianDecision | None = None,
              offline: bool = False, calendar: ClinicCalendar | None = None, visit_date: date | None = None, window_days: int = 3,
              model: str = DEFAULT_MODEL, feedback_path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Draft note + patient message + schedule + FHIR for one visit; apply physician review if given.

    client: any object with `.beta.messages.create` (tests inject a fake); None -> a real client iff
            ANTHROPIC_API_KEY is set, else the deterministic template path.
    offline: True forces the template path (zero network) even if a key/client is available.
    approve: a PhysicianDecision (review.approve / review.override). When given, the schedule,
             patient message and FHIR bundle are regenerated with the signed values and the review
             is appended to feedback.jsonl. The draft note is kept as drafted (the audit record of
             what the AI proposed); the signed decision is in `review`.
    """
    client = None if offline else (client if client is not None else default_client())
    vdate = packet.visit_date or visit_date or date.today()
    schedule = propose_next_visit(vdate, packet.next_interval_wk, calendar=calendar, window_days=window_days)
    note = draft_note(packet, client=client, schedule=schedule, model=model)
    msg = draft_patient_message(packet, client=client, schedule=schedule, model=model)
    out: dict[str, Any] = {
        "status": "draft",
        "packet": packet.to_dict(),
        "note": note.to_dict(),
        "patient_message": msg.to_dict(),
        "schedule": schedule.to_dict(),
        "fhir": build_bundle(packet, schedule),
        "grounding": {"note": note.grounding.to_dict(), "patient_message": msg.grounding.to_dict()},
        "review": None,
    }
    if approve is not None:
        rev = apply_review(packet, approve, visit_date=vdate, calendar=calendar, window_days=window_days,
                           feedback=feedback_path)
        out.update(status=rev["status"], schedule=rev["schedule"], fhir=rev["fhir"], review=rev["decision"],
                   draft_schedule=schedule.to_dict(), draft_patient_message=out["patient_message"],
                   patient_message=rev["patient_message"], feedback_path=rev["feedback_path"])
        out["grounding"]["patient_message"] = rev["patient_message"]["grounding"]
    return out
