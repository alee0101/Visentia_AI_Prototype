"""FHIR bundle for one visit, validated with `fhir.resources` (R4B models; R4-compatible here).

Draft vs approved is carried by resource statuses:
                      draft (AI only)            approved / overridden (physician signed)
  Observation         preliminary                final
  ServiceRequest      draft, intent=proposal     active, intent=order
  Appointment         proposed, participants     booked, participants accepted
                      needs-action
  Provenance          agent: author=Device       + verifier=Practitioner, activity UPDATE if overridden

Coding policy: no LOINC codes are guessed. CST, BCVA and the injection procedure use a clearly
labelled local CodeSystem (LOCAL_CS); only standard HL7 terminology (observation-category,
provenance-participant-type, v3-DataOperation), UCUM units and SNOMED CT eye body sites are used.
Resource ids are uuid5 of (study, visit, kind), so bundles are reproducible.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fhir.resources.R4B.bundle import Bundle

from .packet import PhysicianDecision, VisitPacket
from .scheduler import ScheduleProposal

LOCAL_CS = "http://visentia.ai/fhir/CodeSystem/te-copilot"  # local, NOT LOINC
UCUM = "http://unitsofmeasure.org"
SNOMED = "http://snomed.info/sct"
EYE_SITE = {"OD": ("18944008", "Right eye structure"), "OS": ("8966001", "Left eye structure")}
_NS = uuid.UUID("6f1c2a52-6d0e-4b7a-9a43-0c1f6a6b9e11")
DEVICE_NAME, DEVICE_VERSION = "Visentia T&E Copilot (research prototype)", "0.1.0"

DRAFT_STATUS = {"Observation": "preliminary", "ServiceRequest": "draft", "Appointment": "proposed"}
FINAL_STATUS = {"Observation": "final", "ServiceRequest": "active", "Appointment": "booked"}


def _id(packet: VisitPacket, kind: str) -> str:
    return str(uuid.uuid5(_NS, f"{packet.study_id}|{packet.visit}|{kind}"))


def _ref(rtype: str, rid: str) -> dict:
    return {"reference": f"urn:uuid:{rid}", "type": rtype}


def _local(code: str, display: str) -> dict:
    return {"coding": [{"system": LOCAL_CS, "code": code, "display": display}], "text": display}


def build_bundle(packet: VisitPacket, schedule: ScheduleProposal | None, *,
                 decision: PhysicianDecision | None = None, now: datetime | None = None) -> dict:
    """Build + validate a collection Bundle. Returns plain JSON (dict). `decision` None = AI draft."""
    signed = decision is not None
    st = FINAL_STATUS if signed else DRAFT_STATUS
    action = (decision.action if decision and decision.action else packet.action)
    interval = (decision.interval_wk if decision and decision.interval_wk else packet.next_interval_wk)
    recorded = (decision.decided_at if decision and decision.decided_at else now or datetime.now(timezone.utc))
    if recorded.tzinfo is None:
        recorded = recorded.replace(tzinfo=timezone.utc)
    eff = packet.visit_date.isoformat() if packet.visit_date else None

    pid, did = _id(packet, "patient"), _id(packet, "device")
    patient = {"resourceType": "Patient", "id": pid,
               "identifier": [{"system": f"{LOCAL_CS}/study-id", "value": packet.study_id}]}
    device = {"resourceType": "Device", "id": did, "version": [{"value": DEVICE_VERSION}],
              "deviceName": [{"name": DEVICE_NAME, "type": "user-friendly-name"}]}
    resources: list[dict] = [patient, device]
    body = {"bodySite": {"coding": [{"system": SNOMED, "code": EYE_SITE[packet.eye][0],
                                     "display": EYE_SITE[packet.eye][1]}]}} if packet.eye in EYE_SITE else {}
    exam = [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "exam"}]}]

    def obs(kind: str, code: dict, qty: dict, note: str | None = None, device_ref: bool = False) -> dict:
        o = {"resourceType": "Observation", "id": _id(packet, kind), "status": st["Observation"], "category": exam,
             "code": code, "subject": _ref("Patient", pid), "valueQuantity": qty, **body}
        if eff:
            o["effectiveDateTime"] = eff
        if device_ref:
            o["device"] = _ref("Device", did)
        if note:
            o["note"] = [{"text": note}]
        return o

    cst = obs("cst", _local("cst-oct", "Central subfield thickness (OCT)"),
              {"value": round(packet.cst_um, 1), "unit": "µm", "system": UCUM, "code": "um"},
              note=(f"Model estimate, SD {packet.cst_sd_um:.1f} µm" if packet.cst_sd_um else None),
              device_ref=packet.cst_sd_um > 0)
    resources.append(cst)
    reason_refs = [_ref("Observation", cst["id"])]
    if packet.bcva_letters is not None:
        bcva = obs("bcva", _local("bcva-etdrs", "Best-corrected visual acuity (ETDRS letters)"),
                   {"value": packet.bcva_letters, "unit": "letters", "system": UCUM, "code": "{letters}"})
        resources.append(bcva)
        reason_refs.append(_ref("Observation", bcva["id"]))

    sr = {"resourceType": "ServiceRequest", "id": _id(packet, "servicerequest"), "status": st["ServiceRequest"],
          "intent": "order" if signed else "proposal",
          "code": _local("intravitreal-anti-vegf", "Intravitreal anti-VEGF injection"),
          "subject": _ref("Patient", pid), "authoredOn": recorded.isoformat(), "reasonReference": reason_refs,
          "note": [{"text": f"T&E {action}: next interval {interval} wk. Engine: {packet.recommended} "
                            f"{packet.next_interval_wk} wk. Trace: {' | '.join(packet.trace)}"}]}
    if body:
        sr["bodySite"] = [body["bodySite"]]  # 0..* on ServiceRequest, 0..1 on Observation
    if decision and decision.is_override:
        sr["note"].append({"text": f"Physician override: {decision.reason}"})
    if schedule is not None:
        sr["occurrencePeriod"] = {"start": schedule.window_start.isoformat(), "end": schedule.window_end.isoformat()}
    prac = None
    if signed:
        prac = {"resourceType": "Practitioner", "id": str(uuid.uuid5(_NS, f"practitioner|{decision.physician_id}")),
                "identifier": [{"system": f"{LOCAL_CS}/practitioner-id", "value": decision.physician_id}]}
        if decision.physician_name:
            prac["name"] = [{"text": decision.physician_name}]
        sr["requester"] = _ref("Practitioner", prac["id"])
        resources.append(prac)
    resources.append(sr)

    if schedule is not None and schedule.slot is not None:
        start = schedule.slot.replace(tzinfo=schedule.slot.tzinfo or timezone.utc)
        parts = [{"actor": _ref("Patient", pid), "status": "accepted" if signed else "needs-action"}]
        if prac:
            parts.append({"actor": _ref("Practitioner", prac["id"]), "status": "accepted"})
        resources.append({"resourceType": "Appointment", "id": _id(packet, "appointment"),
                          "status": st["Appointment"], "start": start.isoformat(),
                          "end": (schedule.slot_end.replace(tzinfo=start.tzinfo)).isoformat(),
                          "description": "Intravitreal anti-VEGF injection visit (treat-and-extend)",
                          "basedOn": [_ref("ServiceRequest", sr["id"])], "participant": parts})

    pt = "http://terminology.hl7.org/CodeSystem/provenance-participant-type"
    agents = [{"type": {"coding": [{"system": pt, "code": "author"}]}, "who": _ref("Device", did)}]
    if prac:
        agents.append({"type": {"coding": [{"system": pt, "code": "verifier"}]}, "who": _ref("Practitioner", prac["id"])})
    targets = [_ref(r["resourceType"], r["id"]) for r in resources if r["resourceType"] in ("Observation", "ServiceRequest", "Appointment")]
    prov = {"resourceType": "Provenance", "id": _id(packet, "provenance"), "target": targets,
            "recorded": recorded.isoformat(), "agent": agents}
    if decision and decision.is_override:
        prov["activity"] = {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-DataOperation", "code": "UPDATE"}]}
    resources.append(prov)

    bundle = {"resourceType": "Bundle", "id": _id(packet, "bundle"), "type": "collection",
              "timestamp": recorded.isoformat(),
              "entry": [{"fullUrl": f"urn:uuid:{r['id']}", "resource": r} for r in resources]}
    validated = Bundle.model_validate(bundle)  # raises pydantic.ValidationError on schema problems
    return validated.model_dump(mode="json", exclude_none=True)


def statuses(bundle: dict) -> dict[str, str]:
    """{resourceType: status} for the status-bearing resources (handy for UI + tests)."""
    return {e["resource"]["resourceType"]: e["resource"]["status"] for e in bundle["entry"] if "status" in e["resource"]}
