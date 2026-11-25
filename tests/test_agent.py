"""Agent layer tests -- fully offline. The Claude client is a fake injected object; no network."""
import json
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from fhir.resources.R4B.bundle import Bundle

from visentia_te.agent import review
from visentia_te.agent.copilot import run_visit
from visentia_te.agent.fhir import build_bundle, statuses
from visentia_te.agent.grounding import build_fact_pool, check_grounding
from visentia_te.agent.note import draft_note, template_note
from visentia_te.agent.packet import VisitPacket, from_replay_row, step_and_pack
from visentia_te.agent.patient_message import draft_patient_message, flesch_kincaid_grade, template_message
from visentia_te.agent.scheduler import ClinicCalendar, propose_next_visit, us_federal_holidays
from visentia_te.engine.protocol import EyeState, Observation, load_protocol


class FakeClient:
    """Mimics `anthropic.Anthropic().beta.messages.create`; returns canned texts in order."""

    def __init__(self, *texts, stop_reason="end_turn"):
        self.texts, self.calls, self.stop_reason = list(texts), [], stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(stop_reason=self.stop_reason,
                               content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text=self.texts.pop(0))])


def make_packet(**kw) -> VisitPacket:
    base = dict(study_id="0201GOD", visit=9, visit_date=date(2026, 3, 3), cst_um=295.6, ref_cst_um=293.0,
                ratio=295.6 / 293.0, bcva_letters=72, best_bcva_letters=75, fluid_probs={"any_fluid": 0.42},
                cst_sd_um=11.8, current_interval_wk=8, action="EXTEND", recommended="EXTEND", next_interval_wk=10,
                trace=("CST 296 vs ref 293 (+1%) <= +10% -> extend",), eye="OD")
    base.update(kw)
    return VisitPacket(**base)


OPEN_CAL = ClinicCalendar(fill_rate=0.0)  # every slot free -> deterministic earliest slot


@pytest.fixture(autouse=True)
def _no_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("VISENTIA_FEEDBACK_PATH", str(tmp_path / "feedback.jsonl"))


# ---- grounding ---------------------------------------------------------------------------------
def test_grounding_accepts_rounded_and_formatted_numbers():
    p = make_packet()
    pool = build_fact_pool(p, dates=[date(2026, 5, 12)])
    text = ("1. CST 296 µm (295.6) vs reference 293 µm, +1% change; SD 12 µm.\n"
            "2. BCVA 72 letters, 3 below best of 75. P(fluid) 0.42 (42%), about 0.4.\n"
            "Plan: extend q8w -> 10 wk (+2 wk). Seen 2026-03-03 (3/3/2026); next May 12, 2026 or 12 May.")
    rep = check_grounding(text, pool)
    assert rep.ok, rep.ungrounded
    assert "296" in rep.checked and "May 12, 2026" in rep.checked


def test_grounding_catches_invented_numbers_and_dates():
    pool = build_fact_pool(make_packet())
    rep = check_grounding("CST improved by 37 µm to about 300 µm; fluid 42 µm; return June 9; HbA1c 7.2%.", pool)
    spans = {u["span"] for u in rep.ungrounded}
    assert not rep.ok
    assert {"37", "300", "42", "June 9", "7.2%"} <= spans  # 42 is only grounded as a percentage


# ---- note ---------------------------------------------------------------------------------------
GOOD_NOTE = ("DRAFT A&P, pending physician approval. 0201GOD OD visit 9.\nAssessment: CST 296 µm vs ref 293 µm "
             "(+1%); BCVA 72 letters.\nPlan: extend interval 8 -> 10 wk per protocol (draft).")
BAD_NOTE = "A&P (draft). CST 296 µm, down 40 µm since last visit. Extend to 10 wk; HbA1c 8.1."


def test_template_note_is_grounded_without_client():
    p = make_packet()
    res = draft_note(p, client=None, schedule=propose_next_visit(p.visit_date, 10, calendar=OPEN_CAL))
    assert res.source == "template" and res.grounding.ok and not res.llm_failed
    assert "pending physician approval" in res.text and "296" in res.text


def test_note_retry_then_accept():
    fake = FakeClient(BAD_NOTE, GOOD_NOTE)
    res = draft_note(make_packet(), client=fake)
    assert res.source == "llm_retry" and res.text == GOOD_NOTE and res.llm_failed
    assert {u["span"] for u in res.attempts[0]["grounding"]["ungrounded"]} == {"40", "8.1"}
    assert len(fake.calls) == 2 and "'40'" in fake.calls[1]["messages"][0]["content"]
    sys_prompt = fake.calls[0]["system"]
    assert "Do not invent" in sys_prompt and "pending physician approval" in sys_prompt
    assert fake.calls[0]["model"] == "claude-sonnet-5-5"


def test_note_falls_back_to_template_after_two_failures():
    p = make_packet()
    res = draft_note(p, client=FakeClient(BAD_NOTE, BAD_NOTE))
    assert res.source == "template_fallback" and res.llm_failed and res.grounding.ok
    assert res.text == template_note(p) and len(res.attempts) == 2


def test_refusal_goes_straight_to_template():
    res = draft_note(make_packet(), client=FakeClient(GOOD_NOTE, stop_reason="refusal"))
    assert res.source == "template_fallback" and "refusal" in res.attempts[0]["error"]


# ---- patient message ---------------------------------------------------------------------------
def test_patient_message_template_readable_and_grounded():
    p = make_packet()
    sched = propose_next_visit(p.visit_date, p.next_interval_wk, calendar=OPEN_CAL)
    res = draft_patient_message(p, schedule=sched)
    assert res.source == "template" and res.grounding.ok
    assert res.extra["fk_grade"] <= 8 and "10 weeks" in res.text and "doctor will look" in res.text
    assert flesch_kincaid_grade("The cat sat. The dog ran.") < 3


def test_patient_message_rejects_final_wording_and_invented_date():
    p = make_packet()
    sched = propose_next_visit(p.visit_date, p.next_interval_wk, calendar=OPEN_CAL)
    bad = "Your next shot has been scheduled for June 2. See you then."
    res = draft_patient_message(p, client=FakeClient(bad, bad), schedule=sched)
    assert res.source == "template_fallback"
    g = res.attempts[0]["grounding"]
    assert g["ungrounded"][0]["span"] == "June 2" and any("final" in i for i in g["issues"])


# ---- scheduler ---------------------------------------------------------------------------------
def test_scheduler_skips_weekends_and_holidays():
    target = date(2026, 12, 5)  # Saturday
    last = date(2026, 10, 24)
    cal = ClinicCalendar(holidays=frozenset({date(2026, 12, 3)}), fill_rate=0.0)
    s = propose_next_visit(last, 6, calendar=cal, window_days=2)
    assert s.target_date == target and s.slot == datetime(2026, 12, 4, 8, 0)
    assert s.skipped == [(date(2026, 12, 3), "holiday")]
    cal2 = ClinicCalendar(holidays=frozenset({date(2026, 12, 3), date(2026, 12, 4)}), fill_rate=0.0)
    s2 = propose_next_visit(last, 6, calendar=cal2, window_days=2)
    assert s2.slot.date() == date(2026, 12, 7) and (date(2026, 12, 6), "weekend") in s2.skipped
    assert s2.slot.weekday() < 5 and s.status == "proposed"


def test_scheduler_deterministic_and_us_holidays():
    a = propose_next_visit(date(2026, 1, 5), 8, calendar=ClinicCalendar(seed=7))
    b = propose_next_visit(date(2026, 1, 5), 8, calendar=ClinicCalendar(seed=7))
    assert a.slot == b.slot and a.window_start <= a.slot.date() <= a.window_end
    hol = us_federal_holidays(2026)
    assert {date(2026, 11, 26), date(2026, 7, 3), date(2026, 1, 19), date(2026, 5, 25)} <= hol
    none = propose_next_visit(date(2026, 1, 5), 8, calendar=ClinicCalendar(fill_rate=1.0))
    assert none.slot is None and "manual" in none.reason


# ---- FHIR --------------------------------------------------------------------------------------
def test_fhir_bundle_validates_and_statuses_flip_on_approval():
    p = make_packet()
    sched = propose_next_visit(p.visit_date, p.next_interval_wk, calendar=OPEN_CAL)
    draft = build_bundle(p, sched)
    Bundle.model_validate(draft)
    assert statuses(draft) == {"Observation": "preliminary", "ServiceRequest": "draft", "Appointment": "proposed"}
    assert "loinc" not in json.dumps(draft).lower()
    signed = build_bundle(p, sched, decision=review.approve("dr_lee", physician_name="Dr Lee"))
    Bundle.model_validate(signed)
    assert statuses(signed) == {"Observation": "final", "ServiceRequest": "active", "Appointment": "booked"}
    prov = next(e["resource"] for e in signed["entry"] if e["resource"]["resourceType"] == "Provenance")
    roles = {a["type"]["coding"][0]["code"]: a["who"]["type"] for a in prov["agent"]}
    assert roles == {"author": "Device", "verifier": "Practitioner"}
    draft_prov = next(e["resource"] for e in draft["entry"] if e["resource"]["resourceType"] == "Provenance")
    assert [a["type"]["coding"][0]["code"] for a in draft_prov["agent"]] == ["author"]


# ---- review / orchestration -------------------------------------------------------------------
def test_override_is_logged_and_regenerates_schedule_and_fhir(tmp_path):
    fb = tmp_path / "fb.jsonl"
    p = make_packet()
    dec = review.override("dr_lee", action="MAINTAIN", interval_wk=8, reason="new PDR; keep q8w")
    out = run_visit(p, approve=dec, calendar=OPEN_CAL, feedback_path=fb)
    assert out["status"] == "overridden"
    assert out["schedule"]["interval_wk"] == 8 and out["draft_schedule"]["interval_wk"] == 10
    rec = json.loads(fb.read_text().strip().splitlines()[-1])
    assert rec["type"] == "override" and rec["reason"] == "new PDR; keep q8w"
    assert rec["engine"]["next_interval_wk"] == 10 and rec["final"] == {"action": "MAINTAIN", "interval_wk": 8}
    sr = next(e["resource"] for e in out["fhir"]["entry"] if e["resource"]["resourceType"] == "ServiceRequest")
    assert sr["status"] == "active" and "MAINTAIN: next interval 8 wk" in sr["note"][0]["text"]
    assert any("override" in n["text"] for n in sr["note"])
    assert out["patient_message"]["grounding"]["ok"] and "8 weeks" in out["patient_message"]["text"]
    with pytest.raises(ValueError):
        review.override("dr_lee", action="MAINTAIN", interval_wk=8, reason="  ")


def test_run_visit_template_path_offline_and_approve_logged(tmp_path):
    out = run_visit(make_packet(), calendar=OPEN_CAL)  # no key in env -> template, zero network
    assert out["status"] == "draft" and out["note"]["source"] == "template"
    assert out["grounding"]["note"]["ok"] and out["grounding"]["patient_message"]["ok"]
    assert statuses(out["fhir"])["ServiceRequest"] == "draft"
    fb = tmp_path / "a.jsonl"
    out2 = run_visit(make_packet(), approve=review.approve("dr_lee"), calendar=OPEN_CAL, feedback_path=fb)
    assert out2["status"] == "approved" and json.loads(fb.read_text())["type"] == "approve"


def test_run_visit_with_fake_client():
    fake = FakeClient(GOOD_NOTE, "We checked your eye today. Things look stable. The plan is a visit in about "
                                 "10 weeks. Your doctor will review this plan.")
    out = run_visit(make_packet(), client=fake, calendar=OPEN_CAL)
    assert out["note"]["source"] == "llm" and out["patient_message"]["source"] == "llm"
    assert len(fake.calls) == 2


# ---- packet builders --------------------------------------------------------------------------
def test_packet_builders():
    st = EyeState(load_protocol())
    for v in (1, 2, 3):
        st.step(Observation(v, 450))
    st.step(Observation(4, 300, bcva=70))
    pk = step_and_pack(st, Observation(5, 360, bcva=70, cst_sd=20, p_fluid=0.5), study_id="0201TOS")
    assert pk.eye == "OS" and pk.current_interval_wk == 6 and pk.ref_cst_um == pytest.approx(300)
    assert pk.flagged and pk.recommended == "FLAG" and pk.flag_reasons and pk.fluid_probs == {"any_fluid": 0.5}
    assert check_grounding(template_note(pk), build_fact_pool(pk)).ok
    row = {"study_id": "0201GOD", "visit": 9, "engine": "EXTEND", "shown": "EXTEND", "flagged": False,
           "cst_input": 277.9738, "interval_in": 10, "trace": "CST 278 vs ref 293 (-5%) <= +10% -> extend"}
    rp = from_replay_row(row)
    assert rp.ref_cst_um == 293 and rp.current_interval_wk == 8 and rp.next_interval_wk == 10
    assert template_message(rp, None).count("10 weeks") == 1
