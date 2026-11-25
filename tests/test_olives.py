import pytest

from visentia_te.data.olives import LABELS_DIR, load_trex_visits, make_patient_splits, patient_of

pytestmark = pytest.mark.skipif(not LABELS_DIR.exists(), reason="OLIVES labels not downloaded")


@pytest.fixture(scope="module")
def visits():
    v, rep = load_trex_visits()
    return v, rep


def test_arms_and_eyes(visits):
    v, _ = visits
    assert v.groupby("arm").study_id.nunique().to_dict() == {"GILA": 60, "Monthly": 30, "TREX": 60}


def test_known_eye_values(visits):
    # 0105TOS screening row in the TREX sheet: ETDRS 55, OCT 556
    v, _ = visits
    r = v[(v.study_id == "0105TOS") & (v.visit == 1)].iloc[0]
    assert (r.etdrs, r.cst, r.week) == (55, 556, 0.0)


def test_alignment_quality(visits):
    _, rep = visits
    assert (rep.vision_row >= 0).all()
    assert rep.match_ratio.median() > 0.95


def test_actions_only_after_loading(visits):
    v, _ = visits
    assert v[v.visit <= 3].action.isna().all()
    assert v[v.arm == "Monthly"].action.isna().all()
    assert set(v.action.dropna()) == {"EXTEND", "MAINTAIN", "SHORTEN"}


def test_intervals_plausible(visits):
    v, _ = visits
    iv = v.interval_wk.dropna()
    assert iv.between(2.5, 20).all()


def test_splits_patient_disjoint(visits):
    v, _ = visits
    sp = make_patient_splits(sorted(v.study_id.unique()))
    by_patient = {}
    for sid, s in sp.items():
        by_patient.setdefault(patient_of(sid), set()).add(s)
    assert all(len(s) == 1 for s in by_patient.values())
    assert set(sp.values()) == {"train", "val", "test"}
