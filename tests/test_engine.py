from visentia_te.engine.protocol import EXTEND, FLAG, LOADING, MAINTAIN, SHORTEN, EyeState, Observation, load_protocol

P = load_protocol()


def eye_after_loading(ref=300.0) -> EyeState:
    st = EyeState(P)
    for v in (1, 2, 3):
        assert st.step(Observation(v, 450)).action == LOADING
    d = st.step(Observation(4, ref, bcva=70))  # first decision: below 325 gate -> T&E starts, extends
    assert d.action == EXTEND and st.ref_cst == ref and d.next_interval_wk == 6
    return st


def test_gate_keeps_q4w_above_325():
    st = EyeState(P)
    for v in (1, 2, 3):
        st.step(Observation(v, 500))
    d = st.step(Observation(4, 400))
    assert d.action == MAINTAIN and d.next_interval_wk == 4 and not st.te_started


def test_rule_bands():
    st = eye_after_loading(300)
    assert st.step(Observation(5, 325, bcva=70)).action == EXTEND  # +8%
    assert st.step(Observation(6, 350, bcva=70)).action == MAINTAIN  # +17%
    d = st.step(Observation(7, 400, bcva=70))  # +33%
    assert d.action == SHORTEN and d.next_interval_wk == 6


def test_interval_caps():
    st = eye_after_loading(300)
    for v in range(5, 15):
        d = st.step(Observation(v, 300, bcva=70))
    assert d.next_interval_wk == P["interval"]["max_wk"]


def test_bcva_loss_blocks_extension():
    st = eye_after_loading(300)
    d = st.step(Observation(5, 300, bcva=62))  # 8 letters below best
    assert d.action == MAINTAIN and any("BCVA" in t for t in d.trace)


def test_uncertain_cst_flags_and_caps_extension():
    st = eye_after_loading(300)
    d = st.step(Observation(5, 325, bcva=70, cst_sd=20))  # +8% but ±30 µm crosses the 10% line
    assert d.flagged and d.recommended == FLAG and d.action == MAINTAIN


def test_rebaseline_after_three_stable_low_visits():
    st = eye_after_loading(300)
    st.ref_cst = 400  # simulate an eye whose reference is well above its current thickness
    for v, c in zip((5, 6, 7), (330, 340, 335)):
        d = st.step(Observation(v, c, bcva=70))
    assert abs(st.ref_cst - 335) < 1 and any("re-baseline" in t for t in d.trace)


def test_qc_failure_flags_and_never_sets_reference():
    st = EyeState(P)
    for v in (1, 2, 3):
        st.step(Observation(v, 450))
    d = st.step(Observation(4, 250, bcva=70, image_quality_ok=False))
    assert d.flagged and d.recommended == FLAG and not st.te_started and st.ref_cst is None
    d = st.step(Observation(5, 300, bcva=70))
    assert st.ref_cst == 300 and d.action == EXTEND


def test_qc_failure_skips_rebaseline():
    st = eye_after_loading(300)
    st.ref_cst = 400
    for v, c in zip((5, 6, 7), (330, 340, 335)):
        d = st.step(Observation(v, c, bcva=70, image_quality_ok=(v != 7)))
    assert st.ref_cst == 400 and d.flagged
