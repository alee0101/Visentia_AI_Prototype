"""Treat-and-extend copilot — physician review UI.

  .venv/bin/streamlit run app/streamlit_app.py

Retrospective replay of real TREX-DME eyes (OLIVES, CC-BY-4.0). At each visit the copilot reads
the OCT (layer-segmentation CST, fluid classifier), applies the TREX protocol with safety and
uncertainty overrides, drafts the note / patient message / schedule / FHIR, and waits for the
physician to approve or override. Every model output shown is out-of-fold for this eye.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from visentia_te import demo  # noqa: E402
from visentia_te.agent import review  # noqa: E402
from visentia_te.agent.copilot import run_visit  # noqa: E402
from visentia_te.agent.fhir import statuses  # noqa: E402
from visentia_te.data.olives_images import read_bscan  # noqa: E402

try:  # optional .env with ANTHROPIC_API_KEY
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))
except FileNotFoundError:
    pass

st.set_page_config(page_title="T&E Copilot", page_icon="👁", layout="wide")

BLUE, ORANGE, GRAY = "#257de0", "#f47230", "#8a8a8a"
SHAPE = {"EXTEND": "triangle-up", "MAINTAIN": "circle", "SHORTEN": "triangle-down", "LOADING": "square"}
ARROW = {"EXTEND": "▲", "MAINTAIN": "●", "SHORTEN": "▼", "LOADING": "■", "FLAG": "⚑"}
FEEDBACK = ROOT / "feedback.jsonl"


# ---- sidebar ---------------------------------------------------------------------------------
@st.cache_data
def eye_table() -> pd.DataFrame:
    return demo.eye_summary()


eyes = eye_table()
with st.sidebar:
    st.title("T&E Copilot")
    st.caption("Anti-VEGF treat-and-extend decision support for DME")
    label = {r.study_id: f"{r.study_id} · {r.arm} · {r.visits} visits" for r in eyes.itertuples()}
    ids = list(eyes.study_id)
    sid = st.selectbox("Patient eye", ids, index=ids.index(demo.DEFAULT_EYE) if demo.DEFAULT_EYE in ids else 0,
                       format_func=label.get)
    source = st.radio("CST source", list(demo.SOURCES), format_func=lambda s: demo.SOURCES[s][0], index=0,
                      help="Geometric = ILM→BM distance from a retina segmentation trained only on Duke DME. "
                           "Trial-measured = the device CST (oracle, for comparison).")
    k = st.slider("Uncertainty band (± SD)", 0.0, 1.5, demo.DEFAULT_K, 0.25,
                  help="Flag for physician review when CST ± k·SD crosses a protocol boundary.")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    use_llm = st.toggle("Draft with Claude", value=False, disabled=not has_key,
                        help="Needs ANTHROPIC_API_KEY. Off = deterministic templates (no network).")
    if not has_key:
        st.caption("No ANTHROPIC_API_KEY — using grounded templates.")
    physician = st.text_input("Physician ID", "dr_demo")
    show_truth = st.toggle("Show investigator's actual decision", value=True,
                           help="Retrospective replay: what the trial investigator actually did.")
    st.divider()
    st.caption("Research prototype — not a medical device. Data: OLIVES / TREX-DME (CC-BY-4.0), "
               "Duke DME (Chiu 2015). Dates are synthesized from trial weeks.")


@st.cache_data(show_spinner=False)
def replay(sid: str, source: str, k: float) -> list[demo.VisitView]:
    return demo.run_eye(sid, source, k)


views = replay(sid, source, k)
by_visit = {v.visit: v for v in views}
decision_visits = [v.visit for v in views if v.packet.action != "LOADING"]

tab_visit, tab_eye, tab_eval = st.tabs(["Visit review", "Eye timeline", "Validation"])


# ---- helpers ---------------------------------------------------------------------------------
def rec_label(p) -> str:
    if p.recommended == "FLAG":
        return f"⚑ Physician review — proposal {p.action} → {p.next_interval_wk} wk"
    return f"{ARROW[p.action]} {p.action} → {p.next_interval_wk} wk"


def agent_outputs(v: demo.VisitView, decision=None) -> dict:
    key = ("agent", sid, v.visit, source, k, use_llm, None if decision is None else json.dumps(decision.__dict__, default=str))
    if key not in st.session_state:
        with st.spinner("Drafting note and patient message…" if use_llm else None):
            st.session_state[key] = run_visit(v.packet, offline=not use_llm, approve=decision, feedback_path=FEEDBACK)
    return st.session_state[key]


@st.cache_data(show_spinner=False)
def scan_image(sid: str, visit: int, sl: int, fluid: bool, layers: bool):
    p = demo.bscan_path(sid, visit, sl)
    if p is None:
        return None
    img = read_bscan(p, (252, 256))
    return demo.overlay(img, demo.fluid_mask(img) if fluid else None, demo.boundaries(sid, visit, sl) if layers else None)


# ---- tab 1: visit review ---------------------------------------------------------------------
with tab_visit:
    if not decision_visits:
        st.info("This eye has no decision visits after loading.")
        st.stop()
    flagged = [v.visit for v in views if v.packet.flagged and v.packet.action != "LOADING"]
    c1, c2 = st.columns([3, 1])
    visit = c1.select_slider("Visit", options=[v.visit for v in views], value=(flagged or decision_visits)[0],
                             format_func=lambda x: f"V{x}{' ⚑' if x in flagged else ''}")
    c2.caption(f"{len(decision_visits)} decision visits · {len(flagged)} flagged")
    v = by_visit[visit]
    p = v.packet

    m = st.columns(5)
    m[0].metric("Visit date", f"{v.visit_date:%b %d, %Y}")
    m[0].caption(f"V{v.visit} · week {v.week:.0f}")
    if p.ref_cst_um and p.action != "LOADING":
        ch = p.cst_um / p.ref_cst_um - 1
        m[1].metric("CST", f"{p.cst_um:.0f} µm", f"{ch:+.0%} vs ref {p.ref_cst_um:.0f}",
                    delta_color="inverse" if abs(ch) >= 0.005 else "off")
    else:
        m[1].metric("CST", f"{p.cst_um:.0f} µm")
    if p.bcva_letters is not None:
        loss = (p.best_bcva_letters or p.bcva_letters) - p.bcva_letters
        m[2].metric("BCVA", f"{p.bcva_letters:.0f} letters", f"{-loss:+.0f} vs best" if loss else "at best", delta_color="normal" if loss else "off")
    m[3].metric("Interval in force", f"{p.current_interval_wk} wk")
    pf = (p.fluid_probs or {}).get("any_fluid")
    m[4].metric("P(foveal fluid)", "—" if pf is None else f"{pf:.2f}")

    with st.container(border=True):
        a, b = st.columns([2, 1])
        a.subheader(rec_label(p))
        cst_note = (f"CST input: {demo.SOURCES[source][0]} {v.cst_in:.0f} µm"
                    + (f" (±{1.5 * v.cst_sd:.0f} µm band)" if v.cst_sd else "")
                    + ("" if v.imaged or source == "trial" else " — no OCT for this visit, device value used")
                    + ("" if v.qc_ok else " — image QC failed"))
        a.caption(cst_note + f" · device CST {v.cst_true:.0f} µm")
        if show_truth:
            if v.trial_action:
                agree = v.trial_action == p.action
                b.markdown(f"**Investigator:** {ARROW[v.trial_action]} {v.trial_action}"
                           f" → {v.trial_next_interval:.1f} wk  \n{'✓ agrees with proposal' if agree else '✗ differs from proposal'}")
            else:
                b.caption("Investigator action not defined for this visit (loading or missing gap).")

    left, right = st.columns([1.15, 1])
    with left:
        sl = st.slider("B-scan (24 = fovea)", 20, 28, 24, key="slice")
        o1, o2 = st.columns(2)
        show_fluid = o1.checkbox("Fluid segmentation", True)
        show_layers = o2.checkbox("Retina boundaries (ILM / BM)", True)
        im = scan_image(sid, visit, sl, show_fluid, show_layers)
        if im is None:
            st.info("No OCT volume for this visit.")
        else:
            st.image(im, width=520)
            st.caption("Yellow = ILM, aqua = Bruch's membrane (thickness = distance between them), red = fluid.")
    with right:
        st.markdown("**Rule trace**")
        for i, t in enumerate(p.trace, 1):
            st.markdown(f"{i}. {'⚑ ' if t.startswith(('uncertain', 'image quality', 'flagged', 'safety')) else ''}{t}")
        if p.flagged:
            st.warning("Flagged for physician review: " + "; ".join(p.flag_reasons), icon=":material/flag:")

    st.divider()
    st.markdown("#### Drafts")
    out = agent_outputs(v)
    t1, t2, t3, t4, t5 = st.tabs(["Clinic note", "Patient message", "Schedule", "FHIR bundle", "Grounding check"])
    with t1:
        st.code(out["note"]["text"], language=None, wrap_lines=True)
        st.caption(f"source: {out['note']['source']}" + (f" · model {out['note']['model']}" if out["note"].get("model") else ""))
    with t2:
        st.info(out["patient_message"]["text"])
    with t3:
        s = out["schedule"]
        st.markdown(f"**Proposed slot:** {s['slot'] or 'none available — manual booking'}  \n"
                    f"Target {s['target_date']} · window {s['window_start']} → {s['window_end']}  \n{s['reason']}")
    with t4:
        st.caption(" · ".join(f"{r}: {s_}" for r, s_ in statuses(out["fhir"]).items()))
        st.json(out["fhir"], expanded=False)
    with t5:
        for name in ("note", "patient_message"):
            g = out["grounding"][name]
            (st.success if g["ok"] else st.error)(
                f"{name.replace('_', ' ')}: {len(g['checked'])} numbers checked, {len(g['ungrounded'])} ungrounded")
        st.caption("Every number in a draft must trace to the visit packet (engine inputs, trace, schedule). "
                   "Ungrounded drafts are retried once with feedback, then replaced by the template.")

    st.markdown("#### Physician sign-off")
    rkey = ("review", sid, visit, source, k)
    with st.form(f"review-{sid}-{visit}"):
        c = st.columns([1, 1, 1, 2])
        choice = c[0].radio("Decision", ["Approve", "Override"], horizontal=True)
        ov_action = c[1].selectbox("Action", review.OVERRIDE_ACTIONS, index=review.OVERRIDE_ACTIONS.index(p.action)
                                   if p.action in review.OVERRIDE_ACTIONS else 1)
        ov_iv = c[2].number_input("Interval (wk)", 4, 16, int(p.next_interval_wk), 2)
        reason = c[3].text_input("Reason (required to override)")
        if st.form_submit_button("Sign", type="primary"):
            try:
                d = (review.approve(physician) if choice == "Approve" else
                     review.override(physician, action=ov_action, interval_wk=int(ov_iv), reason=reason))
                st.session_state[rkey] = agent_outputs(v, d)
            except ValueError as e:
                st.error(str(e))
    if rkey in st.session_state:
        r = st.session_state[rkey]
        st.success(f"{r['status'].upper()} by {r['review']['physician_id']}: {r['review']['final']['action']} → "
                   f"{r['review']['final']['interval_wk']} wk · slot {r['schedule']['slot']} · logged to {Path(r['feedback_path']).name}")
        st.caption("FHIR now " + " · ".join(f"{a}: {b}" for a, b in statuses(r["fhir"]).items()))
        st.info(r["patient_message"]["text"])


# ---- tab 2: eye timeline ---------------------------------------------------------------------
with tab_eye:
    rows = []
    for x in views:
        q = x.packet
        rows.append(dict(visit=x.visit, week=x.week, date=str(x.visit_date), device_cst=x.cst_true, model_cst=x.cst_in,
                         lo=x.cst_in - 1.5 * x.cst_sd, hi=x.cst_in + 1.5 * x.cst_sd, ref=q.ref_cst_um if q.action != "LOADING" else np.nan,
                         copilot=q.recommended, action=q.action, interval=q.next_interval_wk, investigator=x.trial_action,
                         inv_interval=x.trial_next_interval, flagged=q.flagged, qc_ok=x.qc_ok))
    df = pd.DataFrame(rows)
    dec = df[df.action != "LOADING"]
    have = dec[dec.investigator.notna()]
    c = st.columns(4)
    c[0].metric("Decision visits", len(dec))
    c[1].metric("Agree with investigator", f"{(have.action == have.investigator).mean():.0%}" if len(have) else "—")
    c[2].metric("Flagged for review", int(dec.flagged.sum()))
    c[3].metric("Max interval reached", f"{dec.interval.max():.0f} wk" if len(dec) else "—")

    dev_name, cop_name = "Device CST (trial)", f"Copilot input ({demo.SOURCES[source][0].split(' (')[0]})"
    long = df.melt(id_vars=["week", "visit", "date"], value_vars=["device_cst", "model_cst"], var_name="series", value_name="cst")
    long["series"] = long.series.map({"device_cst": dev_name, "model_cst": cop_name})
    lo_y = float(np.nanmin([df.device_cst.min(), df.lo.min()])) - 20
    hi_y = float(np.nanmax([df.device_cst.max(), df.hi.max(), (df.ref * 1.2).max()])) + 20
    X = alt.X("week:Q", title="Week", scale=alt.Scale(domain=[0, float(df.week.max())]))
    Y = alt.Y("cst:Q", title="CST (µm)", scale=alt.Scale(domain=[lo_y, hi_y]))
    refs = df[df.ref.notna()].melt(id_vars=["week"], value_vars=["ref"], value_name="cst")
    refs = pd.concat([refs.assign(line="reference"), refs.assign(line="+10%", cst=refs.cst * 1.1), refs.assign(line="+20%", cst=refs.cst * 1.2)])
    hover = alt.selection_point(fields=["week"], nearest=True, on="pointerover", empty=False)
    band = alt.Chart(df).mark_area(opacity=0.18, color=ORANGE).encode(x=X, y=alt.Y("lo:Q", scale=alt.Scale(domain=[lo_y, hi_y])), y2="hi:Q")
    ref = alt.Chart(refs).mark_line(color=GRAY, interpolate="step-after", strokeWidth=1.2).encode(
        x=X, y=Y, detail="line:N",
        strokeDash=alt.StrokeDash("line:N", scale=alt.Scale(domain=["reference", "+10%", "+20%"], range=[[5, 3], [1, 3], [1, 3]]), legend=None))
    ref_lab = alt.Chart(refs.groupby("line").tail(1)).mark_text(align="left", dx=4, fontSize=11, color=GRAY).encode(x=X, y=Y, text="line:N")
    lines = alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=45, filled=True)).encode(
        x=X, y=Y, color=alt.Color("series:N", scale=alt.Scale(domain=[dev_name, cop_name], range=[BLUE, ORANGE]),
                                  legend=alt.Legend(orient="top", title=None)))
    rule = alt.Chart(df).mark_rule(color=GRAY).encode(
        x=X, opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("date:N"), alt.Tooltip("visit:Q"), alt.Tooltip("device_cst:Q", format=".0f", title="device CST"),
                 alt.Tooltip("model_cst:Q", format=".0f", title="copilot CST"), alt.Tooltip("ref:Q", format=".0f"),
                 alt.Tooltip("copilot:N"), alt.Tooltip("interval:Q", title="proposed wk"), alt.Tooltip("investigator:N"),
                 alt.Tooltip("inv_interval:Q", format=".1f", title="investigator wk")]).add_params(hover)
    sel = alt.Chart(df[df.visit == visit]).mark_rule(color=ORANGE, strokeWidth=1, strokeDash=[2, 2]).encode(x=X)
    st.markdown(f"**CST over time** — protocol reference with +10% / +20% lines; shaded = ±1.5·SD band; dashed orange = V{visit}")
    st.altair_chart((band + ref + ref_lab + lines + rule + sel).properties(height=320), width="stretch")

    rows_ = ["Copilot proposal", "Investigator"]
    strip = dec.melt(id_vars=["week", "visit", "flagged"], value_vars=["action", "investigator"], var_name="who", value_name="decision").dropna(subset=["decision"])
    strip["who"] = strip.who.map({"action": rows_[0], "investigator": rows_[1]})
    strip["flag"] = np.where((strip.who == rows_[0]) & strip.flagged, "⚑", "")
    strip["row"] = np.where(strip.who == rows_[0], 1, 0)
    Yw = alt.Y("row:Q", title=None, scale=alt.Scale(domain=[-0.6, 1.6]),
               axis=alt.Axis(values=[0, 1], grid=False, labelExpr=f"datum.value == 1 ? '{rows_[0]}' : '{rows_[1]}'", labelLimit=200))
    pts = alt.Chart(strip).mark_point(size=110, filled=True, color=GRAY, opacity=1).encode(
        x=X, y=Yw, shape=alt.Shape("decision:N", scale=alt.Scale(domain=list(SHAPE)[:3], range=list(SHAPE.values())[:3]),
                                   legend=alt.Legend(orient="top", title=None)),
        tooltip=["visit", "who", "decision", "flagged"])
    flags = alt.Chart(strip[strip.flag != ""]).mark_text(dy=-15, fontSize=13, color="#f2a600").encode(x=X, y=Yw, text="flag:N")
    st.markdown("**Decisions** — ⚑ = flagged for physician review (an extension is capped at MAINTAIN)")
    st.altair_chart(alt.layer(pts, flags).properties(height=150), width="stretch", height=170)
    with st.expander("Table view"):
        st.dataframe(df.round(1), hide_index=True, width="stretch")


# ---- tab 3: validation -----------------------------------------------------------------------
with tab_eval:
    R = ROOT / "eval" / "results"

    def J(name):
        f = R / name
        return json.loads(f.read_text()) if f.exists() else {}

    st.markdown("All numbers are out-of-fold (patient-level 5-fold CV); the geometric CST model never saw OLIVES.")
    seg, lay, bio, src = J("seg_cv.json"), J("layers_duke_cv.json"), J("biomarkers_cv.json"), J("cst_sources.json")
    st.markdown("##### Imaging")
    so = seg.get("overall", {})
    img_rows = [
        ("Fluid segmentation (Duke)", "pooled Dice vs grader 1", f"{so.get('pooled_dice_g1', np.nan):.2f}",
         f"inter-grader {so.get('inter_grader_pooled_dice', np.nan):.2f}"),
        ("Retina ILM→BM (Duke)", "thickness MAE per column", f"{lay.get('cv_mean', {}).get('col_thickness_mae_um_vs_g1', np.nan):.1f} µm",
         f"inter-grader {lay.get('cv_mean', {}).get('col_thickness_mae_um_g1_vs_g2', np.nan):.1f} µm"),
        ("Biomarkers (OLIVES)", "slice AUROC IRF / SRF / DRT",
         " / ".join(f"{bio.get(f'slice_{x}', {}).get('auroc', np.nan):.2f}" for x in ("irf", "srf", "drt")), ""),
    ]
    st.dataframe(pd.DataFrame(img_rows, columns=["Model", "Metric", "Result", "Reference"]), hide_index=True, width="stretch")
    st.markdown("##### CST source vs device CST (933 visits)")
    names = {"cnn": "CNN regression", "geo_cal": "Geometric", "geo_cal_qc_ok": "Geometric, QC pass", "stack": "Geometric + CNN"}
    cs = pd.DataFrame({names[k_]: src[k_] for k_ in names if k_ in src}).T
    if len(cs):
        st.dataframe(cs[["mae_um", "within_eye_sd_um", "change_mae_um", "band_agreement"]], width="stretch", column_config={
            "mae_um": st.column_config.NumberColumn("MAE (µm)", format="%.1f"),
            "within_eye_sd_um": st.column_config.NumberColumn("within-eye SD (µm)", format="%.1f"),
            "change_mae_um": st.column_config.NumberColumn("visit-to-visit change MAE (µm)", format="%.1f"),
            "band_agreement": st.column_config.NumberColumn("protocol band agreement", format="percent")})
        st.caption("Within-eye SD is what matters: decisions compare CST with the eye's own reference, so a constant "
                   "per-eye offset cancels, visit-to-visit noise does not. The protocol's extend band is ±10% (~30 µm).")
    st.markdown("##### Engine replay (449 imaged decisions vs investigators)")
    rows = []
    for tag, name in (("oracle", "Device CST (oracle)"), ("model", "CNN CST"), ("model_geo", "Geometric CST")):
        r = J(f"engine_{tag}.json")
        a = r.get("imaged_eyes") or r.get("model_no_uncertainty") or {}
        vo = r.get("vs_oracle_engine_no_uncertainty") or {}
        if a:
            rows.append(dict(inputs=name, accuracy=a["accuracy"], kappa=a["kappa_linear"], unsafe_extension=a["unsafe_extension_rate"],
                             same_call_as_oracle_engine=vo.get("agreement", np.nan)))
    pc = st.column_config.NumberColumn
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "accuracy": pc("accuracy vs investigator", format="percent"), "kappa": pc("linear κ", format="%.2f"),
        "unsafe_extension": pc("unsafe extension (engine EXTEND | investigator SHORTEN)", format="percent"),
        "same_call_as_oracle_engine": pc("same call as engine on device CST", format="percent")})
    st.caption("Even with perfect CST the written protocol agrees with investigators 65% of the time: the rest is "
               "investigator judgment. Geometric CST reproduces the device-CST engine on 91% of decisions (CNN: 81%); "
               "with the uncertainty flag, unflagged decisions agree 93% and flagged ones only 40% — the flag finds the errors.")
    sweep = []
    for tag, name in (("model", "CNN CST"), ("model_geo", "Geometric CST")):
        for s_ in J(f"engine_{tag}.json").get("abstention_sweep", []):
            sweep.append(dict(source=name, flag_rate=s_["flag_rate"], accuracy_unflagged=s_["accuracy_unflagged"], k=s_["k_sd"]))
    if sweep:
        sw = pd.DataFrame(sweep)
        ch = alt.Chart(sw).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
            x=alt.X("flag_rate:Q", title="Share of decisions sent to physician", axis=alt.Axis(format="%")),
            y=alt.Y("accuracy_unflagged:Q", title="Accuracy on the rest", scale=alt.Scale(zero=False), axis=alt.Axis(format="%")),
            color=alt.Color("source:N", scale=alt.Scale(range=[BLUE, ORANGE]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["source", "k", alt.Tooltip("flag_rate:Q", format=".0%"), alt.Tooltip("accuracy_unflagged:Q", format=".1%")])
        st.markdown("**Abstention trade-off** (k = ±SD band width)")
        st.altair_chart(ch.properties(height=280), width="stretch")
    for f, cap in (("layers_on_olives.png", "Retina boundaries on OLIVES (out-of-domain)"), ("seg_examples.png", "Fluid segmentation on Duke")):
        if (ROOT / "eval" / "figures" / f).exists():
            st.image(str(ROOT / "eval" / "figures" / f), caption=cap, width=860)
