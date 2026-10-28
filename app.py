"""Stroke rehab outcome-prediction POC (synthetic data, NOT clinically validated)."""
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import outcome_model as om

st.set_page_config(page_title="Stroke Rehab POC", layout="wide")
st.title("Stroke rehabilitation: outcome prediction POC")
st.error("**Research prototype on synthetic data.** Predictions are not clinically validated "
         "and must not be used for patient care or treatment decisions.")


@st.cache_resource(show_spinner="Training model on synthetic cohort…")
def get_model():
    return om.load_or_train()


model = get_model()
ss = st.session_state
ss.setdefault("log", [])
ss.setdefault("video_feats", None)

# ------------------------------------------------------------------ sidebar: patient
with st.sidebar:
    st.header("Baseline patient data")
    p = dict(om.DEFAULT_PATIENT)
    p["age"] = st.slider("Age (years)", 30, 90, p["age"])
    p["days_since_stroke"] = st.slider("Days since stroke", 7, 180, p["days_since_stroke"])
    p["nihss"] = st.slider("NIHSS (severity, higher = worse)", 1, 25, p["nihss"])
    p["fm_le"] = st.slider("Fugl-Meyer lower extremity (0-34)", 4, 34, p["fm_le"])
    p["baseline_speed"] = st.number_input("Baseline walking speed (m/s)", 0.05, 1.2, p["baseline_speed"], 0.05)
    vf = ss.video_feats
    use_video = st.checkbox("Use video-derived gait features", value=vf is not None, disabled=vf is None,
                            help="Analyse a video in the Video tab first.")
    if use_video and vf:
        p["knee_rom_deg"], p["asymmetry_pct"] = vf["knee_rom_deg"], vf["asymmetry_pct"]
        st.caption(f"Knee ROM {p['knee_rom_deg']:.1f}°, asymmetry {p['asymmetry_pct']:.1f}% (from video)")
    else:
        p["knee_rom_deg"] = st.slider("Paretic knee ROM in gait (°)", 8.0, 65.0, p["knee_rom_deg"])
        p["asymmetry_pct"] = st.slider("Knee ROM asymmetry (%)", 2.0, 60.0, p["asymmetry_pct"])

    st.header("Therapy plan (current)")
    t = dict(om.DEFAULT_THERAPY)
    t["sessions_per_week"] = st.slider("Sessions / week", 1, 7, t["sessions_per_week"])
    t["weeks"] = st.slider("Duration (weeks)", 2, 16, t["weeks"])
    t["session_min"] = st.select_slider("Minutes / session", [20, 30, 45, 60, 90], t["session_min"])

tab_pred, tab_video, tab_scen, tab_actual = st.tabs(
    ["1. Prediction", "2. Video analysis", "3. Therapy scenarios", "4. Predicted vs actual"])

pred = model.predict_one(p, t)

# ------------------------------------------------------------------------ prediction
with tab_pred:
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted walking speed at end of programme", f"{pred['pred']:.2f} m/s")
    c2.metric(f"{int(model.metrics['nominal_coverage']*100)}% prediction interval",
              f"{pred['lo']:.2f} – {pred['hi']:.2f} m/s")
    c3.metric("Expected change vs baseline", f"{pred['pred'] - p['baseline_speed']:+.2f} m/s")
    for w in model.ood_warnings(p, t):
        st.warning(w)
    with st.expander("Model details and held-out performance"):
        m = model.metrics
        st.markdown(
            f"- Gradient boosting (mean) + quantile gradient boosting, widened by split-conformal calibration (CQR).\n"
            f"- Synthetic cohort: {m['n_train']} train / {m['n_calibration']} calibration / {m['n_test']} test.\n"
            f"- Test MAE **{m['mae']:.3f} m/s** (vs {m['mae_no_change_baseline']:.3f} for 'no improvement'), R² {m['r2']:.2f}.\n"
            f"- Interval coverage on test **{m['interval_coverage']:.0%}** (nominal {m['nominal_coverage']:.0%}), "
            f"mean width {m['mean_interval_width']:.2f} m/s.\n"
            f"- These numbers describe fit to *synthetic* data only.")

# ------------------------------------------------------------------------------ video
with tab_video:
    st.write("Upload a short **side-on** walking video (one person, full body visible, ~5-20 s).")
    side = st.radio("Affected (paretic) side", ["left", "right"], horizontal=True)
    up = st.file_uploader("Walking video", type=["mp4", "mov", "avi", "m4v"])
    if up and st.button("Analyse video", type="primary"):
        import pose_analysis as pa
        tmp = Path(tempfile.mkdtemp())
        vp = tmp / up.name
        vp.write_bytes(up.getvalue())
        try:
            with st.spinner("Running MediaPipe Pose (first run downloads the model)…"):
                res = pa.analyze_video(str(vp), tmp / "out")
            ss.video_res, ss.video_side = res, side
            f = res["features"]
            if f.get(side) and "knee_rom_asymmetry_pct" in f:
                ss.video_feats = {"knee_rom_deg": f[side]["knee_rom_deg"],
                                  "asymmetry_pct": f["knee_rom_asymmetry_pct"]}
        except Exception as e:  # surface model-download / decoding problems clearly
            st.error(f"Video analysis failed: {e}")
    res = ss.get("video_res")
    if res:
        f = res["features"]
        st.image(str(res["sheet"]), caption="Annotated frames (green = skeleton, yellow = knee flexion °)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cadence", f"{f['cadence_steps_per_min']} steps/min")
        c2.metric("Knee ROM asymmetry", f"{f.get('knee_rom_asymmetry_pct', '–')} %")
        c3.metric("Frames with pose", f"{f['frames_with_pose_pct']} %")
        st.dataframe(pd.DataFrame({k: f[k] for k in ("left", "right") if f.get(k)}))
        st.line_chart(res["angles"].set_index("time_s")[["left_knee_flexion", "right_knee_flexion"]])
        st.caption("Knee flexion = 180° − hip-knee-ankle angle, 2-D image plane. "
                   "Valid only for a true side-on view; low-visibility frames are interpolated.")
        if ss.video_feats:
            st.success("Knee ROM and asymmetry now feed the prediction (tick the box in the sidebar).")

# ------------------------------------------------------------------------- scenarios
with tab_scen:
    st.write("Alter the plan and see how the model's prediction changes.")
    st.caption("Scenario A is the plan in the sidebar.")
    b1, b2, b3 = st.columns(3)
    t_b = {"sessions_per_week": b1.slider("B: sessions / week", 1, 7, min(t["sessions_per_week"] + 1, 7)),
           "weeks": b2.slider("B: duration (weeks)", 2, 16, t["weeks"]),
           "session_min": b3.select_slider("B: minutes / session", [20, 30, 45, 60, 90], t["session_min"])}
    cmp = om.compare_scenarios(model, p, t, t_b)
    d = pd.DataFrame({
        "Total therapy (h)": [cmp["hours_A"], cmp["hours_B"]],
        "Predicted speed (m/s)": [cmp["A"]["pred"], cmp["B"]["pred"]],
        "Interval low": [cmp["A"]["lo"], cmp["B"]["lo"]],
        "Interval high": [cmp["A"]["hi"], cmp["B"]["hi"]]}, index=["A (current)", "B (alternative)"]).round(2)
    st.dataframe(d)
    st.metric("Model-estimated difference (B − A)", f"{cmp['delta']:+.3f} m/s")
    if cmp["intervals_overlap"]:
        st.info("The prediction intervals overlap heavily: the model cannot separate these plans "
                "from patient-to-patient variability.")
    for w in cmp["warnings"]:
        st.warning(w)
    st.warning(
        "**Not causal.** The model learned an *association* between therapy dose and outcome in synthetic data. "
        "Dose was assigned partly according to patient severity, so the estimated benefit of more therapy is "
        "confounded, and with no real treatment-response data (ideally randomised) it cannot be validated. "
        "Treat the difference as a what-if illustration of the model, not an expected treatment effect.")
    with st.expander("Synthetic ground truth (only possible because the data are synthetic)"):
        tr = om.synthetic_truth_delta(p, t, t_b)
        st.write(f"Data-generator effect of B vs A: **{tr:+.3f} m/s** vs model estimate **{cmp['delta']:+.3f} m/s**. "
                 "In real use no such ground truth exists.")

# ----------------------------------------------------------------------- pred vs actual
with tab_actual:
    st.write("Enter the measured outcome for the patient/plan currently set in the sidebar.")
    actual = st.number_input("Actual walking speed at end of programme (m/s)", 0.0, 2.0, 0.0, 0.01)
    if st.button("Record", type="primary") and actual > 0:
        ss.log.append({"predicted": round(pred["pred"], 2), "interval": f"{pred['lo']:.2f}-{pred['hi']:.2f}",
                       "actual": actual, "error (actual − pred)": round(actual - pred["pred"], 2),
                       "abs error": round(abs(actual - pred["pred"]), 2),
                       "inside interval": pred["lo"] <= actual <= pred["hi"]})
    if ss.log:
        last = ss.log[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted", f"{last['predicted']:.2f} m/s")
        c2.metric("Actual", f"{last['actual']:.2f} m/s")
        c3.metric("Error (actual − predicted)", f"{last['error (actual − pred)']:+.2f} m/s")
        log = pd.DataFrame(ss.log)
        st.dataframe(log)
        st.caption(f"Running MAE over {len(log)} record(s): {log['abs error'].mean():.3f} m/s. "
                   "A handful of cases says little about model quality.")
