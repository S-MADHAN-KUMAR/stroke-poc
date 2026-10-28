# Stroke rehabilitation outcome prediction: 1-day POC

End-to-end prototype: baseline patient data + video-derived gait features -> predicted walking speed with an
uncertainty interval -> therapy-scenario what-if -> predicted vs actual.

> **Synthetic data only. Not clinically validated. Not for patient care or treatment decisions.**

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -r requirements.txt
streamlit run app.py
```

The first run trains the model (~10 s) and caches it in `models/`. The first video analysis downloads the
MediaPipe `pose_landmarker_lite.task` file (~6 MB) into `models/` (needs internet once).

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: 4 tabs (prediction, video, scenarios, predicted vs actual) |
| `outcome_model.py` | Synthetic cohort generator, model, uncertainty, scenario comparison |
| `pose_analysis.py` | MediaPipe PoseLandmarker -> joint angles -> gait features, annotated output |
| `make_sample_video.py` | Builds a synthetic stick-figure walking clip with known joint angles |
| `validate_gait_math.py` | Checks the angle/gait maths against that clip's ground truth |
| `sample_prediction.py` | Writes the sample prediction / scenario / error to `sample/` |
| `sample/` | Sample outputs (see "Sample results") |

## Model

**Target:** walking speed (m/s, 10-metre walk) at the end of the therapy programme.

**Inputs (10):** age, days since stroke, NIHSS, Fugl-Meyer lower extremity, baseline walking speed,
paretic-knee ROM and left/right asymmetry (both video-derived), and therapy sessions/week, weeks, minutes/session.

**Synthetic data.** 2,000 patients from a generator in `outcome_model.py`. Severity measures are correlated
(NIHSS vs Fugl-Meyer vs baseline speed). Recovery follows a saturating dose-response curve
(`gain = capacity * (1 - exp(-0.05 * hours))`), with capacity falling with age, time since stroke and
poorer motor function. Noise grows with the size of the gain. Therapy dose is assigned partly by severity, to mimic
confounded clinic data.

**Algorithm.** Gradient-boosted trees (scikit-learn): one for the mean prediction, plus two quantile
models (10th/90th percentile). The quantile band is widened by **split-conformal calibration (CQR)** on a
held-out calibration set, which gives an interval with a coverage guarantee under exchangeability rather than
an uncalibrated one. Split: 60% train / 20% calibration / 20% test.

**Held-out results (synthetic test set, n = 400).** MAE 0.075 m/s (0.293 for a "no improvement" baseline),
R² 0.76, 80% interval covers 77.7% of cases (mean width 0.24 m/s).

## Computer vision

`pose_analysis.py` runs MediaPipe `PoseLandmarker` (video mode) frame by frame and takes hip, knee, ankle and
shoulder landmarks. Frames where any needed landmark has visibility < 0.5 are dropped and short gaps interpolated;
signals are smoothed with a Savitzky-Golay filter.

Features: **knee flexion** (180° − hip-knee-ankle angle), **hip flexion** (180° − shoulder-hip-knee), peak
knee flexion and ROM per leg, stride time and cadence (from swing-phase peaks of knee flexion), and **left/right
knee-ROM asymmetry**. ROM and asymmetry can feed the outcome model.

Requirements for valid output: side-on camera, one person, full body in frame, steady camera. Angles are 2-D
image-plane angles, so a camera that is not perpendicular to the walking direction biases them.

## Sample results

- **Sample prediction** (`sample/sample_prediction.json`): 67-year-old, 45 days post-stroke, NIHSS 8, FM-LE 20,
  baseline 0.45 m/s, therapy 3x/week x 8 weeks x 45 min -> predicted **0.73 m/s, 80% interval 0.59-0.82**.
  Entering an illustrative actual of 0.78 gives an error of +0.05 m/s (inside the interval).
- **Scenario:** 5x/week instead of 3x -> 0.82 m/s (+0.095). Intervals overlap. The synthetic generator's own
  effect is +0.084, which is only knowable because the data are synthetic.
- **Video analysis** (`sample/sample_video_result.json`, `sample/synthetic_contact_sheet.jpg`): on the synthetic
  stick-figure clip the pipeline recovers left peak knee flexion 59.9° (truth 60°), right 34.8° (truth 35°),
  stride time 1.1 s (truth 1.1 s), asymmetry 56%. **This result is computed from the clip's exact ground-truth
  landmarks, not from MediaPipe detections**, so it validates the angle/feature maths and not detection quality.
  Run a real side-on walking video through the app to exercise MediaPipe.

## Assumptions

1. Synthetic cohort statistics resemble real stroke patients only loosely; they are not drawn from any real data.
2. Outcome is measured at the end of the programme, so duration changes the horizon as well as the dose.
3. Camera is side-on and angles from 2-D video are comparable to lab-measured ones (they are not exactly).
4. Intervals assume the patient is exchangeable with the synthetic training population.

## Limitations (read before interpreting anything)

- **Nothing is validated.** The model is accurate on synthetic data because it learns the generator's rules. This says nothing about real patients.
- **Scenario comparison is not a treatment effect.** With no real treatment-response data (ideally randomised, with
  patients followed under different doses), the model only knows associations. Here dose was partly assigned by severity,
  so the learned effect is confounded; the real one is unknown. The what-if answers reflect the model's assumptions.
- **Dose-response shape is assumed.** The diminishing-returns curve was written into the generator; real curves vary by
  patient, therapy type and recovery phase.
- **Extrapolation.** Doses or patients outside the training range trigger a warning, but inside it the model can still be wrong.
- **Intervals** cover model/outcome noise under exchangeability; they do not cover unmeasured factors (cognition, motivation, comorbidity,
  home support), dataset shift, or misdetected landmarks.
- **Pose quality.** The lite model can fail on occluded limbs, walking aids, clothing, orthoses and poor lighting; pathological gait
  (circumduction, hip hiking) is outside what the simple knee/hip angle features capture.
- **Pose pipeline test coverage.** The feature maths was checked on synthetic ground truth; MediaPipe inference itself was not run
  in the build environment (model download was blocked) and should be tried on real video.
- **Single outcome, small feature set.** Real use would need clinically chosen outcomes, more covariates, external validation and ethical/regulatory review.

## Next steps toward something real

Real longitudinal data with recorded therapy dose; causal methods (randomised data, or g-methods/target-trial emulation with
adjustment for confounding); external validation; calibration checks by subgroup; video validation against motion-capture or
instrumented walkway data.
