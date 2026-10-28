"""Synthetic stroke-rehab cohort + outcome model with uncertainty.

Outcome predicted: walking speed (m/s, 10-metre walk test) at the END of the
therapy programme.

IMPORTANT: everything here is synthetic. The dose-response relationship is an
assumption written into the data generator (see `expected_end_speed`), so the
model can only ever "learn" that assumption back. It is NOT clinical evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_PATH = Path(__file__).parent / "models" / "outcome_model.joblib"

BASELINE_FEATURES = [
    "age",
    "days_since_stroke",
    "nihss",              # stroke severity, 0-42 (higher = worse)
    "fm_le",              # Fugl-Meyer lower extremity, 0-34 (higher = better)
    "baseline_speed",     # m/s at start of therapy
    "knee_rom_deg",       # paretic-side knee flexion range during gait (video-derived)
    "asymmetry_pct",      # left/right knee ROM asymmetry (video-derived)
]
THERAPY_FEATURES = ["sessions_per_week", "weeks", "session_min"]
FEATURES = BASELINE_FEATURES + THERAPY_FEATURES
TARGET = "end_speed"

DEFAULT_PATIENT = {
    "age": 67,
    "days_since_stroke": 45,
    "nihss": 8,
    "fm_le": 20,
    "baseline_speed": 0.45,
    "knee_rom_deg": 32.0,
    "asymmetry_pct": 28.0,
}
DEFAULT_THERAPY = {"sessions_per_week": 3, "weeks": 8, "session_min": 45}


def total_hours(sessions_per_week, weeks, session_min):
    return sessions_per_week * weeks * session_min / 60.0


def expected_end_speed(row) -> float:
    """Noise-free 'true' mechanism of the synthetic world (diminishing returns)."""
    hrs = total_hours(row["sessions_per_week"], row["weeks"], row["session_min"])
    capacity = 0.25 + 0.50 * row["fm_le"] / 34.0
    capacity *= 0.5 + 0.5 * np.exp(-row["days_since_stroke"] / 150.0)   # earlier = more plastic
    capacity *= 1.0 - 0.004 * max(row["age"] - 60, 0)
    gain = capacity * (1.0 - np.exp(-0.05 * hrs))
    return float(np.clip(row["baseline_speed"] + gain, 0.05, 1.8))


def generate_cohort(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(65, 11, n), 30, 90)
    days = rng.uniform(7, 180, n)
    nihss = np.clip(rng.gamma(2.2, 3.0, n) + 1, 1, 25)
    fm_le = np.clip(34 - 1.1 * nihss + rng.normal(0, 4, n), 4, 34)
    base = np.clip(0.12 + 0.012 * fm_le - 0.002 * (age - 65) + rng.normal(0, 0.06, n), 0.05, 1.2)
    knee_rom = np.clip(18 + 0.9 * fm_le + rng.normal(0, 5, n), 8, 65)
    asym = np.clip(45 - 0.9 * fm_le + rng.normal(0, 8, n), 2, 60)

    # Dose assignment is mildly confounded: milder patients tend to get more sessions.
    # (Realistic for observational clinic data, and the reason scenario answers are shaky.)
    sess_options = np.array([2, 3, 4, 5])
    logits = (fm_le[:, None] - 20) / 15.0 * (sess_options[None, :] - 3.5)
    p = np.exp(logits); p /= p.sum(1, keepdims=True)
    sessions = np.array([rng.choice(sess_options, p=pi) for pi in p])
    weeks = rng.choice([4, 6, 8, 12], n)
    minutes = rng.choice([30, 45, 60], n)

    df = pd.DataFrame(dict(
        age=age.round(0), days_since_stroke=days.round(0), nihss=nihss.round(0),
        fm_le=fm_le.round(0), baseline_speed=base.round(2), knee_rom_deg=knee_rom.round(1),
        asymmetry_pct=asym.round(1), sessions_per_week=sessions, weeks=weeks, session_min=minutes,
    ))
    mean = df.apply(expected_end_speed, axis=1).to_numpy()
    gain = mean - df["baseline_speed"].to_numpy()
    sd = 0.04 + 0.15 * gain                      # heteroscedastic: bigger gains, more spread
    df[TARGET] = np.clip(mean + rng.normal(0, sd), 0.05, 1.8).round(2)
    return df


@dataclass
class OutcomeModel:
    mean_model: GradientBoostingRegressor
    lo_model: GradientBoostingRegressor
    hi_model: GradientBoostingRegressor
    conformal_q: float
    alpha: float
    train_ranges: dict
    metrics: dict = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[FEATURES]
        mean = self.mean_model.predict(X)
        lo = self.lo_model.predict(X) - self.conformal_q
        hi = self.hi_model.predict(X) + self.conformal_q
        lo, hi = np.minimum(lo, mean), np.maximum(hi, mean)   # keep point inside the band
        return pd.DataFrame({"pred": mean, "lo": np.clip(lo, 0, None), "hi": hi}, index=X.index)

    def predict_one(self, patient: dict, therapy: dict) -> dict:
        row = pd.DataFrame([{**patient, **therapy}])
        return self.predict(row).iloc[0].to_dict()

    def ood_warnings(self, patient: dict, therapy: dict) -> list[str]:
        warn = []
        row = {**patient, **therapy}
        for k, (lo, hi) in self.train_ranges.items():
            if k == "total_hours":
                continue                      # handled below as a derived quantity
            if not (lo <= row[k] <= hi):
                warn.append(f"`{k}` = {row[k]} is outside the range seen in training data ({lo:.3g}-{hi:.3g}).")
        hrs = total_hours(row["sessions_per_week"], row["weeks"], row["session_min"])
        lo, hi = self.train_ranges["total_hours"]
        if not lo <= hrs <= hi:
            warn.append(f"Total therapy dose {hrs:.0f} h is outside the training range ({lo:.0f}-{hi:.0f} h).")
        return warn


def train(df: pd.DataFrame | None = None, alpha: float = 0.2, seed: int = 0) -> OutcomeModel:
    """Mean GBM + quantile GBMs, widened by split-conformal calibration (CQR)."""
    df = generate_cohort(seed=seed) if df is None else df
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)

    kw = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed)
    mean_m = GradientBoostingRegressor(loss="squared_error", **kw).fit(X_tr, y_tr)
    lo_m = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, **kw).fit(X_tr, y_tr)
    hi_m = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2, **kw).fit(X_tr, y_tr)

    # Conformalized quantile regression: widen the band so coverage holds on calibration data
    scores = np.maximum(lo_m.predict(X_cal) - y_cal, y_cal - hi_m.predict(X_cal))
    n = len(scores)
    q = float(np.quantile(scores, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)))

    ranges = {c: (float(df[c].min()), float(df[c].max())) for c in FEATURES}
    hrs = total_hours(df.sessions_per_week, df.weeks, df.session_min)
    ranges["total_hours"] = (float(hrs.min()), float(hrs.max()))

    model = OutcomeModel(mean_m, lo_m, hi_m, q, alpha, ranges)
    pred = model.predict(X_te)
    base_mae = mean_absolute_error(y_te, X_te["baseline_speed"])   # "no improvement" baseline
    model.metrics = {
        "n_train": len(X_tr), "n_calibration": len(X_cal), "n_test": len(X_te),
        "mae": float(mean_absolute_error(y_te, pred["pred"])),
        "r2": float(r2_score(y_te, pred["pred"])),
        "mae_no_change_baseline": float(base_mae),
        "interval_coverage": float(((y_te >= pred["lo"]) & (y_te <= pred["hi"])).mean()),
        "nominal_coverage": 1 - alpha,
        "mean_interval_width": float((pred["hi"] - pred["lo"]).mean()),
    }
    return model


def load_or_train(force: bool = False) -> OutcomeModel:
    if MODEL_PATH.exists() and not force:
        try:
            return joblib.load(MODEL_PATH)
        except Exception:          # stale / incompatible cache (e.g. other sklearn version): retrain
            pass
    MODEL_PATH.parent.mkdir(exist_ok=True)
    m = train()
    joblib.dump(m, MODEL_PATH)
    return m


def compare_scenarios(model: OutcomeModel, patient: dict, therapy_a: dict, therapy_b: dict) -> dict:
    a, b = model.predict_one(patient, therapy_a), model.predict_one(patient, therapy_b)
    return {
        "A": a, "B": b,
        "delta": b["pred"] - a["pred"],
        "intervals_overlap": not (a["hi"] < b["lo"] or b["hi"] < a["lo"]),
        "hours_A": total_hours(**therapy_a), "hours_B": total_hours(**therapy_b),
        "warnings": sorted(set(model.ood_warnings(patient, therapy_a) + model.ood_warnings(patient, therapy_b))),
    }


def synthetic_truth_delta(patient: dict, therapy_a: dict, therapy_b: dict) -> float:
    """Only computable because the data are synthetic; used to show how far the model's
    estimated therapy effect can be from the built-in mechanism."""
    return expected_end_speed({**patient, **therapy_b}) - expected_end_speed({**patient, **therapy_a})


if __name__ == "__main__":
    import outcome_model          # re-import so the pickled class path is `outcome_model.OutcomeModel`
    m = outcome_model.load_or_train(force=True)
    for k, v in m.metrics.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
