"""Produce the sample prediction, a scenario comparison, and a predicted-vs-actual line."""
import json
import outcome_model as om

m = om.load_or_train()
p = {**om.DEFAULT_PATIENT, "knee_rom_deg": 31.1, "asymmetry_pct": 56.3}  # video-derived values from sample clip
a = dict(om.DEFAULT_THERAPY)                      # 3 x/wk, 8 wk, 45 min
b = {**a, "sessions_per_week": 5}                 # alternative: 5 x/wk
pred = m.predict_one(p, a)
cmp = om.compare_scenarios(m, p, a, b)
actual = 0.78
out = {"patient": p, "therapy": a, "prediction": {k: round(v, 3) for k, v in pred.items()},
       "scenario_B": b, "scenario": {"pred_A": round(cmp["A"]["pred"], 3), "pred_B": round(cmp["B"]["pred"], 3),
       "delta": round(cmp["delta"], 3), "intervals_overlap": cmp["intervals_overlap"],
       "synthetic_truth_delta": round(om.synthetic_truth_delta(p, a, b), 3)},
       "illustrative_actual": actual, "error": round(actual - pred["pred"], 3),
       "metrics": {k: round(v, 3) for k, v in m.metrics.items()}}
json.dump(out, open("sample/sample_prediction.json", "w"), indent=2)
print(json.dumps(out, indent=2))
