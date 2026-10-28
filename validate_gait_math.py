"""Check the angle/gait maths against the synthetic clip's ground truth, and save a sample result.

This does NOT exercise MediaPipe (it feeds exact ground-truth landmarks into the same
feature code the real pipeline uses). It shows the feature extraction is correct;
detection quality on real video is a separate matter.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from pose_analysis import angle_timeseries, annotate_frames, compute_gait_features, contact_sheet

OUT = Path(__file__).parent / "sample"
gt = np.load(OUT / "synthetic_walk_groundtruth_landmarks.npy")
truth = json.load(open(OUT / "groundtruth.json"))
W, H = truth["size"]; fps = truth["fps"]

feats = compute_gait_features(gt, fps, (W, H))
print(json.dumps(feats, indent=2))

cap = cv2.VideoCapture(str(OUT / "synthetic_walk.mp4"))
frames = [cap.read()[1] for _ in range(len(gt))]
ang = angle_timeseries(gt, (W, H), fps)
ann = annotate_frames(frames, gt, (W, H), ang)
cv2.imwrite(str(OUT / "synthetic_contact_sheet.jpg"), contact_sheet(ann))
ang.to_csv(OUT / "synthetic_angles.csv", index=False)

checks = {
    "left peak knee flexion (truth 60)": feats["left"]["peak_knee_flexion_deg"],
    "right peak knee flexion (truth 35)": feats["right"]["peak_knee_flexion_deg"],
    "stride time s (truth 1.1)": feats["left"]["stride_time_s"],
}
print(checks)
assert abs(feats["left"]["peak_knee_flexion_deg"] - 60) < 3
assert abs(feats["right"]["peak_knee_flexion_deg"] - 35) < 3
assert abs(feats["left"]["stride_time_s"] - 1.1) < 0.1
json.dump({"features": feats, "note": "computed from ground-truth landmarks of a synthetic stick-figure clip, not MediaPipe output"},
          open(OUT / "sample_video_result.json", "w"), indent=2)
print("OK")
