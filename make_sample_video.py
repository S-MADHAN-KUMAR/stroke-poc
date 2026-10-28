"""Generate a SYNTHETIC side-view stick-figure walking clip with known joint angles.

Purpose: validate the angle/gait-feature maths against ground truth when no real
walking video is at hand. MediaPipe will not reliably detect a stick figure, so
for the real pipeline use a genuine side-on walking video (see README).

Ground truth: left knee peak flexion 60 deg, right (simulated paretic) 35 deg,
stride time 1.1 s.
"""
import json
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, SECONDS = 640, 480, 30, 12
STRIDE_S = 1.1
THIGH, SHANK, TRUNK = 105, 100, 120
OUT = Path(__file__).parent / "sample"


def leg(phase, peak_knee, hip_amp=22):
    hip = hip_amp * np.sin(phase)                                   # thigh angle from vertical
    knee = 4 + (peak_knee - 4) * np.maximum(0, np.sin(phase - 0.9)) ** 1.2
    return hip, knee


def main():
    OUT.mkdir(exist_ok=True)
    n = FPS * SECONDS
    gt = np.full((n, 33, 4), np.nan)
    vw = cv2.VideoWriter(str(OUT / "synthetic_walk.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    hip_c = np.array([W / 2, 230.0])
    for i in range(n):
        t = i / FPS
        img = np.full((H, W, 3), 245, np.uint8)
        cv2.line(img, (0, 430), (W, 430), (150, 150, 150), 2)
        sho = hip_c + [0, -TRUNK]
        pts = {11: sho, 12: sho + [0, 0], 23: hip_c, 24: hip_c + [0, 0]}
        for (hj, kj, aj), peak, off in (((23, 25, 27), 60.0, 0.0), ((24, 26, 28), 35.0, np.pi)):
            hip_a, knee_f = leg(2 * np.pi * t / STRIDE_S + off, peak)
            a1 = np.radians(hip_a); a2 = np.radians(hip_a - knee_f)
            knee = hip_c + THIGH * np.array([np.sin(a1), np.cos(a1)])
            ank = knee + SHANK * np.array([np.sin(a2), np.cos(a2)])
            pts[kj], pts[aj] = knee, ank
        for j, p in pts.items():
            gt[i, j] = [p[0] / W, p[1] / H, 0.0, 1.0]
        col = {"L": (200, 80, 0), "R": (0, 80, 200)}
        for (a, b), c in (((23, 25), "L"), ((25, 27), "L"), ((24, 26), "R"), ((26, 28), "R")):
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), col[c], 8)
        cv2.line(img, tuple(sho.astype(int)), tuple(hip_c.astype(int)), (60, 60, 60), 10)
        cv2.circle(img, tuple((sho + [0, -25]).astype(int)), 22, (60, 60, 60), -1)
        vw.write(img)
    vw.release()
    np.save(OUT / "synthetic_walk_groundtruth_landmarks.npy", gt)
    json.dump({"fps": FPS, "size": [W, H], "left_peak_knee_flexion_deg": 60,
               "right_peak_knee_flexion_deg": 35, "stride_time_s": STRIDE_S}, open(OUT / "groundtruth.json", "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
