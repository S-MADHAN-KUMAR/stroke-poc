"""Video gait analysis: MediaPipe Pose landmarks -> joint angles -> gait features.

Uses the MediaPipe Tasks API (`PoseLandmarker`). The model file (~6 MB) is
downloaded once to ./models on first use.

Assumes a SAGITTAL (side-on) view with one person walking. Angles are computed
in 2-D image space, so they are only meaningful for that camera geometry.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
MODEL_FILE = Path(__file__).parent / "models" / "pose_landmarker_lite.task"

# MediaPipe Pose landmark indices
L_SHO, R_SHO, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 23, 24, 25, 26, 27, 28
SKELETON = [(L_SHO, L_HIP), (R_SHO, R_HIP), (L_SHO, R_SHO), (L_HIP, R_HIP),
            (L_HIP, L_KNEE), (L_KNEE, L_ANK), (R_HIP, R_KNEE), (R_KNEE, R_ANK)]
MIN_VIS = 0.5


def ensure_model(path: Path = MODEL_FILE) -> Path:
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


# --------------------------------------------------------------------- extraction
def extract_landmarks(video_path: str, max_seconds: float = 20.0, max_width: int = 640,
                      model_path: Path | None = None):
    """Run MediaPipe PoseLandmarker over a video.

    Returns (landmarks[T,33,4] = x,y (normalised), z, visibility; NaN if no detection),
            fps, (width, height) of processed frames, list of BGR frames (for annotation).
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = ensure_model() if model_path is None else Path(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)

    lms, frames, wh = [], [], None
    with vision.PoseLandmarker.create_from_options(opts) as landmarker:
        i = 0
        while i < max_seconds * fps:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] > max_width:
                s = max_width / frame.shape[1]
                frame = cv2.resize(frame, (max_width, int(frame.shape[0] * s)))
            wh = (frame.shape[1], frame.shape[0])
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
            if res.pose_landmarks:
                lms.append([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]])
            else:
                lms.append(np.full((33, 4), np.nan))
            frames.append(frame)
            i += 1
    cap.release()
    if not lms:
        raise ValueError("No frames read from video.")
    return np.asarray(lms, float), fps, wh, frames


# ------------------------------------------------------------------- feature maths
def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle (deg) at b between segments b->a and b->c; inputs [T,2] pixel coords."""
    v1, v2 = a - b, c - b
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = (v1 * v2).sum(1)
    return np.degrees(np.abs(np.arctan2(cross, dot)))


def _clean(sig: np.ndarray, fps: float) -> np.ndarray:
    """Interpolate short gaps and low-pass with Savitzky-Golay."""
    s = pd.Series(sig).interpolate(limit=int(fps // 2), limit_direction="both").to_numpy()
    if np.isnan(s).any():
        return s
    win = max(5, int(fps / 4) // 2 * 2 + 1)
    return savgol_filter(s, min(win, len(s) // 2 * 2 - 1), 2) if len(s) > 7 else s


def angle_timeseries(lm: np.ndarray, wh: tuple[int, int], fps: float) -> pd.DataFrame:
    """Per-frame knee flexion (180 - hip-knee-ankle angle) and hip flexion (180 - shoulder-hip-knee)."""
    W, H = wh
    scale = np.array([W, H])
    out = {}
    for side, (sho, hip, kn, ank) in {"left": (L_SHO, L_HIP, L_KNEE, L_ANK),
                                       "right": (R_SHO, R_HIP, R_KNEE, R_ANK)}.items():
        xy = lambda j: lm[:, j, :2] * scale
        vis_ok = np.all(lm[:, [sho, hip, kn, ank], 3] >= MIN_VIS, axis=1)
        knee = 180 - joint_angle(xy(hip), xy(kn), xy(ank))
        hipf = 180 - joint_angle(xy(sho), xy(hip), xy(kn))
        knee[~vis_ok] = np.nan
        hipf[~vis_ok] = np.nan
        out[f"{side}_knee_flexion"] = _clean(knee, fps)
        out[f"{side}_hip_flexion"] = _clean(hipf, fps)
    df = pd.DataFrame(out)
    df.insert(0, "time_s", np.arange(len(df)) / fps)
    return df


def compute_gait_features(lm: np.ndarray, fps: float, wh: tuple[int, int]) -> dict:
    """Summarise a landmark sequence into clinically-flavoured gait features."""
    ang = angle_timeseries(lm, wh, fps)
    dur = len(ang) / fps
    feats: dict = {"duration_s": round(dur, 2), "fps": round(fps, 1),
                   "frames_with_pose_pct": round(100 * float(np.mean(~np.isnan(lm[:, L_KNEE, 0]))), 1)}
    rom = {}
    for side in ("left", "right"):
        k = ang[f"{side}_knee_flexion"].to_numpy()
        if np.isnan(k).all():
            feats[side] = None
            continue
        peaks, _ = find_peaks(np.nan_to_num(k, nan=np.nanmedian(k)), prominence=12,
                              distance=max(1, int(0.35 * fps)))
        rom[side] = float(np.nanpercentile(k, 95) - np.nanpercentile(k, 5))
        stride = float(np.median(np.diff(peaks)) / fps) if len(peaks) > 1 else None
        feats[side] = {
            "peak_knee_flexion_deg": round(float(np.nanmedian(k[peaks])) if len(peaks) else float(np.nanmax(k)), 1),
            "knee_rom_deg": round(rom[side], 1),
            "peak_hip_flexion_deg": round(float(np.nanpercentile(ang[f"{side}_hip_flexion"], 95)), 1),
            "swing_peaks_detected": int(len(peaks)),
            "stride_time_s": round(stride, 2) if stride else None,
        }
    n_steps = sum(f["swing_peaks_detected"] for f in (feats.get("left"), feats.get("right")) if f)
    feats["cadence_steps_per_min"] = round(n_steps / dur * 60, 1) if dur > 0 else None
    if len(rom) == 2:
        feats["knee_rom_asymmetry_pct"] = round(
            100 * abs(rom["left"] - rom["right"]) / (0.5 * (rom["left"] + rom["right"])), 1)
    return feats


# ------------------------------------------------------------------- visual output
def annotate_frames(frames, lm, wh, angles: pd.DataFrame, out_video: str | None = None, fps: float = 30):
    """Draw skeleton + knee angles. Optionally write an mp4; always return annotated frames."""
    W, H = wh
    out = []
    for i, fr in enumerate(frames):
        img = fr.copy()
        pts = {j: (int(lm[i, j, 0] * W), int(lm[i, j, 1] * H)) for j in {s for e in SKELETON for s in e}
               if not np.isnan(lm[i, j, 0])}
        for a, b in SKELETON:
            if a in pts and b in pts:
                cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
        for j, p in pts.items():
            cv2.circle(img, p, 4, (0, 0, 255), -1)
        for side, kj in (("left", L_KNEE), ("right", R_KNEE)):
            v = angles[f"{side}_knee_flexion"].iloc[i]
            if kj in pts and not np.isnan(v):
                cv2.putText(img, f"{v:.0f}", (pts[kj][0] + 8, pts[kj][1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 0), 1, cv2.LINE_AA)
        out.append(img)
    if out_video:
        vw = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for img in out:
            vw.write(img)
        vw.release()
    return out


def contact_sheet(frames: list, n: int = 6) -> np.ndarray:
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return np.hstack([frames[i] for i in idx])


def analyze_video(video_path: str, out_dir: str | Path, max_seconds: float = 20.0) -> dict:
    """Full pipeline: video -> landmarks -> angles -> features + annotated outputs."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lm, fps, wh, frames = extract_landmarks(video_path, max_seconds=max_seconds)
    ang = angle_timeseries(lm, wh, fps)
    feats = compute_gait_features(lm, fps, wh)
    ann = annotate_frames(frames, lm, wh, ang, str(out_dir / "annotated.mp4"), fps)
    cv2.imwrite(str(out_dir / "contact_sheet.jpg"), contact_sheet(ann))
    ang.to_csv(out_dir / "angles.csv", index=False)
    return {"features": feats, "angles": ang, "sheet": out_dir / "contact_sheet.jpg",
            "video": out_dir / "annotated.mp4"}
