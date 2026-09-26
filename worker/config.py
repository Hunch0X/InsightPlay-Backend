"""Central worker configuration. Everything is overridable through environment variables."""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, fields


def _env(name, default):
    v = os.getenv(name)
    if v is None or v == "":
        return default
    if isinstance(default, bool):
        return v.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


@dataclass
class Config:
    # --- infrastructure
    database_url: str = "postgres://insightplay:insightplay@localhost:5432/insightplay"
    redis_url: str = "redis://localhost:6379"
    queue_key: str = "insightplay:jobs"
    progress_channel: str = "insightplay:progress"
    max_attempts: int = 2
    stale_job_seconds: int = 180

    # --- detection / tracking
    detector_weights: str = "yolo26m.pt"          # swap for football-trained weights via DETECTOR_WEIGHTS
    detector_class_map: str = ""                  # JSON {"0":"player","1":"goalkeeper",...}; auto-detected from class names when empty
    det_conf: float = 0.20
    imgsz: int = 1280
    device: str = ""                              # "" = auto (cuda if available)
    tracker_cfg: str = os.path.join(os.path.dirname(__file__), "config", "botsort.yaml")
    analysis_fps: float = 10.0
    cut_threshold: float = 0.55                   # histogram distance that marks a camera cut (broadcast footage)

    # --- ball
    ball_weights: str = ""                        # optional dedicated ball model
    ball_conf: float = 0.12
    ball_tiling: bool = False                     # run the ball model on overlapping tiles (slower, finds small balls)
    ball_max_gap_frames: int = 6                  # interpolate gaps up to this many analysed frames (flagged interpolated)
    ball_outlier_frac: float = 0.10               # residual vs rolling median, as a fraction of frame width

    # --- appearance / OCR
    reid_encoder: str = "color"                   # color | osnet
    jersey_ocr: str = "none"                      # none | easyocr
    ocr_min_h: int = 80
    ocr_max_samples: int = 12
    ocr_min_gap_s: float = 1.5

    # --- pitch calibration
    calibration_mode: str = "auto"                # auto | manual | keypoint | none
    pitch_keypoint_weights: str = ""              # YOLO-pose model emitting the landmarks in pitch_layout
    pitch_layout_path: str = ""
    calib_kp_conf: float = 0.5
    calib_min_points: int = 5
    calib_max_err_m: float = 1.5
    calib_hold_frames: int = 0                    # reuse the last good homography for this many frames (fixed cameras only)

    # --- pose (optional)
    pose_enabled: bool = False
    pose_weights: str = "yolo26n-pose.pt"
    pose_max_crops: int = 20000

    # --- identity
    identity_threshold: float = 0.80
    team_outlier_factor: float = 2.5
    min_track_obs: int = 15

    # --- physical
    max_speed_ms: float = 12.0
    max_gap_s: float = 1.0
    smooth_window: int = 7
    hsr_kmh: float = 19.8
    sprint_kmh: float = 25.2
    min_sprint_s: float = 1.0
    accel_thr: float = 3.0
    min_accel_s: float = 0.5
    heatmap_nx: int = 21
    heatmap_ny: int = 14
    path_points: int = 1500

    # --- events
    control_dist: float = 0.70                    # ball-to-feet distance (in player heights) that counts as control
    release_dist: float = 1.10
    min_control_frames: int = 2
    release_frames: int = 3
    merge_gap_s: float = 1.0                      # same-player spells closer than this merge into one carry
    max_pass_gap_s: float = 3.0
    min_pass_m: float = 2.5
    min_pass_norm: float = 1.5                    # uncalibrated fallback, in player heights
    tackle_gap_s: float = 0.6
    pressure_radius_m: float = 3.0
    duel_radius_m: float = 2.0
    shot_min_speed: float = 13.0
    shot_max_dist_m: float = 40.0
    progressive_frac: float = 0.25
    min_carry_m: float = 5.0

    # --- AI
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"
    ai_max_players: int = 30
    ai_min_minutes: float = 10.0

    @classmethod
    def from_env(cls) -> "Config":
        kw = {}
        for f in fields(cls):
            kw[f.name] = _env(f.name.upper(), f.default)
        cfg = cls(**kw)
        over = os.getenv("THRESHOLDS_JSON")
        if over:
            for k, v in json.loads(over).items():
                if not hasattr(cfg, k):
                    raise ValueError(f"THRESHOLDS_JSON: unknown setting {k}")
                setattr(cfg, k, v)
        return cfg
