"""Stage 4 — pitch calibration: camera pixels -> pitch metres.

Two real sources of homographies (no guessing):
  * manual   : >= 4 pixel<->pitch point pairs supplied for a FIXED camera (PUT /api/videos/:id/calibration)
  * keypoint : a YOLO-pose model that outputs pitch landmarks (PITCH_KEYPOINT_WEIGHTS), one homography per frame,
               which is what panning broadcast/tactical cameras require.
If neither is available, pitch coordinates stay NULL and every metric that needs them is reported as not_available.
"""
from __future__ import annotations
import json
import logging
import os

import cv2
import numpy as np

from db import bulk_update, copy_rows, execute, fetch_df, to_json

log = logging.getLogger("calibration")
NAME = "calibration"


# ------------------------------------------------------------------------------------------------------ pitch layout
def default_layout(L: float = 105.0, W: float = 68.0) -> list[dict]:
    """29 pitch landmarks (FIFA-standard marking dimensions), in this order. A keypoint model must be trained to emit
    them in this order — or edit/replace the layout with PITCH_LAYOUT_PATH to match your own model."""
    pbl, pbw, gbl, gbw, ccr, psd = 16.5, 40.32, 5.5, 18.32, 9.15, 11.0
    cy = W / 2
    pts = [
        ("corner_tl", 0, 0), ("pen_box_tl", 0, (W - pbw) / 2), ("goal_box_tl", 0, (W - gbw) / 2), ("goal_box_bl", 0, (W + gbw) / 2),
        ("pen_box_bl", 0, (W + pbw) / 2), ("corner_bl", 0, W),
        ("goal_box_in_t_l", gbl, (W - gbw) / 2), ("goal_box_in_b_l", gbl, (W + gbw) / 2), ("pen_spot_l", psd, cy),
        ("pen_box_in_t_l", pbl, (W - pbw) / 2), ("pen_box_in_b_l", pbl, (W + pbw) / 2),
        ("halfway_t", L / 2, 0), ("halfway_b", L / 2, W), ("circle_t", L / 2, cy - ccr), ("circle_b", L / 2, cy + ccr),
        ("pen_box_in_t_r", L - pbl, (W - pbw) / 2), ("pen_box_in_b_r", L - pbl, (W + pbw) / 2), ("pen_spot_r", L - psd, cy),
        ("goal_box_in_t_r", L - gbl, (W - gbw) / 2), ("goal_box_in_b_r", L - gbl, (W + gbw) / 2),
        ("corner_tr", L, 0), ("pen_box_tr", L, (W - pbw) / 2), ("goal_box_tr", L, (W - gbw) / 2), ("goal_box_br", L, (W + gbw) / 2),
        ("pen_box_br", L, (W + pbw) / 2), ("corner_br", L, W),
        ("circle_l", L / 2 - ccr, cy), ("circle_r", L / 2 + ccr, cy), ("centre_spot", L / 2, cy),
    ]
    return [{"name": n, "x": float(x), "y": float(y)} for n, x, y in pts]


def load_layout(path: str, L: float, W: float) -> list[dict]:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default_layout(L, W)


# ------------------------------------------------------------------------------------------------------- homography
def fit_homography(px: np.ndarray, pitch: np.ndarray, ransac_thr_m: float = 1.0):
    """pixels (N,2) -> pitch metres (N,2). Returns (H, mean_error_m, n_inliers) or None for degenerate input."""
    px, pitch = np.asarray(px, np.float32), np.asarray(pitch, np.float32)
    if len(px) < 4:
        return None
    sv = np.linalg.svd(pitch - pitch.mean(0), compute_uv=False)
    if sv[1] < 2.0:  # points (nearly) collinear on the pitch: homography is ill-conditioned
        return None
    if len(px) == 4:
        H, mask = cv2.findHomography(px, pitch, 0), np.ones(4, np.uint8)
    else:
        H, mask = cv2.findHomography(px, pitch, cv2.RANSAC, ransac_thr_m)
    if H is None or mask is None:
        return None
    inl = mask.ravel().astype(bool)
    if inl.sum() < 4 or not np.all(np.isfinite(H)):
        return None
    proj = cv2.perspectiveTransform(px[inl][None], H)[0]
    err = float(np.mean(np.linalg.norm(proj - pitch[inl], axis=1)))
    return H, err, int(inl.sum())


def apply_homographies(px: np.ndarray, py: np.ndarray, frames: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray,
                       seg_H: np.ndarray, seg_conf: np.ndarray):
    """Vectorised projection of points using the segment covering each row's frame. Rows without a segment -> NaN."""
    order = np.argsort(seg_start)
    seg_start, seg_end, seg_H, seg_conf = seg_start[order], seg_end[order], seg_H[order], seg_conf[order]
    idx = np.searchsorted(seg_start, frames, side="right") - 1
    valid = (idx >= 0) & (frames <= seg_end[np.clip(idx, 0, None)])
    idx = np.clip(idx, 0, None)
    Hn = seg_H[idx]
    v = np.stack([px, py, np.ones_like(px)], axis=1)
    out = np.einsum("nij,nj->ni", Hn, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = out[:, 0] / out[:, 2], out[:, 1] / out[:, 2]
    x, y = np.where(valid, x, np.nan), np.where(valid, y, np.nan)
    return x, y, np.where(valid, seg_conf[idx], np.nan)


# ------------------------------------------------------------------------------------------------------- calibrators
class KeypointCalibrator:
    def __init__(self, cfg, layout: list[dict]):
        from ultralytics import YOLO
        self.cfg, self.model = cfg, YOLO(cfg.pitch_keypoint_weights)
        self.layout = np.array([[p["x"], p["y"]] for p in layout], np.float32)

    def calibrate(self, frame):
        r = self.model.predict(frame, conf=0.25, verbose=False, device=self.cfg.device or None)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return None
        i = int(r.boxes.conf.argmax())
        xy = r.keypoints.xy[i].cpu().numpy()
        kc = r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(xy))
        idx = [k for k in range(min(len(xy), len(self.layout))) if kc[k] >= self.cfg.calib_kp_conf and xy[k].sum() > 0]
        if len(idx) < self.cfg.calib_min_points:
            return None
        return fit_homography(xy[idx], self.layout[idx], 1.0)


def _confidence(err: float, n: int, max_err: float) -> float:
    return float(np.clip(1.0 - err / max_err, 0.0, 1.0) * min(1.0, n / 8.0))


def choose_mode(cfg, video: dict) -> tuple[str, str]:
    m = cfg.calibration_mode
    has_manual = bool(video.get("calibration_points")) and len(video["calibration_points"]) >= 4
    has_kp = bool(cfg.pitch_keypoint_weights)
    if m == "manual" or (m == "auto" and has_manual):
        return ("manual", "") if has_manual else ("none", "manual calibration requested but no calibration points were supplied")
    if m == "keypoint" or (m == "auto" and has_kp):
        return ("keypoint", "") if has_kp else ("none", "keypoint calibration requested but PITCH_KEYPOINT_WEIGHTS is not set")
    return "none", ("calibration disabled" if m == "none" else
                    "no calibration source: supply >=4 pitch points for a fixed camera (PUT /api/videos/:id/calibration) "
                    "or set PITCH_KEYPOINT_WEIGHTS for a pitch-landmark model")


def enabled(ctx):
    return True, ""


def run(ctx) -> None:
    cfg, conn, job_id, video = ctx.cfg, ctx.conn, ctx.job_id, ctx.video
    L, W = ctx.pitch
    execute(conn, "DELETE FROM calibration_segments WHERE job_id=%s", (job_id,))
    execute(conn, "UPDATE player_tracks SET pitch_x=NULL, pitch_y=NULL, calib_conf=NULL WHERE job_id=%s AND pitch_x IS NOT NULL", (job_id,))
    execute(conn, "UPDATE ball_tracks SET pitch_x=NULL, pitch_y=NULL, calib_conf=NULL WHERE job_id=%s AND pitch_x IS NOT NULL", (job_id,))
    mode, why = choose_mode(cfg, video)
    vid = ctx.summary["video"]
    det = ctx.summary["detection"]
    segs: list[dict] = []   # {start,end,H,conf,err,method}

    if mode == "manual":
        pts = video["calibration_points"]
        px = np.array([[p["px"], p["py"]] for p in pts], np.float32)
        pm = np.array([[p["x"], p["y"]] for p in pts], np.float32)
        fit = fit_homography(px, pm, 1.0)
        if fit is None or fit[1] > cfg.calib_max_err_m:
            mode, why = "none", f"manual calibration rejected (error {fit[1]:.2f} m > {cfg.calib_max_err_m} m or degenerate points)" if fit else "manual calibration points are degenerate"
        else:
            H, err, n = fit
            segs.append({"start": 0, "end": 2 ** 31 - 2, "H": H, "conf": _confidence(err, n, cfg.calib_max_err_m), "err": err, "method": "manual"})

    elif mode == "keypoint":
        import cv2 as _cv
        layout = load_layout(cfg.pitch_layout_path, L, W)
        cal = KeypointCalibrator(cfg, layout)
        cap = _cv.VideoCapture(video["path"])
        step, fps = vid["analysis_step"], vid["src_fps"]
        jc = ctx.job_config
        start_f = int(float(jc.get("start_s") or 0) * fps)
        end_f = int(float(jc["end_s"]) * fps) if jc.get("end_s") else int(cap.get(_cv.CAP_PROP_FRAME_COUNT))
        if start_f:
            cap.set(_cv.CAP_PROP_POS_FRAMES, start_f)
        n_expected = max(1, (end_f - start_f) // step)
        f, k, last_good, last_good_frame = start_f, 0, None, -10 ** 9
        while f < end_f:
            if not cap.grab():
                break
            if (f - start_f) % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    fit = cal.calibrate(frame)
                    if fit is not None and fit[1] <= cfg.calib_max_err_m:
                        last_good, last_good_frame = fit, f
                        H, err, n = fit
                        segs.append({"start": f, "end": f, "H": H, "conf": _confidence(err, n, cfg.calib_max_err_m), "err": err, "method": "keypoint_model"})
                    elif last_good is not None and (f - last_good_frame) <= cfg.calib_hold_frames * step:
                        H, err, n = last_good
                        segs.append({"start": f, "end": f, "H": H, "conf": 0.5 * _confidence(err, n, cfg.calib_max_err_m), "err": err, "method": "keypoint_model"})
                k += 1
                if k % 10 == 0:
                    ctx.rep.check_cancel()
                    ctx.rep.progress(NAME, min(1.0, k / n_expected), {"frame": f})
            f += 1
        cap.release()

    if not segs:
        ctx.summary["calibration"] = {"method": None, "reason": why or "no frame could be calibrated", "calibrated_frame_share": 0.0}
        ctx.warn(f"pitch coordinates unavailable: {why or 'no frame could be calibrated'}. Distance, speed, heatmaps, zones and event distances "
                 "are reported as not_available.")
        return

    copy_rows(conn, "calibration_segments", ["job_id", "start_frame", "end_frame", "method", "matrix", "confidence", "error_m"],
              [(job_id, s["start"], s["end"], s["method"], np.asarray(s["H"]).tolist(), s["conf"], s["err"]) for s in segs])

    start = np.array([s["start"] for s in segs]); end = np.array([s["end"] for s in segs])
    Hs = np.stack([np.asarray(s["H"], np.float64) for s in segs]); conf = np.array([s["conf"] for s in segs])

    tr = fetch_df(conn, "SELECT id, track_id, frame, px, py FROM player_tracks WHERE job_id=%s", (job_id,))
    x, y, c = apply_homographies(tr["px"].to_numpy(float), tr["py"].to_numpy(float), tr["frame"].to_numpy(), start, end, Hs, conf)
    # projections far outside the pitch are projection noise, not positions
    bad = (x < -10) | (x > L + 10) | (y < -10) | (y > W + 10)
    x, y, c = np.where(bad, np.nan, x), np.where(bad, np.nan, y), np.where(bad, np.nan, c)
    tr["pitch_x"], tr["pitch_y"], tr["calib_conf"] = x, y, c
    upd = tr.dropna(subset=["pitch_x"])
    bulk_update(conn, "player_tracks", "id", upd, ["pitch_x", "pitch_y", "calib_conf"],
                {"id": "BIGINT", "pitch_x": "REAL", "pitch_y": "REAL", "calib_conf": "REAL"})

    # tracks that mostly stand outside the pitch are not players in play (substitutes, staff, spectators, ball boys)
    outside = (upd["pitch_x"] < -3) | (upd["pitch_x"] > L + 3) | (upd["pitch_y"] < -3) | (upd["pitch_y"] > W + 3)
    frac = outside.groupby(upd["track_id"]).mean()
    cnt = upd.groupby("track_id").size()
    off_ids = [int(t) for t in frac.index if frac[t] >= 0.7 and cnt[t] >= 5]
    if off_ids:
        execute(conn, "UPDATE player_tracks SET role='offpitch' WHERE job_id=%s AND track_id = ANY(%s)", (job_id, off_ids))

    bl = fetch_df(conn, "SELECT id, frame, px, py FROM ball_tracks WHERE job_id=%s", (job_id,))
    if len(bl):
        bx, by, bc = apply_homographies(bl["px"].to_numpy(float), bl["py"].to_numpy(float), bl["frame"].to_numpy(), start, end, Hs, conf)
        bad = (bx < -15) | (bx > L + 15) | (by < -15) | (by > W + 15)
        bl["pitch_x"], bl["pitch_y"], bl["calib_conf"] = np.where(bad, np.nan, bx), np.where(bad, np.nan, by), np.where(bad, np.nan, bc)
        bulk_update(conn, "ball_tracks", "id", bl.dropna(subset=["pitch_x"]), ["pitch_x", "pitch_y", "calib_conf"],
                    {"id": "BIGINT", "pitch_x": "REAL", "pitch_y": "REAL", "calib_conf": "REAL"})

    cal_frames = tr.loc[tr["pitch_x"].notna(), "frame"].nunique()
    all_frames = max(1, tr["frame"].nunique())
    share = cal_frames / all_frames
    ctx.summary["calibration"] = {
        "method": segs[0]["method"], "segments": len(segs), "mean_error_m": round(float(np.mean([s["err"] for s in segs])), 3),
        "calibrated_frame_share": round(share, 3), "offpitch_tracks": len(off_ids),
    }
    if share < 0.5:
        ctx.warn(f"only {100 * share:.0f}% of frames could be calibrated: physical metrics cover that share of the footage only")
    if mode == "manual" and ctx.video.get("camera_type") != "tactical" and not ctx.video.get("static_camera"):
        ctx.warn("manual calibration assumes a fixed camera; results are wrong if the camera pans or zooms")
