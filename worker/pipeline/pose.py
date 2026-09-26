"""Stage (optional, POSE_ENABLED=true) — body pose on ball-proximal player crops.

Pose is only computed where it can matter (a player within ~1.5 player-heights of the ball), not for every player in
every frame. It yields orientation, torso lean and a 'kick_like' swing flag used to corroborate pass/shot starts."""
from __future__ import annotations
import cv2
import numpy as np

from db import copy_rows, execute, fetch_df, to_json

NAME = "pose"
L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 5, 6, 11, 12, 15, 16


def enabled(ctx):
    return (True, "") if ctx.cfg.pose_enabled else (False, "pose disabled (POSE_ENABLED=false)")


def pose_features(kp: np.ndarray, conf: np.ndarray, box_h: float) -> dict | None:
    """Derive features from COCO-17 keypoints (crop coordinates). Pure function, unit-tested."""
    ok = lambda i: conf[i] >= 0.3
    out = {}
    if ok(L_SH) and ok(R_SH):
        d = kp[R_SH] - kp[L_SH]
        out["shoulder_angle_deg"] = float(np.degrees(np.arctan2(d[1], d[0])))
    if ok(L_SH) and ok(R_SH) and ok(L_HIP) and ok(R_HIP):
        sh, hp = (kp[L_SH] + kp[R_SH]) / 2, (kp[L_HIP] + kp[R_HIP]) / 2
        v = sh - hp
        out["torso_lean_deg"] = float(np.degrees(np.arctan2(v[0], -v[1])))
    swings = []
    for hip, ank in ((L_HIP, L_ANK), (R_HIP, R_ANK)):
        if ok(hip) and ok(ank) and box_h > 0:
            swings.append(abs(float(kp[ank][0] - kp[hip][0])) / box_h)
    if swings:
        out["leg_swing"] = max(swings)
        out["kick_like"] = bool(max(swings) > 0.45)
    return out or None


def run(ctx) -> None:
    from ultralytics import YOLO

    cfg, conn, job_id = ctx.cfg, ctx.conn, ctx.job_id
    execute(conn, "DELETE FROM pose_features WHERE job_id=%s", (job_id,))
    near = fetch_df(conn, """SELECT b.frame, b.nearest_track AS track_id, t.x1, t.y1, t.x2, t.y2
                             FROM ball_tracks b JOIN player_tracks t ON t.job_id=b.job_id AND t.frame=b.frame AND t.track_id=b.nearest_track
                             WHERE b.job_id=%s AND b.nearest_dist <= 1.5 ORDER BY b.frame LIMIT %s""", (job_id, cfg.pose_max_crops))
    if near.empty:
        ctx.summary["pose"] = {"crops": 0}
        return
    model = YOLO(cfg.pose_weights)
    vid = ctx.summary["video"]
    want = {int(f): g for f, g in near.groupby("frame")}
    cap = cv2.VideoCapture(ctx.video["path"])
    step, fps = vid["analysis_step"], vid["src_fps"]
    start_f = int(float(ctx.job_config.get("start_s") or 0) * fps)
    if start_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    last = max(want)
    f, rows, n = start_f, [], 0
    while f <= last:
        if not cap.grab():
            break
        if f in want:
            ok, frame = cap.retrieve()
            if ok:
                H, W = frame.shape[:2]
                for r in want[f].itertuples():
                    h, w = r.y2 - r.y1, r.x2 - r.x1
                    x1, y1, x2, y2 = int(max(0, r.x1 - 0.15 * w)), int(max(0, r.y1 - 0.1 * h)), int(min(W, r.x2 + 0.15 * w)), int(min(H, r.y2 + 0.05 * h))
                    if x2 - x1 < 16 or y2 - y1 < 32:
                        continue
                    res = model.predict(frame[y1:y2, x1:x2], verbose=False, conf=0.25, device=cfg.device or None)[0]
                    if res.keypoints is None or len(res.boxes) == 0:
                        continue
                    i = int(res.boxes.conf.argmax())
                    feat = pose_features(res.keypoints.xy[i].cpu().numpy(), res.keypoints.conf[i].cpu().numpy(), float(y2 - y1))
                    if feat:
                        rows.append((job_id, int(r.track_id), int(f), to_json(feat)))
                        n += 1
            if n % 50 == 0:
                ctx.rep.check_cancel()
                ctx.rep.progress(NAME, min(1.0, f / max(1, last)))
        f += 1
    cap.release()
    copy_rows(conn, "pose_features", ["job_id", "track_id", "frame", "features"], rows)
    ctx.summary["pose"] = {"crops": n}
