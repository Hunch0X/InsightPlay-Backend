"""Stage 1 — read the video, detect players / goalkeepers / referees / ball with YOLO, track with BoT-SORT.

Also collects, per track, the evidence later stages need: kit colour, appearance descriptor, jersey-number OCR votes.
Camera cuts (broadcast footage) reset the tracker; track ids stay globally unique across cuts.
"""
from __future__ import annotations
import json
import logging
from collections import Counter

import cv2
import numpy as np

from db import copy_rows, execute, to_json
from pipeline.appearance import kit_colour, make_encoder, make_ocr, torso_crop

log = logging.getLogger("detect_track")
NAME = "detect_track"

ALIASES = {
    "player": "player", "person": "player", "goalkeeper": "goalkeeper", "goalie": "goalkeeper", "keeper": "goalkeeper",
    "referee": "referee", "ref": "referee", "ball": "ball", "sports ball": "ball", "football": "ball", "soccer ball": "ball",
}
PLAYER_ROLES = ("player", "goalkeeper", "referee")


def build_roles(names: dict, override: str = "") -> dict[int, str]:
    """Map the model's class ids to InsightPlay roles. Custom football weights work as long as their class names
    are recognisable (or DETECTOR_CLASS_MAP is set)."""
    if override:
        roles = {int(k): v for k, v in json.loads(override).items()}
    else:
        roles = {int(i): ALIASES[str(n).strip().lower()] for i, n in names.items() if str(n).strip().lower() in ALIASES}
    if "player" not in roles.values():
        raise RuntimeError(f"Detector classes {list(names.values())[:10]}... contain no player/person class. Set DETECTOR_CLASS_MAP.")
    return roles


def reset_tracker(model) -> bool:
    pred = getattr(model, "predictor", None)
    ok = False
    for t in getattr(pred, "trackers", None) or []:
        try:
            t.reset()
            ok = True
        except Exception:  # older Ultralytics builds have no reset()
            pass
    return ok


class TrackAcc:
    """Evidence accumulated for one track."""
    __slots__ = ("roles", "first", "last", "n", "labs", "feats", "votes", "last_ocr_ts", "last_feat_ts", "last_enc_ts")

    def __init__(self, frame):
        self.roles, self.first, self.last, self.n = Counter(), frame, frame, 0
        self.labs, self.feats, self.votes = [], [], {}
        self.last_ocr_ts, self.last_feat_ts, self.last_enc_ts = -1e9, -1e9, -1e9


def _hist_dist(a, b) -> float:
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def _frame_signature(frame) -> np.ndarray:
    small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).astype(np.float32)
    cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L1)
    return h


def _ball_from_model(model, frame, cfg, tiling: bool) -> tuple[float, float, float, float, float] | None:
    """Dedicated ball detection (optional model, optional tiling). Returns x1,y1,x2,y2,conf of the best candidate."""
    H, W = frame.shape[:2]
    tiles = [(0, 0, W, H)]
    if tiling:
        tw, th = int(W * 0.6), int(H * 0.6)
        tiles = [(x, y, min(W, x + tw), min(H, y + th)) for x in (0, W - tw) for y in (0, H - th)]
    best = None
    for (tx1, ty1, tx2, ty2) in tiles:
        res = model.predict(frame[ty1:ty2, tx1:tx2], conf=cfg.ball_conf, imgsz=cfg.imgsz, verbose=False, device=cfg.device or None)[0]
        for b in res.boxes:
            c = float(b.conf)
            if best is None or c > best[4]:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                best = (x1 + tx1, y1 + ty1, x2 + tx1, y2 + ty1, c)
    return best


def run(ctx) -> None:
    from ultralytics import YOLO  # imported lazily: heavy

    cfg, conn, job_id = ctx.cfg, ctx.conn, ctx.job_id
    video = ctx.video
    for t in ("player_tracks", "ball_tracks", "track_features", "calibration_segments", "pose_features", "identity_candidates", "events",
              "entity_physical", "player_match_stats", "team_match_stats", "ai_analyses"):
        execute(conn, f"DELETE FROM {t} WHERE job_id=%s", (job_id,))

    cap = cv2.VideoCapture(video["path"])
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video['path']}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or video.get("fps") or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or video.get("frame_count") or 0)
    step = max(1, int(round(src_fps / cfg.analysis_fps)))
    jc = ctx.job_config
    start_f = int(float(jc.get("start_s") or 0) * src_fps)
    end_f = int(float(jc["end_s"]) * src_fps) if jc.get("end_s") else total
    end_f = min(end_f, total) if total else end_f
    if start_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    n_expected = max(1, (end_f - start_f) // step)

    model = YOLO(cfg.detector_weights)
    roles = build_roles(model.names, cfg.detector_class_map)
    ball_model = YOLO(cfg.ball_weights) if cfg.ball_weights else None
    encoder = make_encoder(cfg)
    ocr = make_ocr(cfg)
    dev = cfg.device or None
    call_conf = min(cfg.det_conf, cfg.ball_conf)

    ctx.summary["video"] = {"src_fps": src_fps, "analysis_step": step, "analysis_fps": round(src_fps / step, 3),
                            "width": video.get("width"), "height": video.get("height")}
    ctx.set_model_versions({"detector": cfg.detector_weights, "ball_detector": cfg.ball_weights or "same as detector",
                            "tracker": "botsort", "appearance": encoder.name, "jersey_ocr": cfg.jersey_ocr})

    accs: dict[int, TrackAcc] = {}
    prow, brow = [], []
    cuts: list[int] = []
    id_offset, max_gid = 0, 0
    prev_sig = None
    frame_idx, n_done, n_player_dets = start_f, 0, 0

    def flush():
        nonlocal prow, brow
        if prow:
            copy_rows(conn, "player_tracks", ["job_id", "track_id", "frame", "ts", "role", "px", "py", "x1", "y1", "x2", "y2", "conf"], prow)
            prow = []
        if brow:
            copy_rows(conn, "ball_tracks", ["job_id", "frame", "ts", "px", "py", "conf", "interpolated"], brow)
            brow = []

    while frame_idx < end_f:
        ok = cap.grab()
        if not ok:
            break
        if (frame_idx - start_f) % step != 0:
            frame_idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            frame_idx += 1
            continue
        ts = frame_idx / src_fps

        sig = _frame_signature(frame)
        if prev_sig is not None and _hist_dist(prev_sig, sig) > cfg.cut_threshold:
            cuts.append(frame_idx)
            if reset_tracker(model):
                id_offset = max_gid
        prev_sig = sig

        res = model.track(frame, persist=True, tracker=cfg.tracker_cfg, conf=call_conf, imgsz=cfg.imgsz, verbose=False, device=dev)[0]
        ball_best = None
        boxes = res.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(clss)
            for (x1, y1, x2, y2), c, k, tid in zip(xyxy, confs, clss, ids):
                role = roles.get(int(k))
                if role is None:
                    continue
                if role == "ball":
                    if ball_best is None or c > ball_best[4]:
                        ball_best = (float(x1), float(y1), float(x2), float(y2), float(c))
                    continue
                if tid is None or c < cfg.det_conf:
                    continue
                gid = int(tid) + id_offset
                max_gid = max(max_gid, gid)
                px, py = (x1 + x2) / 2.0, float(y2)
                prow.append((job_id, gid, frame_idx, ts, role, px, py, float(x1), float(y1), float(x2), float(y2), float(c)))
                n_player_dets += 1
                acc = accs.get(gid)
                if acc is None:
                    acc = accs[gid] = TrackAcc(frame_idx)
                acc.roles[role] += 1
                acc.last, acc.n = frame_idx, acc.n + 1
                h = y2 - y1
                if role != "referee" and h >= 40:
                    box = (float(x1), float(y1), float(x2), float(y2))
                    if len(acc.labs) < 40 and (ts - acc.last_feat_ts) >= 0.5:
                        lab = kit_colour(torso_crop(frame, box))
                        if lab is not None:
                            acc.labs.append(lab)
                        if len(acc.feats) < 8 and (ts - acc.last_enc_ts) >= 1.5:
                            f = encoder.encode(frame, box)
                            acc.last_enc_ts = ts
                            if f is not None:
                                acc.feats.append(f)
                        acc.last_feat_ts = ts
                    if ocr is not None and h >= cfg.ocr_min_h and len(acc.votes.get("_n", [])) < cfg.ocr_max_samples \
                            and (ts - acc.last_ocr_ts) >= cfg.ocr_min_gap_s:
                        acc.last_ocr_ts = ts
                        acc.votes.setdefault("_n", []).append(ts)
                        for num, conf in ocr.read(frame, box):
                            v = acc.votes.setdefault(str(num), {"w": 0.0, "n": 0})
                            v["w"] += conf
                            v["n"] += 1

        if ball_model is not None:
            ball_best = _ball_from_model(ball_model, frame, cfg, cfg.ball_tiling) or ball_best
        if ball_best is not None:
            bx1, by1, bx2, by2, bc = ball_best
            brow.append((job_id, frame_idx, ts, (bx1 + bx2) / 2.0, (by1 + by2) / 2.0, bc, False))

        n_done += 1
        if len(prow) >= 5000:
            flush()
        if n_done % 5 == 0:
            ctx.rep.check_cancel()
            ctx.rep.progress(NAME, min(1.0, n_done / n_expected), {"frame": frame_idx, "tracks": len(accs)})
        frame_idx += 1

    cap.release()
    flush()
    if n_done == 0 or n_player_dets == 0:
        raise RuntimeError("no players were detected in the analysed frames — check the video, DETECTOR_WEIGHTS and class names")

    feat_rows = []
    for gid, a in accs.items():
        role = a.roles.most_common(1)[0][0]
        lab = np.median(np.stack(a.labs), axis=0).tolist() if a.labs else None
        emb = None
        if a.feats:
            m = np.mean(np.stack(a.feats), axis=0)
            n = np.linalg.norm(m)
            emb = (m / n if n > 0 else m).tolist()
        votes = {k: v for k, v in a.votes.items() if k != "_n"}
        feat_rows.append((job_id, gid, role, a.first, a.last, a.n, lab, emb, votes or None))
    copy_rows(conn, "track_features", ["job_id", "track_id", "role", "first_frame", "last_frame", "n_obs", "colour", "appearance", "jersey_votes"], feat_rows)
    # one role per track: rows keep the track's majority role so goalkeepers do not flicker
    for gid, a in accs.items():
        if len(a.roles) > 1:
            maj = a.roles.most_common(1)[0][0]
            execute(conn, "UPDATE player_tracks SET role=%s WHERE job_id=%s AND track_id=%s AND role<>%s", (maj, job_id, gid, maj))
    n_ball = int(conn.execute("SELECT count(*) FROM ball_tracks WHERE job_id=%s", (job_id,)).fetchone()[0])
    ctx.summary["detection"] = {
        "analysed_frames": n_done, "tracks": len(accs), "player_detections": n_player_dets,
        "ball_detected_frames": n_ball, "ball_detection_rate": round(n_ball / n_done, 3),
        "scene_cuts": cuts[:5000], "n_scene_cuts": len(cuts),
        "class_roles": {str(k): v for k, v in roles.items()},
    }
    if n_ball / n_done < 0.15:
        ctx.warn(f"ball was detected in only {100 * n_ball / n_done:.0f}% of frames — passes/shots will be sparse. "
                 "Use football-trained weights (DETECTOR_WEIGHTS / BALL_WEIGHTS) and consider BALL_TILING=true.")
    if cuts:
        ctx.warn(f"{len(cuts)} camera cuts detected; replays are not detected automatically and may be analysed as live play.")
    if ocr is None:
        ctx.warn("jersey OCR is disabled (JERSEY_OCR=none): players will not be identified automatically.")
