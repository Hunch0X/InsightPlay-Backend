"""Stage — physical metrics from pitch coordinates: distance, speed, sprints, accelerations, heatmap, zones, path.

Everything here needs calibrated positions. If the video could not be calibrated the stage records that and produces
nothing — it never substitutes pixel distances or invented numbers. Every metric carries its coverage."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from db import bulk_update, execute, fetch_df, to_json
from pipeline.geometry import attack_sign

NAME = "physical"
MIN_SAMPLES = 20          # ignore fragments shorter than ~2 s at 10 fps
MAX_ACCEL = 10.0          # m/s^2 — anything above is tracking noise, not a human


def enabled(ctx):
    return True, ""


def metric(value, unit, coverage=None, confidence=None, source="derived", **extra):
    m = {"value": None if value is None else (round(float(value), 3) if isinstance(value, (float, np.floating)) else value),
         "unit": unit, "source": source, "confidence": None if confidence is None else round(float(confidence), 2),
         "coverage_pct": None if coverage is None else round(float(coverage), 1)}
    m.update(extra)
    return m


def runs(mask: np.ndarray, ts: np.ndarray, min_dur: float) -> list[tuple[int, int]]:
    """Index ranges [i0, i1] where mask stays True for at least min_dur seconds. Pure function, unit-tested."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if ts[j] - ts[i] >= min_dur:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def analyse_series(ts: np.ndarray, x: np.ndarray, y: np.ndarray, cfg, max_speed: float | None = None, win: int | None = None):
    """Smooth a position series, split it at tracking gaps, derive velocity/speed/acceleration.
    Returns dict of arrays (same length as input) + step arrays. Pure function, unit-tested."""
    max_speed = max_speed or cfg.max_speed_ms
    win = win or cfg.smooth_window
    n = len(ts)
    xs, ys = x.astype(float).copy(), y.astype(float).copy()
    vx, vy, sp, ac = (np.full(n, np.nan) for _ in range(4))
    seg = np.zeros(n, int)
    breaks = np.where(np.diff(ts) > cfg.max_gap_s)[0] + 1
    bounds = np.concatenate([[0], breaks, [n]])
    for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg[a:b] = s
        m = b - a
        if m < 5:
            continue
        w = min(win, m if m % 2 == 1 else m - 1)
        if w >= 5:
            xs[a:b], ys[a:b] = savgol_filter(x[a:b], w, 2), savgol_filter(y[a:b], w, 2)
        t = ts[a:b]
        vx[a:b], vy[a:b] = np.gradient(xs[a:b], t), np.gradient(ys[a:b], t)
        sp[a:b] = np.hypot(vx[a:b], vy[a:b])
    valid = np.isfinite(sp) & (sp <= max_speed)
    sp_v = np.where(valid, sp, np.nan)
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= 5:
            seg_sp = np.where(np.isfinite(sp_v[a:b]), sp_v[a:b], 0.0)
            ac[a:b] = np.gradient(seg_sp, ts[a:b])
    ac = np.where(np.abs(ac) <= MAX_ACCEL, ac, np.nan)
    # step i -> i+1 (only inside a segment, only when both endpoints are valid)
    dt = np.zeros(n)
    step = np.zeros(n)
    same = np.r_[seg[1:] == seg[:-1], False]
    ok = same & np.r_[valid[1:], False] & valid
    dt[ok] = (np.r_[ts[1:], ts[-1]] - ts)[ok]
    step[ok] = np.hypot(np.r_[xs[1:], xs[-1]] - xs, np.r_[ys[1:], ys[-1]] - ys)[ok]
    return {"xs": xs, "ys": ys, "vx": vx, "vy": vy, "speed": sp_v, "accel": ac, "valid": valid, "dt": dt, "step": step, "seg": seg}


def entity_metrics(ts, res, cfg, calib_conf: float, active_span_s: float, poss_in: np.ndarray | None):
    valid, sp, ac, dt, step = res["valid"], res["speed"], res["accel"], res["dt"], res["step"]
    observed = float(dt.sum())
    coverage = 100.0 * min(1.0, observed / active_span_s) if active_span_s > 0 else 0.0
    conf = float(np.clip(calib_conf, 0, 1) * min(1.0, coverage / 80.0))
    hsr_ms, spr_ms = cfg.hsr_kmh / 3.6, cfg.sprint_kmh / 3.6
    spv = np.where(valid, sp, 0.0)
    sprint_runs = runs(spv >= spr_ms, ts, cfg.min_sprint_s)
    acc_runs = runs(np.nan_to_num(ac, nan=0.0) >= cfg.accel_thr, ts, cfg.min_accel_s)
    dec_runs = runs(np.nan_to_num(ac, nan=0.0) <= -cfg.accel_thr, ts, cfg.min_accel_s)
    dist = float(step.sum())
    mk = lambda v, u, **kw: metric(v, u, coverage, conf, **kw)
    m = {
        "distance_km": mk(dist / 1000.0, "km"),
        "distance_m": mk(dist, "m"),
        "distance_per_min_m": mk(dist / (observed / 60.0) if observed > 0 else None, "m/min"),
        "top_speed_kmh": mk(float(np.nanmax(sp)) * 3.6 if np.isfinite(sp).any() else None, "km/h"),
        "avg_speed_kmh": mk(float(np.nanmean(sp)) * 3.6 if np.isfinite(sp).any() else None, "km/h"),
        "high_speed_distance_m": mk(float(step[spv >= hsr_ms].sum()), "m"),
        "sprint_distance_m": mk(float(step[spv >= spr_ms].sum()), "m"),
        "sprints": mk(len(sprint_runs), "count"),
        "accelerations": mk(len(acc_runs), "count"),
        "decelerations": mk(len(dec_runs), "count"),
        "minutes_observed": mk(observed / 60.0, "min"),
    }
    if poss_in is not None:
        m["distance_in_possession_m"] = mk(float(step[poss_in == 1].sum()), "m")
        m["distance_out_of_possession_m"] = mk(float(step[poss_in == 0].sum()), "m")
    return m


def attack_signs(ts: np.ndarray, team: str | None, directions: list[dict]) -> np.ndarray:
    """+1/-1 per sample; 0 where the direction is unknown."""
    out = np.zeros(len(ts), int)
    for d in directions:
        s = attack_sign(team, d.get("home_attacks"))
        if s:
            out[(ts >= d["start_s"]) & (ts < d["end_s"])] = s
    return out


def heatmap(x, y, w, L, W, nx, ny):
    h, _, _ = np.histogram2d(y, x, bins=[ny, nx], range=[[0, W], [0, L]], weights=w)
    return {"grid": {"nx": nx, "ny": ny, "length_m": L, "width_m": W}, "seconds": np.round(h, 1).tolist()}


def run(ctx) -> None:
    cfg, conn, job_id = ctx.cfg, ctx.conn, ctx.job_id
    L, W = ctx.pitch
    execute(conn, "DELETE FROM entity_physical WHERE job_id=%s", (job_id,))
    cal = ctx.summary.get("calibration") or {}
    if not cal.get("method"):
        ctx.summary["physical"] = {"available": False, "reason": "camera_not_calibrated"}
        return
    tr = fetch_df(conn, """SELECT id, track_id, frame, ts, role, team, player_id, pitch_x, pitch_y, calib_conf, conf FROM player_tracks
                           WHERE job_id=%s AND pitch_x IS NOT NULL AND role IN ('player','goalkeeper')""", (job_id,))
    if tr.empty:
        ctx.summary["physical"] = {"available": False, "reason": "no_calibrated_positions"}
        return
    directions = ctx.summary.get("directions") or []
    tr["entity"] = np.where(tr["player_id"].notna(), "player:" + tr["player_id"].astype(str), "track:" + tr["track_id"].astype(str))

    # who has the ball, over time (for in/out-of-possession distance)
    poss = fetch_df(conn, "SELECT ts, possession_track FROM ball_tracks WHERE job_id=%s AND possession_track IS NOT NULL ORDER BY ts", (job_id,))
    track_team = tr.groupby("track_id")["team"].agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None)
    pt = pteam = None
    if not poss.empty:
        poss["team"] = poss["possession_track"].map(track_team)
        poss = poss.dropna(subset=["team"])
        if not poss.empty:
            pt, pteam = poss["ts"].to_numpy(float), poss["team"].to_numpy()

    upd_parts, rows, n_ent = [], [], 0
    for ent, df in tr.groupby("entity"):
        df = df.sort_values(["ts", "conf"], ascending=[True, False]).drop_duplicates("frame")
        if len(df) < MIN_SAMPLES:
            continue
        ts, x, y = df["ts"].to_numpy(float), df["pitch_x"].to_numpy(float), df["pitch_y"].to_numpy(float)
        team = df["team"].dropna().mode().iloc[0] if df["team"].notna().any() else None
        res = analyse_series(ts, x, y, cfg)
        poss_in = None
        if pt is not None and team in ("home", "away"):
            idx = np.searchsorted(pt, ts, side="right") - 1
            known = (idx >= 0) & (ts - pt[np.clip(idx, 0, None)] <= 3.0)
            poss_in = np.where(known, (pteam[np.clip(idx, 0, None)] == team).astype(int), -1)
        span = float(ts[-1] - ts[0]) or 1e-9
        metrics = entity_metrics(ts, res, cfg, float(df["calib_conf"].mean()), span, poss_in)

        dtw = np.diff(ts, append=ts[-1])
        dtw = np.where((dtw > cfg.max_gap_s) | (dtw < 0), 0.0, dtw)
        heat = heatmap(res["xs"], res["ys"], dtw, L, W, cfg.heatmap_nx, cfg.heatmap_ny)
        avg = {"x": round(float(np.mean(res["xs"])), 2), "y": round(float(np.mean(res["ys"])), 2), "orientation": "raw_pitch"}
        zone = None
        signs = attack_signs(ts, team, directions)
        known = signs != 0
        if known.mean() >= 0.5:
            nx_ = np.where(signs[known] > 0, res["xs"][known], L - res["xs"][known])
            w_ = dtw[known]
            zone = {"defensive_third": round(float(w_[nx_ < L / 3].sum()), 1), "middle_third": round(float(w_[(nx_ >= L / 3) & (nx_ < 2 * L / 3)].sum()), 1),
                    "attacking_third": round(float(w_[nx_ >= 2 * L / 3].sum()), 1), "orientation": "attacking_left_to_right",
                    "coverage_pct": round(100 * float(known.mean()), 1)}
            heat["attack_normalised"] = heatmap(nx_, res["ys"][known], w_, L, W, cfg.heatmap_nx, cfg.heatmap_ny)["seconds"]
            avg["attack_normalised"] = {"x": round(float(nx_.mean()), 2), "y": round(float(res["ys"][known].mean()), 2)}
        keep = np.linspace(0, len(ts) - 1, min(len(ts), cfg.path_points)).astype(int)
        path = {"points": [[round(float(res["xs"][i]), 1), round(float(res["ys"][i]), 1), round(float(ts[i]), 1)] for i in keep]}
        rows.append((job_id, ent, to_json(metrics), to_json(heat), to_json(avg), to_json(zone) if zone else None, to_json(path)))
        upd_parts.append(pd.DataFrame({"id": df["id"].to_numpy(), "vx": res["vx"], "vy": res["vy"], "speed": res["speed"], "accel": res["accel"]}))
        n_ent += 1

    with conn.cursor() as cur:
        cur.executemany("INSERT INTO entity_physical(job_id,entity_key,metrics,heatmap,avg_position,zone_time,movement_path) VALUES (%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb)", rows)
    if upd_parts:
        u = pd.concat(upd_parts)
        bulk_update(conn, "player_tracks", "id", u, ["vx", "vy", "speed", "accel"], {"id": "BIGINT", "vx": "REAL", "vy": "REAL", "speed": "REAL", "accel": "REAL"})

    # ball kinematics (faster object: lighter smoothing, higher speed cap)
    b = fetch_df(conn, "SELECT id, ts, pitch_x, pitch_y FROM ball_tracks WHERE job_id=%s AND pitch_x IS NOT NULL ORDER BY ts", (job_id,))
    if len(b) >= 10:
        r = analyse_series(b["ts"].to_numpy(float), b["pitch_x"].to_numpy(float), b["pitch_y"].to_numpy(float), cfg, max_speed=45.0, win=5)
        bulk_update(conn, "ball_tracks", "id", pd.DataFrame({"id": b["id"], "vx": r["vx"], "vy": r["vy"], "speed": r["speed"]}), ["vx", "vy", "speed"],
                    {"id": "BIGINT", "vx": "REAL", "vy": "REAL", "speed": "REAL"})
    ctx.summary["physical"] = {"available": True, "entities": n_ent, "calibrated_frame_share": cal.get("calibrated_frame_share")}
