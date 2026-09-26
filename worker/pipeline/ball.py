"""Stage 2 — ball tracking, independent of player tracking.

Cleans raw ball detections (outlier rejection, short-gap interpolation flagged as interpolated), then links the ball to
players: nearest player, and the player judged to be controlling it (distance measured in player heights, so it needs
no calibration)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from db import bulk_update, copy_rows, execute, fetch_df

NAME = "ball"


def clean_and_interpolate(b: pd.DataFrame, step: int, width: float, cuts: list[int], cfg) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Returns (kept raw rows, interpolated rows, ids of rejected outliers) — pure function, unit-tested."""
    b = b.sort_values("frame").reset_index(drop=True)
    if len(b) < 5:
        return b, pd.DataFrame(columns=b.columns), []
    med_x = b["px"].rolling(5, center=True, min_periods=3).median()
    med_y = b["py"].rolling(5, center=True, min_periods=3).median()
    resid = np.hypot(b["px"] - med_x, b["py"] - med_y)
    keep = resid <= cfg.ball_outlier_frac * width
    kept = b[keep].reset_index(drop=True)
    rejected_ids = b.loc[~keep, "id"].tolist()

    cut_arr = np.asarray(sorted(cuts), dtype=int)
    rows = []
    fr, x, y, ts = kept["frame"].to_numpy(), kept["px"].to_numpy(), kept["py"].to_numpy(), kept["ts"].to_numpy()
    for i in range(len(kept) - 1):
        gap = int(round((fr[i + 1] - fr[i]) / step))
        if gap <= 1 or gap - 1 > cfg.ball_max_gap_frames:
            continue
        if len(cut_arr) and np.any((cut_arr > fr[i]) & (cut_arr <= fr[i + 1])):
            continue
        for k in range(1, gap):
            a = k / gap
            rows.append({"frame": int(fr[i] + k * step), "ts": float(ts[i] + a * (ts[i + 1] - ts[i])),
                         "px": float(x[i] + a * (x[i + 1] - x[i])), "py": float(y[i] + a * (y[i + 1] - y[i]))})
    return kept, pd.DataFrame(rows), rejected_ids


def link_to_players(ball: pd.DataFrame, tracks: pd.DataFrame, control_dist: float) -> pd.DataFrame:
    """Nearest player (in player-heights) and control candidate for every ball row — pure function, unit-tested."""
    t = tracks.assign(h=(tracks["y2"] - tracks["y1"]).clip(lower=8.0))
    m = ball[["frame", "px", "py"]].rename(columns={"px": "bx", "py": "by"}).merge(
        t[["frame", "track_id", "px", "py", "h"]], on="frame", how="inner")
    m["d"] = np.hypot(m["bx"] - m["px"], m["by"] - m["py"]) / m["h"]
    best = m.sort_values("d").groupby("frame", sort=False).first().reset_index()[["frame", "track_id", "d"]]
    out = ball[["id", "frame"]].merge(best, on="frame", how="left")
    out["nearest_track"] = out["track_id"].astype("Int64")
    out["nearest_dist"] = out["d"]
    out["possession_track"] = out["track_id"].where(out["d"] <= control_dist).astype("Int64")
    return out[["id", "nearest_track", "nearest_dist", "possession_track"]]


def run(ctx) -> None:
    cfg, conn, job_id = ctx.cfg, ctx.conn, ctx.job_id
    vid = ctx.summary["video"]
    det = ctx.summary["detection"]
    raw = fetch_df(conn, "SELECT id, frame, ts, px, py, conf FROM ball_tracks WHERE job_id=%s ORDER BY frame", (job_id,))
    n_raw = len(raw)
    if n_raw >= 5:
        kept, interp, rejected = clean_and_interpolate(raw, vid["analysis_step"], float(vid["width"] or 1920), det.get("scene_cuts", []), cfg)
        if rejected:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM ball_tracks WHERE id = ANY(%s)", (rejected,))
        if len(interp):
            copy_rows(conn, "ball_tracks", ["job_id", "frame", "ts", "px", "py", "conf", "interpolated"],
                      [(job_id, r.frame, r.ts, r.px, r.py, None, True) for r in interp.itertuples()])
    else:
        rejected, interp = [], pd.DataFrame()
        ctx.warn("fewer than 5 ball detections: no ball tracking is possible for this video")

    ball = fetch_df(conn, "SELECT id, frame, px, py FROM ball_tracks WHERE job_id=%s ORDER BY frame", (job_id,))
    if len(ball):
        tracks = fetch_df(conn, "SELECT frame, track_id, px, py, x1, y1, x2, y2 FROM player_tracks WHERE job_id=%s AND role IN ('player','goalkeeper')", (job_id,))
        link = link_to_players(ball, tracks, cfg.control_dist)
        for c in ("nearest_track", "possession_track"):
            link[c] = link[c].astype(object).where(link[c].notna(), None)
        bulk_update(conn, "ball_tracks", "id", link, ["nearest_track", "nearest_dist", "possession_track"],
                    {"id": "BIGINT", "nearest_track": "INT", "nearest_dist": "REAL", "possession_track": "INT"})
    n_interp = len(interp)
    ctx.summary["ball"] = {"raw_detections": n_raw, "outliers_removed": len(rejected), "interpolated": n_interp,
                           "frames_with_ball": n_raw - len(rejected) + n_interp,
                           "coverage": round((n_raw - len(rejected) + n_interp) / max(1, det["analysed_frames"]), 3)}
