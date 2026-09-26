"""Stage — attacking direction per period, and goalkeeper/referee resolution.

Direction is needed for progressive actions, thirds, final-third entries, shots and PPDA. Sources, in order:
  1. 'user'      : periods[].home_attacks supplied in the analyze request
  2. 'heuristic' : the team whose outfield players' mean x is lower defends the low-x goal (needs a few minutes of footage)
If neither is possible the direction is None and direction-dependent metrics are reported as not_available."""
from __future__ import annotations
import numpy as np

from db import execute, fetch_df
from pipeline.geometry import attack_sign

NAME = "direction"


def infer_period_direction(home_x: np.ndarray, away_x: np.ndarray, min_rows: int = 200):
    """Returns (home_attacks, confidence) or (None, 0). Pure function, unit-tested."""
    if len(home_x) < min_rows or len(away_x) < min_rows:
        return None, 0.0
    diff = float(np.mean(away_x) - np.mean(home_x))
    if abs(diff) < 3.0:
        return None, 0.0
    return ("right" if diff > 0 else "left"), float(min(1.0, abs(diff) / 8.0) * 0.8)


def is_goalkeeper_position(mean_x: float, mean_y: float, L: float, W: float) -> bool:
    return (mean_x < 16.5 or mean_x > L - 16.5) and abs(mean_y - W / 2) < 20


def enabled(ctx):
    return True, ""


def run(ctx) -> None:
    conn, job_id, cfg = ctx.conn, ctx.job_id, ctx.cfg
    L, W = ctx.pitch
    periods = ctx.periods()
    tr = fetch_df(conn, "SELECT track_id, role, team, ts, pitch_x, pitch_y FROM player_tracks WHERE job_id=%s AND pitch_x IS NOT NULL AND role IN ('player','goalkeeper')", (job_id,))
    directions = []
    if tr.empty:
        for p in periods:
            directions.append({**p, "home_attacks": p.get("home_attacks"), "source": "user" if p.get("home_attacks") else None, "confidence": 1.0 if p.get("home_attacks") else 0.0})
        ctx.summary["directions"] = directions
        if any(d["home_attacks"] is None for d in directions):
            ctx.warn("attack direction unknown (no calibrated positions): direction-dependent metrics are not_available")
        return

    outfield = tr[(tr["role"] == "player") & tr["team"].isin(["home", "away"])]
    for p in periods:
        win = outfield[(outfield["ts"] >= p["start_s"]) & (outfield["ts"] < p["end_s"])]
        if p.get("home_attacks") in ("left", "right"):
            directions.append({**p, "source": "user", "confidence": 1.0})
            continue
        ha, conf = infer_period_direction(win.loc[win["team"] == "home", "pitch_x"].to_numpy(), win.loc[win["team"] == "away", "pitch_x"].to_numpy())
        directions.append({**p, "home_attacks": ha, "source": "heuristic" if ha else None, "confidence": round(conf, 2)})
    dur = ctx.video.get("duration_s") or 0
    if len(periods) == 1 and not ctx.job_config.get("periods") and dur > 55 * 60:
        ctx.warn("video is longer than 55 minutes and no periods were supplied: direction is assumed constant, which is wrong after half-time. "
                 "Pass periods[] with home_attacks in the analyze request.")
    if any(d["home_attacks"] is None for d in directions):
        ctx.warn("attack direction could not be determined for at least one period; progressive/final-third/shot/PPDA metrics are not_available for it")
    elif any(d["source"] == "heuristic" for d in directions):
        ctx.warn("attack direction was inferred from team positions (heuristic); supply periods[].home_attacks to make it exact")

    # ---- goalkeepers (detector class, or inferred from kit outlier + position) and referees
    has_ref_class = "referee" in ctx.summary.get("detection", {}).get("class_roles", {}).values()
    feats = fetch_df(conn, "SELECT track_id, cluster, n_obs FROM track_features WHERE job_id=%s", (job_id,))
    clus = dict(zip(feats["track_id"], feats["cluster"]))
    nobs = dict(zip(feats["track_id"], feats["n_obs"]))
    all_tr = fetch_df(conn, "SELECT track_id, role, team, ts, pitch_x, pitch_y FROM player_tracks WHERE job_id=%s AND pitch_x IS NOT NULL AND role IN ('player','goalkeeper','referee')", (job_id,))
    g = all_tr.groupby("track_id")
    n_gk = n_ref = 0
    gk_updates: list[tuple[str | None, str, int]] = []
    for tid, rows in g:
        role = rows["role"].iloc[0]
        team = rows["team"].iloc[0] if rows["team"].notna().any() else None
        inferred = None
        if role == "player" and team is None and clus.get(tid, 0) == -1 and nobs.get(tid, 0) >= cfg.min_track_obs:
            if is_goalkeeper_position(float(rows["pitch_x"].mean()), float(rows["pitch_y"].mean()), L, W):
                inferred = "goalkeeper"
            elif not has_ref_class:
                inferred = "referee"
        if role == "goalkeeper" or inferred == "goalkeeper":
            # team = the side defending the end this keeper stands at
            pi = None
            best = -1
            for i, d in enumerate(directions):
                m = ((rows["ts"] >= d["start_s"]) & (rows["ts"] < d["end_s"])).sum()
                if m > best:
                    best, pi = m, i
            ha = directions[pi]["home_attacks"] if pi is not None else None
            gk_team = None
            if ha:
                defends_low_x = float(rows["pitch_x"].mean()) < L / 2
                home_defends_low = (ha == "right")
                gk_team = "home" if defends_low_x == home_defends_low else "away"
            gk_updates.append((gk_team, "goalkeeper", int(tid)))
            n_gk += 1 if inferred else 0
        elif inferred == "referee":
            gk_updates.append((None, "referee", int(tid)))
            n_ref += 1
    with conn.cursor() as cur:
        cur.executemany("UPDATE player_tracks SET team=%s, role=%s WHERE job_id=%s AND track_id=%s", [(t, r, job_id, i) for t, r, i in gk_updates])
    if n_gk:
        ctx.warn(f"{n_gk} goalkeeper track(s) were inferred from kit colour outliers standing near a goal; a detector with a goalkeeper class is more reliable")
    if n_ref:
        ctx.warn(f"{n_ref} referee track(s) were inferred from kit colour outliers (detector has no referee class)")
    ctx.summary["directions"] = directions
