"""Stage — aggregate events + physical data into player and team statistics.

Every metric is an object {value, unit, source, confidence, coverage_pct, ...}. A metric that cannot be computed is
{value: null, status: "not_available", reason} — never 0, never a placeholder. Ratings and PAC/SHO/PAS/DRI/DEF/PHY are
transparent formulas over measured values (documented below), not model outputs."""
from __future__ import annotations
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from db import execute, fetch_df, to_json
from pipeline.geometry import attack_sign, attack_x, home_attacks_at
from pipeline.physical import metric

NAME = "statistics"
RATING_VERSION = "rating_v1"
ATTR_VERSION = "attr_v1"
MIN_ENTITY_S = 20.0


def enabled(ctx):
    return True, ""


def na(reason: str) -> dict:
    return {"value": None, "status": "not_available", "reason": reason}


def scale(v, lo, hi, out_lo=40.0, out_hi=99.0) -> int:
    return int(round(out_lo + (out_hi - out_lo) * float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))))


# ------------------------------------------------------------------------------------------- entity aggregation
def count_events(ev: pd.DataFrame, tracks: set[int]) -> dict:
    """Raw counts for one entity (set of track ids). Pure function, unit-tested."""
    mine = ev[ev["track_id"].isin(tracks)]
    tgt = ev[ev["target_track_id"].isin(tracks)]
    c = lambda t, s=None: int(((mine["type"] == t) & ((mine["subtype"] == s) if s else True)).sum())
    passes = mine[mine["type"] == "pass"]
    attr = lambda df, k: df["attributes"].map(lambda a: bool((a or {}).get(k)))
    shots = mine[mine["type"] == "shot"]
    return {
        "passes_attempted": int(len(passes)), "passes_completed": int((passes["subtype"] == "completed").sum()),
        "progressive_passes": int((attr(passes, "progressive") & (passes["subtype"] == "completed")).sum()),
        "final_third_passes": int((attr(passes, "final_third") & (passes["subtype"] == "completed")).sum()),
        "key_passes": c("key_pass"), "assists": c("assist"), "goals": c("goal"),
        "shots": int(len(shots)), "shots_on_target": int(attr(shots, "on_target").sum()),
        "xg": float(sum((a or {}).get("xg", 0.0) for a in shots["attributes"])),
        "carries": c("carry"), "progressive_carries": c("carry", "progressive"),
        "dribbles_successful": c("dribble", "successful"), "dribbles_failed": c("dribble", "failed"),
        "tackles_won": c("tackle"), "interceptions": c("interception"), "recoveries": c("recovery"),
        "clearances": c("clearance"), "blocks": c("block"), "saves": c("save"), "pressures": c("pressure"),
        "duels_won": c("duel", "won"), "duels_lost": int(((tgt["type"] == "duel") & (tgt["subtype"] == "won")).sum()),
        "_mean_conf": float(mine["confidence"].mean()) if len(mine) else None,
        "_n_events": int(len(mine)),
    }


def compute_rating(m: dict, minutes: float, dist_km: float | None, phys_cov: float | None):
    """rating_v1: 6.0 baseline + weighted contributions of measured actions. An explainable heuristic — NOT calibrated
    against outcomes or expert ratings. Returns (rating, contributions) or (None, reason)."""
    if minutes < 10:
        return None, "insufficient_observation (<10 min)"
    parts = {
        "attacking": 0.8 * m["goals"] + 0.5 * m["assists"] + 0.15 * m["key_passes"] + 0.10 * m["shots_on_target"]
                     + 0.08 * m["dribbles_successful"] - 0.05 * m["dribbles_failed"] + 0.05 * m["progressive_carries"],
        "passing": 0.04 * m["progressive_passes"] - 0.02 * (m["passes_attempted"] - m["passes_completed"]),
        "defending": 0.10 * m["tackles_won"] + 0.10 * m["interceptions"] + 0.04 * m["recoveries"] + 0.05 * m["blocks"]
                     + 0.15 * m["saves"] + 0.03 * m["clearances"] + 0.05 * m["duels_won"] - 0.03 * m["duels_lost"],
    }
    if m["passes_attempted"] >= 10:
        acc = m["passes_completed"] / m["passes_attempted"]
        parts["passing"] += (acc - 0.75) * 2.0 * min(1.0, m["passes_attempted"] / 30.0)
    if dist_km is not None and phys_cov is not None and phys_cov >= 70:
        parts["physical"] = float(np.clip((dist_km * 90.0 / minutes - 9.0) * 0.1, -0.4, 0.4))
    r = float(np.clip(6.0 + sum(parts.values()), 3.0, 10.0))
    return round(r, 1), {k: round(v, 2) for k, v in parts.items()}


def attributes(m: dict, phys: dict | None, minutes: float) -> dict:
    """PAC/SHO/PAS/DRI/DEF/PHY on a 0-99 scale from fixed reference ranges (attr_v1). Each is null when its inputs are missing."""
    def obj(v, basis, conf=0.5):
        return {"value": v, "unit": "0-99", "source": "derived", "confidence": conf, "method": ATTR_VERSION, "basis": basis}
    out = {}
    pv = lambda k: ((phys or {}).get(k) or {}).get("value")
    pc = lambda k: ((phys or {}).get(k) or {}).get("coverage_pct")
    if phys is None:
        out["PAC"] = out["PHY"] = na("camera_not_calibrated")
    else:
        top, dist, obs, spr = pv("top_speed_kmh"), pv("distance_km"), pv("minutes_observed"), pv("sprints")
        if not obs or obs < 10:
            out["PAC"] = na("insufficient_observation (<10 min): top speed over a short clip says nothing about pace")
        elif top is not None and (pc("top_speed_kmh") or 0) >= 50:
            out["PAC"] = obj(scale(top, 20, 34), {"top_speed_kmh": top})
        else:
            out["PAC"] = na("insufficient_physical_coverage")
        if dist is not None and obs and obs >= 10 and (pc("distance_km") or 0) >= 60:
            per90, spr90 = dist * 90 / obs, (spr or 0) * 90 / obs
            x = 0.7 * np.clip((per90 - 6) / 6, 0, 1) + 0.3 * np.clip(spr90 / 25, 0, 1)
            out["PHY"] = obj(scale(x, 0, 1), {"distance_km_per90": round(per90, 1), "sprints_per90": round(spr90, 1)})
        else:
            out["PHY"] = na("insufficient_physical_coverage")
    if m["shots"] >= 2:
        x = 0.3 * m["shots_on_target"] / m["shots"] + 0.7 * min(1.0, (m["xg"] / m["shots"]) / 0.20)
        out["SHO"] = obj(scale(x, 0, 1), {"shots": m["shots"], "on_target": m["shots_on_target"], "xg_per_shot": round(m["xg"] / m["shots"], 3)}, 0.3)
    else:
        out["SHO"] = na("no_shots_observed" if m["shots"] == 0 else "fewer_than_2_shots_observed")
    if m["passes_attempted"] >= 8:
        acc, prog = m["passes_completed"] / m["passes_attempted"], m["progressive_passes"] / m["passes_attempted"]
        x = 0.7 * np.clip((acc - 0.5) / 0.45, 0, 1) + 0.3 * np.clip(prog / 0.25, 0, 1)
        out["PAS"] = obj(scale(x, 0, 1), {"attempted": m["passes_attempted"], "accuracy": round(acc, 2), "progressive_share": round(prog, 2)})
    else:
        out["PAS"] = na("fewer_than_8_passes_observed")
    drib = m["dribbles_successful"] + m["dribbles_failed"]
    if drib >= 2 or m["carries"] >= 3:
        rate = m["dribbles_successful"] / drib if drib >= 2 else 0.5
        x = 0.6 * rate + 0.4 * min(1.0, m["progressive_carries"] / 2.0)
        out["DRI"] = obj(scale(x, 0, 1), {"dribbles": drib, "success_rate": round(rate, 2), "progressive_carries": m["progressive_carries"]}, 0.3)
    else:
        out["DRI"] = na("insufficient_dribbling_events")
    if minutes >= 15:
        acts = m["tackles_won"] + m["interceptions"] + m["recoveries"] + m["blocks"] + m["clearances"]
        out["DEF"] = obj(scale(acts * 90 / minutes / 12.0, 0, 1), {"defensive_actions_per90": round(acts * 90 / minutes, 1)}, 0.4)
    else:
        out["DEF"] = na("insufficient_observation")
    return out


# --------------------------------------------------------------------------------------------- team tactics
def estimate_formation(xs: np.ndarray) -> str | None:
    """Lines from the attack-normalised average depth of 8-10 outfield players. Estimated, low confidence."""
    xs = np.asarray(xs, float)
    if len(xs) < 8:
        return None
    best = None
    for k in (3, 4):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(xs.reshape(-1, 1))
        order = np.argsort(km.cluster_centers_.ravel())
        cents = km.cluster_centers_.ravel()[order]
        sizes = [int((km.labels_ == o).sum()) for o in order]
        if k == 4 and (np.diff(cents).min() < 9.0 or min(sizes) < 1):
            continue
        best = "-".join(map(str, sizes))
    return best


def team_shape(tr: pd.DataFrame, team: str, directions, L: float) -> dict | None:
    """Width/depth/line height from frames where >= 8 outfield players of the team are calibrated."""
    t = tr[(tr["team"] == team) & (tr["role"] == "player") & tr["pitch_x"].notna()]
    if t.empty:
        return None
    rows = []
    for f, g in t.groupby("frame"):
        if len(g) < 8:
            continue
        ts = float(g["ts"].iloc[0])
        sgn = attack_sign(team, home_attacks_at(ts, directions))
        ax = g["pitch_x"].to_numpy() if not sgn else np.array([attack_x(x, L, sgn) for x in g["pitch_x"]])
        rows.append((float(g["pitch_y"].max() - g["pitch_y"].min()), float(ax.max() - ax.min()), float(np.sort(ax)[:4].mean()) if sgn else np.nan))
    if not rows:
        return None
    a = np.array(rows)
    return {"frames": len(rows), "width_m": round(float(a[:, 0].mean()), 1), "depth_m": round(float(a[:, 1].mean()), 1),
            "defensive_line_height_m": None if np.isnan(a[:, 2]).all() else round(float(np.nanmean(a[:, 2])), 1)}


def ppda(ev: pd.DataFrame, press_team: str, L: float, directions) -> dict:
    """Opponent passes in their own 60% of the pitch per defensive action (tackle, interception, duel) by the pressing team
    in the same zone. Needs calibrated positions and a known direction."""
    opp = "away" if press_team == "home" else "home"
    def in_zone(row, team):
        x = row["sx"]
        sgn = attack_sign(team, home_attacks_at(row["ts"], directions))
        return bool(pd.notna(x) and sgn and attack_x(x, L, sgn) <= 0.6 * L)
    passes = ev[(ev["type"] == "pass") & (ev["team"] == opp)]
    acts = ev[ev["type"].isin(["tackle", "interception", "duel"]) & (ev["team"] == press_team)]
    if passes["sx"].isna().all():
        return na("camera_not_calibrated")
    zone_p = int(sum(in_zone(r, opp) for _, r in passes.iterrows()))
    # defensive actions are positioned in the pressing team's frame of reference, so test them against the opponent's zone
    zone_a = int(sum(in_zone(r, opp) for _, r in acts.iterrows() if pd.notna(r["sx"])))
    if not len(passes):
        return na("no_opponent_passes")
    if all(attack_sign(opp, home_attacks_at(t, directions)) is None for t in passes["ts"]):
        return na("attack_direction_unknown")
    if zone_a == 0:
        return na("no_defensive_actions_in_zone")
    return metric(zone_p / zone_a, "passes per action", confidence=0.4, source="estimated", passes_in_zone=zone_p, actions_in_zone=zone_a)


# ------------------------------------------------------------------------------------------------------ stage
def run(ctx) -> None:
    conn, job_id, cfg = ctx.conn, ctx.job_id, ctx.cfg
    L, W = ctx.pitch
    execute(conn, "DELETE FROM player_match_stats WHERE job_id=%s", (job_id,))
    execute(conn, "DELETE FROM team_match_stats WHERE job_id=%s", (job_id,))
    fps_an = ctx.summary["video"]["analysis_fps"]
    directions = ctx.summary.get("directions") or []
    ball_cov = 100.0 * float((ctx.summary.get("ball") or {}).get("coverage") or 0.0)

    tr = fetch_df(conn, "SELECT track_id, frame, ts, role, team, player_id, pitch_x, pitch_y FROM player_tracks WHERE job_id=%s AND role IN ('player','goalkeeper')", (job_id,))
    ev = fetch_df(conn, "SELECT type, subtype, ts, team, track_id, target_track_id, sx, sy, ex, ey, confidence, status, attributes FROM events WHERE job_id=%s AND status<>'rejected'", (job_id,))
    idc = fetch_df(conn, "SELECT track_id, player_id, confidence, status FROM identity_candidates WHERE job_id=%s", (job_id,))
    phys = {r.entity_key: r for r in fetch_df(conn, "SELECT * FROM entity_physical WHERE job_id=%s", (job_id,)).itertuples()}
    poss = ctx.summary.get("possession") or {}
    spells = {int(k): v for k, v in (poss.get("spells_by_track") or {}).items()}

    tr["entity"] = np.where(tr["player_id"].notna(), "player:" + tr["player_id"].astype(str), "track:" + tr["track_id"].astype(str))
    ent_rows, per_team = [], defaultdict(list)
    for ent, g in tr.groupby("entity"):
        observed = len(g) / fps_an
        team = g["team"].dropna().mode().iloc[0] if g["team"].notna().any() else None
        pid = g["player_id"].dropna().iloc[0] if g["player_id"].notna().any() else None
        if observed < MIN_ENTITY_S or (team is None and pid is None):
            continue
        tracks = set(int(t) for t in g["track_id"].unique())
        ts = np.sort(g["ts"].unique())
        span = float(ts[-1] - ts[0])
        gaps = np.diff(ts)
        minutes = max(0.0, (span - float(gaps[gaps > 300].sum())) / 60.0)          # half-time breaks do not count
        m = count_events(ev, tracks) if not ev.empty else count_events(pd.DataFrame(columns=["type", "subtype", "track_id", "target_track_id", "attributes", "confidence"]), tracks)
        p = phys.get(ent)
        pm = p.metrics if p is not None else None
        cov_frac = float(np.clip(observed / max(span, 1e-9), 0, 1))
        base_conf = m["_mean_conf"]
        ev_cov = ball_cov * cov_frac
        cm = lambda v, unit="count", extra=None: metric(v, unit, ev_cov, base_conf, **(extra or {}))
        metrics = {
            "minutes": metric(minutes, "min", 100 * cov_frac, 0.6, source="estimated", note="first to last observation, half-time excluded; not the official minutes played"),
            "passes_attempted": cm(m["passes_attempted"]), "passes_completed": cm(m["passes_completed"]),
            "pass_accuracy_pct": (metric(100.0 * m["passes_completed"] / m["passes_attempted"], "%", ev_cov, base_conf) if m["passes_attempted"] else na("no_passes_observed")),
            "progressive_passes": cm(m["progressive_passes"]) if ctx.summary.get("calibration", {}).get("method") and any(d.get("home_attacks") for d in directions) else na("camera_not_calibrated_or_direction_unknown"),
            "key_passes": cm(m["key_passes"]), "assists": cm(m["assists"], extra={"needs_review": True}), "goals": cm(m["goals"], extra={"needs_review": True}),
            "shots": cm(m["shots"]), "shots_on_target": cm(m["shots_on_target"], extra={"note": "on target = saved or scored; height is not observable"}),
            "xg": metric(m["xg"], "xG", ev_cov, 0.3, source="estimated", model="xg_geometric_v0") if m["shots"] else na("no_shots_observed"),
            "carries": cm(m["carries"]), "progressive_carries": cm(m["progressive_carries"]),
            "dribbles_successful": cm(m["dribbles_successful"]), "dribbles_failed": cm(m["dribbles_failed"]),
            "tackles_won": cm(m["tackles_won"]), "interceptions": cm(m["interceptions"]), "recoveries": cm(m["recoveries"]),
            "clearances": cm(m["clearances"]), "blocks": cm(m["blocks"]), "saves": cm(m["saves"]), "pressures": cm(m["pressures"]),
            "duels_won": cm(m["duels_won"]), "duels_lost": cm(m["duels_lost"]),
            "touches": metric(spells.get(next(iter(tracks)), 0) if len(tracks) == 1 else sum(spells.get(t, 0) for t in tracks), "possession spells", ev_cov, 0.4, source="estimated",
                              note="counted as possession spells; individual touches inside a dribble are not resolved"),
            "aerial_duels": na("not_detected_single_camera"),
        }
        if pm:
            metrics.update(pm)
        else:
            for k in ("distance_km", "distance_m", "top_speed_kmh", "avg_speed_kmh", "high_speed_distance_m", "sprint_distance_m", "sprints", "accelerations", "decelerations"):
                metrics[k] = na("camera_not_calibrated" if not ctx.summary.get("calibration", {}).get("method") else "insufficient_calibrated_positions")
        rating, parts = compute_rating(m, minutes, (pm or {}).get("distance_km", {}).get("value") if pm else None, (pm or {}).get("distance_km", {}).get("coverage_pct") if pm else None)
        metrics["rating"] = ({"value": rating, "unit": "1-10", "source": "derived", "confidence": 0.3, "method": RATING_VERSION, "contributions": parts,
                              "note": "transparent heuristic over measured actions; not calibrated against expert ratings"} if rating is not None else na(parts))
        ic = idc[idc["track_id"].isin(tracks)]
        status = ("confirmed" if (ic["status"] == "confirmed").any() else "auto" if (ic["status"] == "auto").any()
                  else (ic["status"].iloc[0] if len(ic) else ("unidentified" if pid is None else "auto")))
        ent_rows.append({
            "key": ent, "player_id": pid, "team": team, "role": "goalkeeper" if (g["role"] == "goalkeeper").any() else "player", "tracks": sorted(tracks),
            "id_conf": float(ic["confidence"].max()) if len(ic) else None, "id_status": status, "metrics": metrics,
            "attrs": attributes(m, pm, minutes), "raw": m, "phys": p, "minutes": minutes,
        })
        if team in ("home", "away"):
            per_team[team].append(ent_rows[-1])

    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO player_match_stats(job_id,entity_key,player_id,team,role,track_ids,identity_confidence,identity_status,metrics,attributes,heatmap,avg_position,zone_time,movement_path)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb)""",
            [(job_id, r["key"], r["player_id"], r["team"], r["role"], r["tracks"], r["id_conf"], r["id_status"], to_json(r["metrics"]), to_json(r["attrs"]),
              to_json(r["phys"].heatmap) if r["phys"] is not None and r["phys"].heatmap else None,
              to_json(r["phys"].avg_position) if r["phys"] is not None and r["phys"].avg_position else None,
              to_json(r["phys"].zone_time) if r["phys"] is not None and r["phys"].zone_time else None,
              to_json(r["phys"].movement_path) if r["phys"] is not None and r["phys"].movement_path else None) for r in ent_rows])

    # ---- teams
    tt = poss.get("time_by_team_s") or {}
    total_ctrl = sum(tt.values())
    for side in ("home", "away"):
        rows = per_team.get(side, [])
        opp = "away" if side == "home" else "home"
        tev = ev[ev["team"] == side] if not ev.empty else ev
        n = lambda t, s=None: int(((tev["type"] == t) & ((tev["subtype"] == s) if s else True)).sum()) if not ev.empty else 0
        passes = tev[tev["type"] == "pass"] if not ev.empty else tev
        pa, pc = len(passes), int((passes["subtype"] == "completed").sum()) if len(passes) else 0
        shots = tev[tev["type"] == "shot"] if not ev.empty else tev
        cm = lambda v, unit="count", conf=None, **kw: metric(v, unit, ball_cov, conf if conf is not None else (float(tev["confidence"].mean()) if len(tev) else None), **kw)
        dist = [r["metrics"].get("distance_km", {}).get("value") for r in rows]
        dist = [d for d in dist if d is not None]
        metrics = {
            "possession_pct": metric(100.0 * tt.get(side, 0.0) / total_ctrl, "%", ball_cov, 0.5, source="estimated", note="share of ball-controlled time") if total_ctrl > 0 else na("ball_not_tracked"),
            "passes": cm(pa), "pass_accuracy_pct": metric(100.0 * pc / pa, "%", ball_cov, 0.5) if pa else na("no_passes_observed"),
            "shots": cm(int(len(shots))), "shots_on_target": cm(int(sum(bool((a or {}).get("on_target")) for a in shots["attributes"])) if len(shots) else 0),
            "xg": metric(float(sum((a or {}).get("xg", 0.0) for a in shots["attributes"])), "xG", ball_cov, 0.3, source="estimated", model="xg_geometric_v0") if len(shots) else na("no_shots_observed"),
            "goals_detected": metric(n("goal"), "count", ball_cov, 0.5, needs_review=True),
            "tackles": cm(n("tackle")), "interceptions": cm(n("interception")), "recoveries": cm(n("recovery")), "clearances": cm(n("clearance")),
            "pressures": cm(n("pressure")), "blocks": cm(n("block")), "carries": cm(n("carry")), "dribbles_successful": cm(n("dribble", "successful")),
            "distance_km": metric(sum(dist), "km", None, 0.5) if dist else na("camera_not_calibrated"),
            "sprints": metric(sum((r["metrics"].get("sprints", {}) or {}).get("value") or 0 for r in rows), "count", None, 0.5) if dist else na("camera_not_calibrated"),
            "ppda": ppda(ev, side, L, directions) if not ev.empty else na("no_events"),
            "corners": na("not_detected"),
        }
        outfield = sorted([r for r in rows if r["role"] == "player" and r["phys"] is not None and r["phys"].avg_position and "attack_normalised" in r["phys"].avg_position],
                          key=lambda r: -(r["phys"].metrics.get("minutes_observed", {}).get("value") or 0))[:10]
        formation = estimate_formation(np.array([r["phys"].avg_position["attack_normalised"]["x"] for r in outfield])) if outfield else None
        edges = defaultdict(int)
        if not ev.empty:
            ent_of = {t: r["key"] for r in rows for t in r["tracks"]}
            for e in passes[passes["subtype"] == "completed"].itertuples():
                a, b = ent_of.get(e.track_id), ent_of.get(e.target_track_id)
                if a and b and a != b:
                    edges[(a, b)] += 1
        tactics = {
            "formation": ({"value": formation, "source": "estimated", "confidence": 0.3, "basis_players": len(outfield)} if formation else na("direction_unknown_or_too_few_tracked_players")),
            "shape": team_shape(tr, side, directions, L) or na("too_few_players_visible_together"),
            "possession_by_third_s": (poss.get("by_third_s") or {}).get(side) or na("direction_unknown"),
            "average_positions": [{"entity": r["key"], "player_id": r["player_id"], **r["phys"].avg_position} for r in rows if r["phys"] is not None and r["phys"].avg_position],
            "passing_network": [{"from": a, "to": b, "passes": c} for (a, b), c in sorted(edges.items(), key=lambda kv: -kv[1])[:60]],
        }
        execute(conn, "INSERT INTO team_match_stats(job_id,team,metrics,tactics) VALUES (%s,%s,%s::jsonb,%s::jsonb)", (job_id, side, to_json(metrics), to_json(tactics)))
    ctx.summary["statistics"] = {"entities": len(ent_rows), "rating_version": RATING_VERSION, "attributes_version": ATTR_VERSION}
