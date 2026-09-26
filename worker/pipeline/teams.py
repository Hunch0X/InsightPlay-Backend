"""Stage 3 — team classification from kit colour (unsupervised).

Tracks are clustered into two kits; tracks far from both centroids are left unaffiliated (goalkeepers, referees, staff).
The clusters are mapped to home/away by comparing centroids to the teams' kit_color if both are set; otherwise the
mapping is a guess and the job is flagged so the user can swap it (POST /api/jobs/:id/swap-teams)."""
from __future__ import annotations
import cv2
import numpy as np
from sklearn.cluster import KMeans

from db import execute, fetch_df

NAME = "teams"


def hex_to_lab(hexcolor: str) -> np.ndarray:
    h = hexcolor.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def cluster_kits(labs: np.ndarray, weights: np.ndarray, outlier_factor: float):
    """Two-pass k-means. Returns (labels with -1 for outliers, distances, centroids)."""
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(labs, sample_weight=weights)
    d = np.linalg.norm(labs - km.cluster_centers_[km.labels_], axis=1)
    inlier = d <= max(outlier_factor * np.median(d), 12.0)
    if inlier.sum() >= 4 and (~inlier).any():
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(labs[inlier], sample_weight=weights[inlier])
    cent = km.cluster_centers_
    dist_all = np.linalg.norm(labs[:, None, :] - cent[None, :, :], axis=2)
    lab = dist_all.argmin(axis=1)
    dmin = dist_all.min(axis=1)
    med = np.median(dmin[inlier]) if inlier.any() else np.median(dmin)
    out = dmin > max(outlier_factor * med, 12.0)
    lab = np.where(out, -1, lab)
    return lab, dmin, cent


def map_clusters(cent: np.ndarray, home_hex: str | None, away_hex: str | None):
    """Returns ({cluster: 'home'|'away'}, confirmed?)."""
    if home_hex and away_hex:
        h, a = hex_to_lab(home_hex), hex_to_lab(away_hex)
        straight = np.linalg.norm(cent[0] - h) + np.linalg.norm(cent[1] - a)
        crossed = np.linalg.norm(cent[0] - a) + np.linalg.norm(cent[1] - h)
        return ({0: "home", 1: "away"} if straight <= crossed else {0: "away", 1: "home"}), True
    return {0: "home", 1: "away"}, False


def run(ctx) -> None:
    cfg, conn, job_id = ctx.cfg, ctx.conn, ctx.job_id
    execute(conn, "UPDATE player_tracks SET team=NULL WHERE job_id=%s", (job_id,))
    f = fetch_df(conn, "SELECT track_id, role, n_obs, colour FROM track_features WHERE job_id=%s AND role='player' AND colour IS NOT NULL AND n_obs>=%s",
                 (job_id, cfg.min_track_obs))
    if len(f) < 6:
        ctx.warn("too few well-observed player tracks to classify teams; team-level statistics will be unavailable")
        ctx.summary["teams"] = {"classified": False}
        return
    labs = np.array(f["colour"].tolist(), dtype=np.float32)
    w = np.minimum(f["n_obs"].to_numpy(dtype=float), 300.0)
    labels, dist, cent = cluster_kits(labs, w, cfg.team_outlier_factor)
    sep = float(np.linalg.norm(cent[0] - cent[1]))
    if sep < 15:
        ctx.warn(f"the two kits are very similar in colour (separation {sep:.0f}); team classification is unreliable")
    mapping, confirmed = map_clusters(cent, (ctx.home or {}).get("kit_color"), (ctx.away or {}).get("kit_color"))
    side = {int(t): mapping.get(int(l)) for t, l in zip(f["track_id"], labels) if l >= 0}
    for s in ("home", "away"):
        ids = [t for t, v in side.items() if v == s]
        if ids:
            execute(conn, "UPDATE player_tracks SET team=%s WHERE job_id=%s AND track_id = ANY(%s)", (s, job_id, ids))
    with conn.cursor() as cur:
        cur.executemany("UPDATE track_features SET cluster=%s, cluster_dist=%s WHERE job_id=%s AND track_id=%s",
                        [(int(l), float(d), job_id, int(t)) for t, l, d in zip(f["track_id"], labels, dist)])
    n_home, n_away = sum(v == "home" for v in side.values()), sum(v == "away" for v in side.values())
    ctx.summary["teams"] = {
        "classified": True, "kit_separation": round(sep, 1), "home_tracks": n_home, "away_tracks": n_away,
        "unaffiliated_tracks": int((labels < 0).sum()),
        "team_mapping": "from_kit_colour" if confirmed else "unconfirmed",
    }
    if not confirmed:
        ctx.summary["review"] = {**(ctx.summary.get("review") or {}), "team_mapping": "Kit clusters were assigned to home/away without kit colours; "
                                 "verify and use POST /api/jobs/:id/swap-teams if reversed."}
        ctx.warn("teams were assigned to home/away without kit colours; verify the mapping (swap-teams endpoint)")
