"""Stage — map tracks to squad players.

Evidence: jersey-number OCR votes aggregated per track, matched against the team's roster. A deep re-ID embedding is used
only if the configured encoder is discriminative (colour histograms are not — same-team players share kit colours).
Low-confidence or conflicting matches are NOT applied: they become 'needs_confirmation' for a human.
Decisions already made by a user (confirmed / rejected) are preserved across recomputes."""
from __future__ import annotations
import numpy as np

from db import execute, fetch_df
from pipeline.appearance import make_encoder

NAME = "identity"


def vote_result(votes: dict | None):
    """votes {"22": {"w": 2.7, "n": 3}, ...} -> (number, confidence, evidence) or None. Pure function, unit-tested.
    confidence = vote share x support (reads/3, capped) x mean OCR confidence: a single read can never reach auto-accept."""
    if not votes:
        return None
    total = sum(v["w"] for v in votes.values())
    if total <= 0:
        return None
    num, best = max(votes.items(), key=lambda kv: kv[1]["w"])
    share, support, mean_conf = best["w"] / total, min(1.0, best["n"] / 3.0), best["w"] / best["n"]
    return int(num), float(share * support * mean_conf), {"reads": int(best["n"]), "share": round(share, 2), "mean_ocr_conf": round(mean_conf, 2),
                                                          "all_votes": {k: v["n"] for k, v in votes.items()}}


def cosine(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def enabled(ctx):
    return True, ""


def run(ctx) -> None:
    cfg, conn, job_id = ctx.cfg, ctx.conn, ctx.job_id
    decided = fetch_df(conn, "SELECT track_id, player_id, status FROM identity_candidates WHERE job_id=%s AND status IN ('confirmed','rejected')", (job_id,))
    execute(conn, "DELETE FROM identity_candidates WHERE job_id=%s AND status NOT IN ('confirmed','rejected')", (job_id,))
    execute(conn, "UPDATE player_tracks SET player_id=NULL WHERE job_id=%s AND player_id IS NOT NULL", (job_id,))
    with conn.cursor() as cur:
        cur.executemany("UPDATE player_tracks SET player_id=%s WHERE job_id=%s AND track_id=%s",
                        [(r.player_id, job_id, int(r.track_id)) for r in decided.itertuples() if r.status == "confirmed" and r.player_id])
    decided_ids = set(int(t) for t in decided["track_id"])

    tr = fetch_df(conn, """SELECT track_id, mode() WITHIN GROUP (ORDER BY team) AS team, count(*) AS n, min(frame) AS f0, max(frame) AS f1
                           FROM player_tracks WHERE job_id=%s AND role IN ('player','goalkeeper') GROUP BY track_id""", (job_id,))
    feats = fetch_df(conn, "SELECT track_id, jersey_votes, appearance FROM track_features WHERE job_id=%s", (job_id,)).set_index("track_id")
    roster = {"home": {}, "away": {}}
    for side, team in (("home", ctx.home), ("away", ctx.away)):
        if team:
            for p in fetch_df(conn, "SELECT id, name, jersey_number, reference_embedding FROM players WHERE team_id=%s", (team["id"],)).itertuples():
                if p.jersey_number is not None:
                    roster[side][int(p.jersey_number)] = p
    deep = make_encoder(cfg).is_discriminative if cfg.reid_encoder != "color" else False

    cands = []   # dicts
    for r in tr.itertuples():
        tid = int(r.track_id)
        if tid in decided_ids or r.team not in ("home", "away") or tid not in feats.index:
            continue
        votes = feats.at[tid, "jersey_votes"]
        vr = vote_result(votes)
        c = {"track_id": tid, "team": r.team, "player_id": None, "jersey": None, "ocr": None, "sim": None, "conf": 0.0, "status": None, "ev": {}, "n": int(r.n), "f0": int(r.f0), "f1": int(r.f1)}
        if vr:
            num, conf, ev = vr
            c["jersey"], c["ocr"], c["ev"] = num, conf, ev
            player = roster[r.team].get(num)
            if player is None:
                c["status"], c["ev"]["reason"] = "unmatched", "jersey_not_in_roster"
            else:
                c["player_id"], c["conf"] = str(player.id), conf
                emb, ref = feats.at[tid, "appearance"], player.reference_embedding
                if deep and emb is not None and ref is not None:
                    c["sim"] = cosine(emb, ref)
                    c["conf"] = 0.75 * conf + 0.25 * max(0.0, c["sim"])
                c["status"] = "auto" if c["conf"] >= cfg.identity_threshold else "needs_confirmation"
        elif r.n >= 100:
            c["status"], c["ev"]["reason"] = "unmatched", "no_jersey_reading"
        else:
            continue
        cands.append(c)

    # two auto tracks claiming the same player at the same time cannot both be right
    by_player: dict[str, list[dict]] = {}
    for c in cands:
        if c["status"] == "auto":
            by_player.setdefault(c["player_id"], []).append(c)
    for pid, lst in by_player.items():
        lst.sort(key=lambda c: -c["conf"])
        for i, c in enumerate(lst):
            if any(min(c["f1"], k["f1"]) - max(c["f0"], k["f0"]) > 0.1 * min(c["f1"] - c["f0"], k["f1"] - k["f0"] + 1) for k in lst[:i]):
                c["status"] = "needs_confirmation"
                c["ev"]["reason"] = "overlaps_another_track_matched_to_same_player"

    from db import to_json
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO identity_candidates(job_id,track_id,team,player_id,jersey_number,ocr_conf,appearance_sim,confidence,status,evidence)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT (job_id,track_id) DO NOTHING""",
            [(job_id, c["track_id"], c["team"], c["player_id"], c["jersey"], c["ocr"], c["sim"], c["conf"], c["status"], to_json(c["ev"])) for c in cands])
        cur.executemany("UPDATE player_tracks SET player_id=%s WHERE job_id=%s AND track_id=%s",
                        [(c["player_id"], job_id, c["track_id"]) for c in cands if c["status"] == "auto"])
    n = {s: sum(c["status"] == s for c in cands) for s in ("auto", "needs_confirmation", "unmatched")}
    ctx.summary["identity"] = {**n, "confirmed_by_user": int((decided["status"] == "confirmed").sum()), "tracks_considered": len(tr)}
    pending = n["needs_confirmation"] + n["unmatched"]
    if pending:
        ctx.summary["review"] = {**(ctx.summary.get("review") or {}), "identity": f"{pending} track(s) need identity confirmation (GET /api/jobs/:id/identity/pending)"}
    if ctx.cfg.jersey_ocr == "none":
        ctx.warn("no jersey OCR: players stay 'Unidentified' until confirmed by a user")
