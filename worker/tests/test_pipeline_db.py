"""Integration: synthetic tracker output -> every analytical stage through the real orchestrator, Postgres and Redis.
Skipped when the services are unreachable. Verifies the analytics against the generator's ground truth; it does NOT
verify detector accuracy on real footage."""
import json
import os
import uuid

import numpy as np
import pytest

from config import Config
from db import connect, copy_rows, fetch_df, fetch_one, to_json
from pipeline.orchestrator import run_job
from pipeline.teams import hex_to_lab
from tests import synth

cfg = Config.from_env()


def _services():
    try:
        connect(cfg.database_url).close()
        import redis
        redis.Redis.from_url(cfg.redis_url).ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _services(), reason="Postgres/Redis not available")

HOME_HEX, AWAY_HEX = "#D62828", "#F5F5F5"


@pytest.fixture(scope="module")
def job():
    import redis
    conn = connect(cfg.database_url, autocommit=True)
    tag = uuid.uuid4().hex[:6]
    home = fetch_one(conn, "INSERT INTO teams(name,kit_color) VALUES (%s,%s) RETURNING id", (f"Home {tag}", HOME_HEX))["id"]
    away = fetch_one(conn, "INSERT INTO teams(name,kit_color) VALUES (%s,%s) RETURNING id", (f"Away {tag}", AWAY_HEX))["id"]
    pids = {}
    for tid, num in ((6, 6), (7, 7), (9, 9), (10, 10)):
        pids[tid] = fetch_one(conn, "INSERT INTO players(team_id,name,jersey_number) VALUES (%s,%s,%s) RETURNING id", (home, f"Home #{num}", num))["id"]
    match = fetch_one(conn, "INSERT INTO matches(home_team_id,away_team_id) VALUES (%s,%s) RETURNING id", (home, away))["id"]
    tracks, balls, n = synth.build()
    video = fetch_one(conn, """INSERT INTO video_assets(match_id,path,camera_type,static_camera,fps,duration_s,width,height,frame_count,calibration_points)
                               VALUES (%s,'/nonexistent/synthetic.mp4','tactical',TRUE,30,%s,1920,1080,%s,%s::jsonb) RETURNING id""",
                      (match, n / synth.FPS, n * synth.STEP, json.dumps(synth.CALIB_POINTS)))["id"]
    summary = {"video": {"src_fps": synth.SRC, "analysis_step": synth.STEP, "analysis_fps": synth.FPS, "width": 1920, "height": 1080},
               "detection": {"analysed_frames": n, "scene_cuts": [], "class_roles": {"0": "player", "1": "goalkeeper", "2": "referee", "3": "ball"}}}
    j = fetch_one(conn, """INSERT INTO analysis_jobs(match_id,video_id,status,stages_done,summary,config)
                           VALUES (%s,%s,'queued','["detect_track"]'::jsonb,%s::jsonb,%s::jsonb) RETURNING id""",
                  (match, video, to_json(summary), to_json({"periods": [{"start_s": 0, "end_s": 60, "home_attacks": "right"}]})))["id"]
    copy_rows(conn, "player_tracks", ["job_id", "track_id", "frame", "ts", "role", "px", "py", "x1", "y1", "x2", "y2", "conf"],
              [(j, t["track_id"], t["frame"], t["ts"], t["role"], t["px"], t["py"], t["x1"], t["y1"], t["x2"], t["y2"], t["conf"]) for t in tracks])
    copy_rows(conn, "ball_tracks", ["job_id", "frame", "ts", "px", "py", "conf", "interpolated"], [(j, b["frame"], b["ts"], b["px"], b["py"], b["conf"], False) for b in balls])
    rng = np.random.default_rng(3)
    votes = {6: "6", 7: "7", 9: "9", 10: "10"}
    rows = []
    for tid, (team, role, *_r) in synth.LAYOUT.items():
        lab = None
        if team:
            lab = (hex_to_lab(HOME_HEX if team == "home" else AWAY_HEX) + rng.normal(0, 2, 3)).tolist()
        rows.append((j, tid, role, 0, (n - 1) * synth.STEP, n, lab, None, {votes[tid]: {"w": 2.7, "n": 3}} if tid in votes else None))
    copy_rows(conn, "track_features", ["job_id", "track_id", "role", "first_frame", "last_frame", "n_obs", "colour", "appearance", "jersey_votes"], rows)

    r = redis.Redis.from_url(cfg.redis_url, decode_responses=True)
    run_job(cfg, r, str(j))
    yield {"conn": conn, "job": str(j), "match": str(match), "pids": {k: str(v) for k, v in pids.items()}, "home": home, "away": away}
    if not os.getenv("INSIGHTPLAY_KEEP_TEST_DATA"):
        conn.execute("DELETE FROM matches WHERE id=%s", (match,))
        conn.execute("DELETE FROM teams WHERE id = ANY(%s)", ([home, away],))
    else:
        print("KEPT match", match)


def test_job_completed_with_expected_stage_outcomes(job):
    d = fetch_one(job["conn"], "SELECT * FROM analysis_jobs WHERE id=%s", (job["job"],))
    assert d["status"] == "completed", d["error"]
    s = d["summary"]
    assert s["stages_skipped"].keys() == {"pose", "ai"}
    assert s["teams"]["team_mapping"] == "from_kit_colour" and s["teams"]["home_tracks"] == 10 and s["teams"]["away_tracks"] == 10
    assert s["calibration"]["method"] == "manual" and s["calibration"]["mean_error_m"] < 0.05 and s["calibration"]["calibrated_frame_share"] == 1.0
    assert s["directions"][0]["source"] == "user" and s["directions"][0]["home_attacks"] == "right"


def test_kits_goalkeepers_and_identity(job):
    c = job["conn"]
    teams = fetch_df(c, "SELECT track_id, mode() WITHIN GROUP (ORDER BY team) AS team FROM player_tracks WHERE job_id=%s GROUP BY 1", (job["job"],)).set_index("track_id")["team"]
    assert all(teams[t] == "home" for t in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)) and all(teams[t] == "away" for t in (22, 23, 24, 25, 26, 27, 28, 29, 30, 31))
    assert teams[1] == "home" and teams[21] == "away"                          # keepers placed by the end they defend + attack direction
    ids = fetch_df(c, "SELECT track_id, player_id, status FROM identity_candidates WHERE job_id=%s ORDER BY track_id", (job["job"],))
    got = {int(r.track_id): (str(r.player_id), r.status) for r in ids.itertuples() if r.player_id}
    assert got == {t: (p, "auto") for t, p in job["pids"].items()}
    pending = ids[ids["status"] == "unmatched"]
    assert len(pending) >= 10                                                  # unread jerseys are flagged for a human, not guessed


def test_physical_distance_matches_ground_truth(job):
    c = job["conn"]
    for tid in (6, 10):
        pts = np.array([synth.pos(tid, i) for i in range(230)])
        truth = float(np.hypot(*np.diff(pts, axis=0).T).sum())
        row = fetch_one(c, "SELECT metrics FROM entity_physical WHERE job_id=%s AND entity_key=%s", (job["job"], f"player:{job['pids'][tid]}"))
        got = row["metrics"]["distance_m"]["value"]
        assert abs(got - truth) / truth < 0.08, (tid, got, truth)
        assert row["metrics"]["distance_m"]["coverage_pct"] > 90 and row["metrics"]["top_speed_kmh"]["value"] < 15


def test_events_and_statistics(job):
    c = job["conn"]
    ev = fetch_df(c, "SELECT type, subtype, track_id, target_track_id, status FROM events WHERE job_id=%s", (job["job"],))
    assert ((ev["type"] == "pass") & (ev["subtype"] == "completed")).sum() >= 4
    shot = ev[ev["type"] == "shot"]
    assert len(shot) == 1 and shot.iloc[0]["subtype"] == "saved" and int(shot.iloc[0]["track_id"]) == 10
    p = fetch_one(c, "SELECT metrics, attributes, track_ids, identity_status FROM player_match_stats WHERE job_id=%s AND entity_key=%s", (job["job"], f"player:{job['pids'][6]}"))
    m = p["metrics"]
    assert m["passes_attempted"]["value"] >= 1 and m["pass_accuracy_pct"]["value"] == 100.0 and p["identity_status"] == "auto"
    assert m["rating"]["value"] is None and "insufficient_observation" in m["rating"]["reason"]      # 23 s of footage: no rating rather than a fake one
    assert m["aerial_duels"]["status"] == "not_available"
    assert p["attributes"]["SHO"]["value"] is None and p["attributes"]["SHO"]["reason"] == "no_shots_observed"
    ph = fetch_one(c, "SELECT metrics FROM player_match_stats WHERE job_id=%s AND entity_key=%s", (job["job"], f"player:{job['pids'][10]}"))["metrics"]
    assert ph["shots"]["value"] == 1 and ph["shots_on_target"]["value"] == 1 and 0 < ph["xg"]["value"] < 0.6 and ph["key_passes"]["value"] == 0
    home = fetch_one(c, "SELECT metrics, tactics FROM team_match_stats WHERE job_id=%s AND team='home'", (job["job"],))
    away = fetch_one(c, "SELECT metrics FROM team_match_stats WHERE job_id=%s AND team='away'", (job["job"],))
    assert abs(home["metrics"]["possession_pct"]["value"] + away["metrics"]["possession_pct"]["value"] - 100) < 0.2
    assert home["metrics"]["goals_detected"]["value"] == 0 and home["metrics"]["corners"]["status"] == "not_available"
    assert home["tactics"]["shape"]["width_m"] > 20


def test_recompute_from_statistics_is_idempotent_and_keeps_user_decisions(job):
    import redis
    c = job["conn"]
    before = fetch_one(c, "SELECT count(*) AS n FROM player_match_stats WHERE job_id=%s", (job["job"],))["n"]
    tid = 11                                                                    # a user assigns an unread track to a squad player
    c.execute("""INSERT INTO identity_candidates(job_id,track_id,team,player_id,confidence,status) VALUES (%s,%s,'home',%s,1,'confirmed')
                 ON CONFLICT (job_id,track_id) DO UPDATE SET player_id=EXCLUDED.player_id, status='confirmed', confidence=1""", (job["job"], tid, job["pids"][10]))
    c.execute("UPDATE player_tracks SET player_id=%s WHERE job_id=%s AND track_id=%s", (job["pids"][10], job["job"], tid))
    c.execute("UPDATE analysis_jobs SET status='recomputing', config = config || '{\"recompute\":{\"from_stage\":\"identity\",\"skip_ai\":true}}'::jsonb WHERE id=%s", (job["job"],))
    run_job(cfg, redis.Redis.from_url(cfg.redis_url, decode_responses=True), job["job"])
    d = fetch_one(c, "SELECT status, error FROM analysis_jobs WHERE id=%s", (job["job"],))
    assert d["status"] == "completed", d["error"]
    keep = fetch_one(c, "SELECT player_id, status FROM identity_candidates WHERE job_id=%s AND track_id=%s", (job["job"], tid))
    assert str(keep["player_id"]) == job["pids"][10] and keep["status"] == "confirmed"
    merged = fetch_one(c, "SELECT track_ids FROM player_match_stats WHERE job_id=%s AND entity_key=%s", (job["job"], f"player:{job['pids'][10]}"))
    assert 10 in merged["track_ids"] and 11 in merged["track_ids"]           # two fragments of one player are combined into one entity
    assert fetch_one(c, "SELECT count(*) AS n FROM player_match_stats WHERE job_id=%s", (job["job"],))["n"] >= before - 1
