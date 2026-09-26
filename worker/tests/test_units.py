import numpy as np
import pandas as pd
import cv2

from config import Config
from pipeline import ball, calibration, direction, geometry, identity, physical, teams, statistics, ai

cfg = Config()


# ---------------------------------------------------------------- geometry
def test_xg_geometric_documented_calibration_points():
    L, W = 105, 68
    assert 0.02 < geometry.xg_geometric(80, 34, L, W, 1) < 0.06            # ~25 m central
    assert 0.15 < geometry.xg_geometric(94, 34, L, W, 1) < 0.30            # penalty spot
    assert geometry.xg_geometric(99, 34, L, W, 1) > 0.55                    # ~6 m
    assert geometry.xg_geometric(80, 34, L, W, 1) < geometry.xg_geometric(90, 34, L, W, 1)
    assert geometry.xg_geometric(25, 34, L, W, -1) == geometry.xg_geometric(80, 34, L, W, 1)   # mirrored


def test_attack_sign_and_direction_lookup():
    assert geometry.attack_sign("home", "right") == 1 and geometry.attack_sign("away", "right") == -1
    assert geometry.attack_sign("away", "left") == 1 and geometry.attack_sign(None, "left") is None
    dirs = [{"start_s": 0, "end_s": 100, "home_attacks": "right"}, {"start_s": 100, "end_s": 200, "home_attacks": "left"}]
    assert geometry.home_attacks_at(150, dirs) == "left" and geometry.home_attacks_at(500, dirs) is None


# ---------------------------------------------------------------- calibration
def test_homography_recovers_known_mapping_and_rejects_collinear():
    H_true = np.array([[17, 2, 40], [1, 14, 60], [0.0001, 0.0002, 1]], float)
    pitch = np.array([[0, 0], [105, 0], [105, 68], [0, 68], [52.5, 34], [16.5, 13.8]], np.float32)
    px = cv2.perspectiveTransform(pitch[None].astype(np.float64), H_true)[0]
    H, err, n = calibration.fit_homography(px, pitch)
    assert err < 1e-3 and n == 6
    x, y, c = calibration.apply_homographies(px[:, 0], px[:, 1], np.zeros(6, int), np.array([0]), np.array([10]), H[None], np.array([0.9]))
    assert np.allclose(np.c_[x, y], pitch, atol=1e-2)
    line = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], np.float32)
    assert calibration.fit_homography(line * 10, line) is None


def test_apply_homographies_outside_segment_is_nan_and_layout_has_29_points():
    H = np.eye(3)[None]
    x, y, c = calibration.apply_homographies(np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([5, 50]), np.array([0]), np.array([10]), H, np.array([1.0]))
    assert np.isfinite(x[0]) and np.isnan(x[1])
    lay = calibration.default_layout(105, 68)
    assert len(lay) == 29 and lay[28]["x"] == 52.5 and lay[8]["x"] == 11.0


# ---------------------------------------------------------------- ball
def test_ball_outlier_rejection_interpolation_and_cut_guard():
    n = 30
    df = pd.DataFrame({"id": range(n), "frame": np.arange(n) * 3, "ts": np.arange(n) / 10.0, "px": 100 + np.arange(n) * 10.0, "py": 500.0, "conf": 0.5})
    df.loc[10, "px"] = 1500.0                                 # detection on a spectator's shirt
    df = df.drop(index=[20, 21])                               # 2 missed frames -> interpolate
    kept, interp, rej = ball.clean_and_interpolate(df, 3, 1920, [], cfg)
    assert rej == [10] and len(interp) == 3                    # frame 30 (outlier) + frames 60, 63 ... gaps
    assert set(interp["frame"]) == {30, 60, 63}
    kept, interp2, _ = ball.clean_and_interpolate(df, 3, 1920, [61], cfg)   # a camera cut inside the gap -> no interpolation there
    assert 60 not in set(interp2["frame"])


def test_link_to_players_uses_player_height_as_unit():
    tr = pd.DataFrame({"frame": [0, 0], "track_id": [1, 2], "px": [100.0, 300.0], "py": [500.0, 500.0], "x1": [92, 292], "y1": [467, 467], "x2": [108, 308], "y2": [500, 500]})
    b = pd.DataFrame({"id": [0], "frame": [0], "px": [110.0], "py": [498.0]})
    out = ball.link_to_players(b, tr, 0.7)
    assert int(out["nearest_track"][0]) == 1 and out["nearest_dist"][0] < 0.5 and int(out["possession_track"][0]) == 1
    b2 = pd.DataFrame({"id": [0], "frame": [0], "px": [200.0], "py": [498.0]})
    assert ball.link_to_players(b2, tr, 0.7)["possession_track"].isna().all()


# ---------------------------------------------------------------- teams
def test_kit_clustering_finds_two_kits_and_flags_outliers():
    rng = np.random.default_rng(0)
    home, away, gk = np.array([90, 150, 110.]), np.array([200, 128, 128.]), np.array([180, 100, 190.])
    labs = np.vstack([home + rng.normal(0, 3, (10, 3)), away + rng.normal(0, 3, (10, 3)), gk[None]])
    labels, dist, cent = teams.cluster_kits(labs.astype(np.float32), np.full(len(labs), 100.0), 2.5)
    assert len(set(labels[:10])) == 1 and len(set(labels[10:20])) == 1 and labels[0] != labels[10] and labels[20] == -1
    m, confirmed = teams.map_clusters(cent, "#D62828", None)   # one team colour missing -> the mapping is a guess
    assert confirmed is False


def test_kit_mapping_from_team_colours():
    # cluster 0 is white, cluster 1 is red; home is red, away is white -> {0: away, 1: home}
    white, red = teams.hex_to_lab("#FFFFFF"), teams.hex_to_lab("#D62828")
    mapping, confirmed = teams.map_clusters(np.stack([white, red]), "#D62828", "#FFFFFF")
    assert confirmed and mapping == {0: "away", 1: "home"}


# ---------------------------------------------------------------- direction
def test_direction_inference():
    ha, conf = direction.infer_period_direction(np.full(300, 40.0), np.full(300, 62.0))
    assert ha == "right" and 0.5 < conf <= 0.8
    assert direction.infer_period_direction(np.full(300, 50.0), np.full(300, 52.0))[0] is None       # too close to call
    assert direction.infer_period_direction(np.full(50, 40.0), np.full(50, 62.0))[0] is None          # too little data
    assert direction.is_goalkeeper_position(4, 34, 105, 68) and not direction.is_goalkeeper_position(50, 34, 105, 68)


# ---------------------------------------------------------------- identity
def test_jersey_vote_confidence_is_conservative():
    num, conf, _ = identity.vote_result({"9": {"w": 2.7, "n": 3}})
    assert num == 9 and conf >= cfg.identity_threshold                       # 3 consistent confident reads -> auto
    assert identity.vote_result({"9": {"w": 0.95, "n": 1}})[1] < 0.4         # a single read is never enough
    num, conf, _ = identity.vote_result({"9": {"w": 1.8, "n": 2}, "8": {"w": 1.5, "n": 2}})
    assert num == 9 and conf < cfg.identity_threshold                        # contested reads -> needs a human
    assert identity.vote_result({}) is None and identity.vote_result(None) is None


# ---------------------------------------------------------------- physical
def test_physical_distance_speed_and_sprints():
    t = np.arange(0, 60, 0.1)
    r = physical.analyse_series(t, 1.5 * t, np.full_like(t, 30.0), cfg)
    assert abs(r["step"].sum() - 1.5 * 59.9) < 2 and abs(np.nanmax(r["speed"]) * 3.6 - 5.4) < 0.3
    sp = np.where((t > 10) & (t < 14), 8.0, 3.0)
    r = physical.analyse_series(t, np.cumsum(sp * 0.1), np.full_like(t, 30.0), cfg)
    assert len(physical.runs(np.nan_to_num(r["speed"]) >= cfg.sprint_kmh / 3.6, t, cfg.min_sprint_s)) == 1


def test_physical_teleport_and_gaps_are_not_counted_as_distance():
    t = np.arange(0, 30, 0.1)
    x = 10 + 1.0 * t
    x[150:] += 60.0                                           # ID switch: the track jumps 60 m
    r = physical.analyse_series(t, x, np.full_like(t, 30.0), cfg)
    assert r["step"].sum() < 40                                # the jump (60 m) must not be added
    t2 = np.r_[np.arange(0, 10, 0.1), np.arange(20, 30, 0.1)]  # 10 s gap in tracking
    r2 = physical.analyse_series(t2, 0.5 * t2, np.full_like(t2, 30.0), cfg)
    assert r2["step"].sum() < 0.5 * 20.5 and r2["step"].sum() > 8


# ---------------------------------------------------------------- statistics
def test_rating_and_attributes_report_missing_inputs_as_null():
    zero = {k: 0 for k in ["passes_attempted", "passes_completed", "progressive_passes", "final_third_passes", "key_passes", "assists", "goals", "shots",
                           "shots_on_target", "carries", "progressive_carries", "dribbles_successful", "dribbles_failed", "tackles_won", "interceptions",
                           "recoveries", "clearances", "blocks", "saves", "pressures", "duels_won", "duels_lost"]}
    zero["xg"] = 0.0
    assert statistics.compute_rating(zero, 5.0, None, None)[0] is None                       # <10 min observed
    r, parts = statistics.compute_rating({**zero, "goals": 1, "passes_attempted": 30, "passes_completed": 27}, 90, 10.0, 90)
    assert r > 7.0 and parts["attacking"] == 0.8
    a = statistics.attributes(zero, None, 90)
    assert a["PAC"]["value"] is None and a["PAC"]["reason"] == "camera_not_calibrated"
    phys = {"top_speed_kmh": {"value": 31.0, "coverage_pct": 90}, "distance_km": {"value": 1.0, "coverage_pct": 90}, "minutes_observed": {"value": 8.0}, "sprints": {"value": 2}}
    assert statistics.attributes(zero, phys, 90)["PAC"]["value"] is None              # 8 min of footage is too little to call anyone quick
    phys["minutes_observed"]["value"] = 80.0
    assert 40 <= statistics.attributes(zero, phys, 90)["PAC"]["value"] <= 99
    assert a["SHO"]["reason"] == "no_shots_observed" and a["PAS"]["value"] is None                    # not 0 — unknown
    assert statistics.attributes({**zero, "shots": 1, "shots_on_target": 1, "xg": 0.2}, None, 90)["SHO"]["value"] is None      # one shot proves nothing
    a = statistics.attributes({**zero, "shots": 4, "shots_on_target": 2, "xg": 0.6}, None, 90)
    assert 40 <= a["SHO"]["value"] <= 99


def test_formation_from_average_depth():
    xs = np.array([28, 27, 29, 26] + [50, 52, 49, 51] + [72, 74])          # 4-4-2 by depth
    assert statistics.estimate_formation(xs) == "4-4-2"
    assert statistics.estimate_formation(np.array([30, 40, 50])) is None


# ---------------------------------------------------------------- AI guard
def test_ai_statements_with_invented_numbers_are_dropped():
    facts = {"home.possession_pct": 57.1, "away.possession_pct": 42.9, "home.shots": 9}
    good = {"text": "Home controlled 57% of the ball and took 9 shots.", "facts_used": ["home.possession_pct", "home.shots"]}
    bad_num = {"text": "Home took 14 shots.", "facts_used": ["home.shots"]}
    bad_id = {"text": "Away pressed high.", "facts_used": ["away.ppda"]}
    uncited = {"text": "Home looked dangerous.", "facts_used": []}
    assert ai.verify_statement(good, facts)[0]
    assert ai.verify_statement(bad_num, facts) == (False, "number_14_not_in_cited_facts")
    assert ai.verify_statement(bad_id, facts)[1] == "unknown_fact_id" and ai.verify_statement(uncited, facts)[1] == "no_facts_cited"
    payload = {"context": {}, "facts": [], "facts_map": facts, "not_available": [], "limitations": []}
    fake = lambda system, user: '```json\n{"headline": %s, "attacking": [%s, %s]}\n```' % (__import__("json").dumps(good), __import__("json").dumps(bad_num), __import__("json").dumps(good))
    content, dropped = ai.analyse(payload, ai.TEAM_SCHEMA, fake)
    assert len(content["attacking"]) == 1 and len(dropped) == 1 and dropped[0]["reason"].startswith("number_14")


def test_ai_flatten_facts_uses_only_measured_values():
    m = {"possession_pct": {"value": 57.1, "unit": "%"}, "corners": {"value": None, "status": "not_available", "reason": "not_detected"},
         "shape": {"width_m": 44.2, "depth_m": 30.1}}
    f = ai.flatten_facts(m, "home", {})
    assert f == {"home.possession_pct": 57.1, "home.shape.width_m": 44.2, "home.shape.depth_m": 30.1}
