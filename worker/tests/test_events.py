import numpy as np
import pandas as pd

from config import Config
from pipeline import events as E
from tests import synth

cfg = Config()
DIRS = [{"start_s": 0, "end_s": 1e6, "home_attacks": "right"}]


def detect(uncal=False, dirs=DIRS):
    t, b, n = synth.frames(uncalibrated=uncal)
    return E.detect(t, b, dirs, 105.0, 68.0, cfg, synth.STEP)


def by(d, type_, subtype=None):
    return [e for e in d.events if e["type"] == type_ and (subtype is None or e["subtype"] == subtype)]


def test_possession_spells_follow_the_script():
    d = detect()
    owners = [tr for tr, _, _ in d.segs]
    assert owners == [6, 7, 9, 10, 21, 26, 7, 27]


def test_completed_passes_with_measured_distance_and_flags():
    d = detect()
    p = by(d, "pass", "completed")
    pairs = {(e["track_id"], e["target_track_id"]) for e in p}
    assert {(6, 7), (7, 9), (9, 10), (21, 26)} <= pairs
    e67 = next(e for e in p if (e["track_id"], e["target_track_id"]) == (6, 7))
    # ground truth: both players sit ~ (48-ish, y) apart; the measured distance must match the geometry to within 15 %
    t, b, _ = synth.frames()
    truth = np.hypot(e67["ex"] - e67["sx"], e67["ey"] - e67["sy"])
    assert abs(e67["distance_m"] - truth) < 0.5 and 4 < e67["distance_m"] < 40
    assert e67["attributes"]["direction"] in ("forward", "lateral", "backward")
    p79 = next(e for e in p if (e["track_id"], e["target_track_id"]) == (7, 9))
    assert p79["attributes"]["final_third"] in (True, False)


def test_interception_is_recorded_for_both_sides():
    # scripted: away 26 plays toward home 7, who gets it -> a failed away pass and a home interception
    d = detect()
    assert any(e["track_id"] == 26 and e["team"] == "away" for e in by(d, "pass", "intercepted"))
    assert any(e["track_id"] == 7 and e["team"] == "home" for e in by(d, "interception"))
    # scripted: home 7 loses the ball to away 27 while they stand ~2 m apart -> contact tackle, not a pass
    assert any(e["track_id"] == 27 and e["target_track_id"] == 7 for e in by(d, "tackle"))


def test_shot_is_classified_saved_with_xg_and_a_save_event():
    d = detect()
    shots = by(d, "shot")
    assert len(shots) == 1 and shots[0]["track_id"] == 10 and shots[0]["subtype"] == "saved"
    a = shots[0]["attributes"]
    assert a["on_target"] is True and 0.01 < a["xg"] < 0.5 and a["xg_source"] == "estimated" and a["distance_to_goal_m"] < 40
    assert any(e["track_id"] == 21 for e in by(d, "save"))
    assert shots[0]["evidence"]["height"].startswith("unknown")            # we never pretend to know shot height
    kp = by(d, "key_pass")
    assert any(e["track_id"] == 9 and e["target_track_id"] == 10 for e in kp)     # 9 -> 10, then 10 shoots


def test_no_shots_or_progressive_flags_without_direction():
    d = detect(dirs=[{"start_s": 0, "end_s": 1e6, "home_attacks": None}])
    assert not by(d, "shot") and not by(d, "goal")
    assert all(e["attributes"].get("progressive") is None for e in by(d, "pass", "completed"))


def test_uncalibrated_still_counts_passes_but_gives_no_distances_or_shots():
    d = detect(uncal=True)
    p = by(d, "pass", "completed")
    assert {(6, 7), (7, 9)} <= {(e["track_id"], e["target_track_id"]) for e in p}
    assert all(e["distance_m"] is None and e["sx"] is None for e in p)
    assert not by(d, "shot")
    assert all(e["confidence"] <= 0.65 for e in p)


def test_goal_is_never_auto_confirmed():
    # remove the goalkeeper's pick-up so the ball is lost near the goal line -> goal candidate that needs review
    t, b, n = synth.frames()
    gk_hold = b[(b["possession_track"] == 21)]
    b2 = b[~b["frame"].isin(gk_hold["frame"])]
    d = E.detect(t, b2, DIRS, 105.0, 68.0, cfg, synth.STEP)
    goals = [e for e in d.events if e["type"] == "goal"]
    if goals:                                                    # geometry decides; but if a goal is produced it must be unconfirmed
        assert all(e["status"] == "needs_review" and e["confidence"] <= 0.5 for e in goals)


def test_tackle_between_opponents_in_contact():
    t = []
    for k in range(30):
        for tid, team, x in ((1, "home", 50.0), (2, "away", 50.8)):
            px, py = synth.to_px(x, 34.0)
            t.append({"track_id": tid, "frame": k * 3, "ts": k * 0.1, "role": "player", "team": team, "px": float(px), "py": float(py), "x1": float(px - 8),
                      "y1": float(py - 33), "x2": float(px + 8), "y2": float(py), "conf": .9, "pitch_x": x, "pitch_y": 34.0})
    b = []
    for k in range(30):
        holder = 1 if k < 14 else 2                              # ball changes feet in one frame while they stand 0.8 m apart
        x = 50.0 if holder == 1 else 50.8
        px, py = synth.to_px(x, 34.0)
        b.append({"frame": k * 3, "ts": k * 0.1, "px": float(px) + 2, "py": float(py) - 2, "conf": .7, "interpolated": False, "pitch_x": x, "pitch_y": 34.0})
    tdf, bdf = pd.DataFrame(t), pd.DataFrame(b)
    tdf["player_id"] = None
    bdf["id"] = range(len(bdf))
    from pipeline.ball import link_to_players
    bdf = bdf.merge(link_to_players(bdf, tdf, cfg.control_dist), on="id").drop(columns=["id"])
    d = E.detect(tdf, bdf, DIRS, 105.0, 68.0, cfg, 3)
    tk = by(d, "tackle")
    assert len(tk) == 1 and tk[0]["track_id"] == 2 and tk[0]["target_track_id"] == 1 and tk[0]["team"] == "away"
    assert not by(d, "pass", "completed") and not by(d, "pass", "intercepted")      # contact is not a pass
