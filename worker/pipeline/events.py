"""Stage — football events derived from tracking + ball + geometry (never from a language model).

Possession spells are found from ball-to-feet distance (in player heights, so it works without calibration). Events are then
derived from spell transitions and ball kinematics:
  pass (completed / intercepted), interception, tackle, recovery, carry (progressive), dribble, pressure, duel,
  shot (outcome + geometric xG baseline), save, block, goal (always needs_review), clearance (needs_review),
  key_pass, assist.
Distances, progressive/final-third flags, shots and xG need calibrated positions AND a known attack direction; without
them those fields are omitted (or the event is not produced) rather than guessed."""
from __future__ import annotations
import math
from collections import defaultdict

import numpy as np
import pandas as pd

from db import copy_rows, execute, fetch_df, to_json
from pipeline.geometry import GOAL_WIDTH, XG_MODEL, attack_sign, attack_x, dist_to_goal, home_attacks_at, third_of, xg_geometric

NAME = "events"


def enabled(ctx):
    if not ctx.summary.get("ball", {}).get("frames_with_ball"):
        return False, "no ball tracking available"
    return True, ""


# --------------------------------------------------------------------------------------------- possession spells
def possession_segments(frames, nearest, nd, poss, cfg, step) -> list[tuple[int, int, int]]:
    """(track, first_row, last_row) spells. A spell starts after `min_control_frames` consecutive rows in which the same track
    is within control distance, survives brief occlusion (`release_frames`), and ends when the ball leaves or another track
    takes over. Pure function, unit-tested."""
    segs, owner, miss, cand, cand_n, cand_start = [], -1, 0, -1, 0, 0
    ch, ch_n, ch_start, seg_start, last_i, prev_f = -1, 0, 0, 0, 0, None
    for i in range(len(frames)):
        f = frames[i]
        gap = 0 if prev_f is None else max(0, int(round((f - prev_f) / step)) - 1)
        prev_f = f
        if owner >= 0:
            if nearest[i] == owner and nd[i] <= cfg.release_dist:
                miss, last_i, ch, ch_n = 0, i, -1, 0
                continue
            miss += gap + 1
            c = poss[i]
            if c >= 0 and c != owner:
                if c == ch:
                    ch_n += 1
                else:
                    ch, ch_n, ch_start = c, 1, i
                if ch_n >= cfg.min_control_frames:
                    segs.append((owner, seg_start, last_i))
                    owner, seg_start, last_i, miss, ch, ch_n = ch, ch_start, i, 0, -1, 0
                    continue
            else:
                ch, ch_n = -1, 0
            if miss > cfg.release_frames:
                segs.append((owner, seg_start, last_i))
                owner, cand, cand_n = -1, -1, 0
            continue
        c = poss[i]
        if c >= 0:
            if c == cand:
                cand_n += 1
            else:
                cand, cand_n, cand_start = c, 1, i
            if cand_n >= cfg.min_control_frames:
                owner, seg_start, last_i, miss, ch, ch_n = cand, cand_start, i, 0, -1, 0
        else:
            cand, cand_n = -1, 0
    if owner >= 0:
        segs.append((owner, seg_start, last_i))
    return segs


def merge_same_entity(segs, ent_of, ts, gap_s):
    out = []
    for s in segs:
        if out and ent_of(out[-1][0]) == ent_of(s[0]) and ts[s[1]] - ts[out[-1][2]] <= gap_s:
            out[-1] = (out[-1][0], out[-1][1], s[2])
        else:
            out.append(s)
    return out


# --------------------------------------------------------------------------------------------------- detector
class Detector:
    def __init__(self, tracks: pd.DataFrame, ball: pd.DataFrame, directions: list[dict], L: float, W: float, cfg, step: int):
        self.cfg, self.L, self.W, self.dirs, self.step = cfg, L, W, directions, step
        t = tracks.copy()
        t["h"] = (t["y2"] - t["y1"]).clip(lower=8.0)
        self.track_team = t.groupby("track_id")["team"].agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None).to_dict()
        self.track_role = t.groupby("track_id")["role"].agg(lambda s: s.mode().iloc[0]).to_dict()
        pid = t.groupby("track_id")["player_id"].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else None).to_dict()
        self.ent = {tid: (f"p:{p}" if p else f"t:{tid}") for tid, p in pid.items()}
        self.track_h = t.groupby("track_id")["h"].median().to_dict()
        b = ball.sort_values("frame").reset_index(drop=True)
        self.b = b
        self.frames = b["frame"].to_numpy(int)
        self.ts = b["ts"].to_numpy(float)
        self.bpx, self.bpy = b["px"].to_numpy(float), b["py"].to_numpy(float)
        self.bX, self.bY = b["pitch_x"].to_numpy(float), b["pitch_y"].to_numpy(float)
        self.interp = b["interpolated"].to_numpy(bool)
        self.calibrated = bool(np.isfinite(self.bX).any())
        fs = set(self.frames.tolist())
        tf = t[t["frame"].isin(fs) & t["role"].isin(["player", "goalkeeper"])]
        self.P = {(int(r.frame), int(r.track_id)): (r.px, r.py, r.h, r.pitch_x, r.pitch_y) for r in tf.itertuples()}
        self.by_frame = defaultdict(list)
        for (f, tid) in self.P:
            self.by_frame[f].append(tid)
        self.events: list[dict] = []

    # ---- helpers
    def sign(self, team, ts):
        return attack_sign(team, home_attacks_at(ts, self.dirs))

    def bxy(self, i):
        return (self.bX[i], self.bY[i]) if np.isfinite(self.bX[i]) else None

    def disp(self, i, j, tid):
        """Ball displacement between rows i and j: (value, 'm'|'norm')."""
        a, c = self.bxy(i), self.bxy(j)
        if a and c:
            return math.hypot(c[0] - a[0], c[1] - a[1]), "m"
        return math.hypot(self.bpx[j] - self.bpx[i], self.bpy[j] - self.bpy[i]) / max(self.track_h.get(tid, 100.0), 1.0), "norm"

    def pdist(self, f, a, b):
        pa, pb = self.P.get((f, a)), self.P.get((f, b))
        if not pa or not pb:
            return None
        if np.isfinite(pa[3]) and np.isfinite(pb[3]):
            return math.hypot(pa[3] - pb[3], pa[4] - pb[4]), "m"
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1]) / max((pa[2] + pb[2]) / 2, 1.0), "norm"

    def conf(self, base, *rows, need_cal=False):
        c = base - 0.1 * sum(bool(self.interp[r]) for r in rows)
        if need_cal and not self.calibrated:
            c -= 0.2
        return float(max(0.05, min(0.95, c)))

    def add(self, type_, subtype, i, team, track, target=None, start=None, end=None, dist=None, conf=0.5, attrs=None, evidence=None, review=False):
        self.events.append({
            "type": type_, "subtype": subtype, "ts": float(self.ts[i]), "frame": int(self.frames[i]), "team": team,
            "track_id": int(track) if track is not None else None, "target_track_id": int(target) if target is not None else None,
            "sx": start[0] if start else None, "sy": start[1] if start else None, "ex": end[0] if end else None, "ey": end[1] if end else None,
            "distance_m": dist, "confidence": conf, "status": "needs_review" if (review or conf < 0.5) else "auto",
            "attributes": attrs or {}, "evidence": evidence or {}})

    # ---- ball flight after a release
    def flight(self, i_end):
        """Ball velocity (vx, vy, speed) in pitch metres over the second after row i_end; None if not measurable."""
        idx = [k for k in range(i_end + 1, min(len(self.ts), i_end + 8)) if self.ts[k] - self.ts[i_end] <= 1.0 and np.isfinite(self.bX[k])]
        if len(idx) < 2 or not np.isfinite(self.bX[i_end]):
            return None
        idx = [i_end] + idx
        t = self.ts[idx] - self.ts[i_end]
        vx, vy = np.polyfit(t, self.bX[idx], 1)[0], np.polyfit(t, self.bY[idx], 1)[0]
        return float(vx), float(vy), float(math.hypot(vx, vy))

    # ---- main
    def run(self) -> list[dict]:
        cfg = self.cfg
        nearest = self.b["nearest_track"].fillna(-1).to_numpy(int)
        nd = self.b["nearest_dist"].fillna(9.9).to_numpy(float)
        poss = self.b["possession_track"].fillna(-1).to_numpy(int)
        segs = possession_segments(self.frames, nearest, nd, poss, cfg, self.step)
        segs = merge_same_entity(segs, lambda t: self.ent.get(t, f"t:{t}"), self.ts, cfg.merge_gap_s)
        self.segs = segs
        shot_at: dict[int, dict] = {}
        for k, (tr, i0, i1) in enumerate(segs):
            team = self.track_team.get(tr)
            if team not in ("home", "away"):
                continue
            self.carry_dribble_pressure(k, tr, team, i0, i1)
            nxt = segs[k + 1] if k + 1 < len(segs) else None
            shot = self.try_shot(k, tr, team, i1, nxt)
            if shot:
                shot_at[k] = shot
                continue
            if nxt is None:
                continue
            self.transition(k, tr, team, i0, i1, nxt)
        self.key_passes(shot_at)
        self.duels(segs)
        return self.events

    # ---- carries, dribbles, pressure on the carrier
    def carry_dribble_pressure(self, k, tr, team, i0, i1):
        cfg, f0, f1 = self.cfg, self.frames[i0], self.frames[i1]
        a, c = self.bxy(i0), self.bxy(i1)
        sgn = self.sign(team, self.ts[i0])
        if a and c:
            d = math.hypot(c[0] - a[0], c[1] - a[1])
            if d >= cfg.min_carry_m:
                prog = False
                if sgn:
                    d0, d1 = dist_to_goal(a[0], a[1], self.L, self.W, sgn), dist_to_goal(c[0], c[1], self.L, self.W, sgn)
                    prog = (d0 - d1) >= 5.0 and (d0 - d1) >= 0.10 * d0
                self.add("carry", "progressive" if prog else "regular", i0, team, tr, start=a, end=c, dist=d,
                         conf=self.conf(0.6, i0, i1, need_cal=True), attrs={"progressive": prog if sgn else None, "duration_s": float(self.ts[i1] - self.ts[i0])})
        # opponents engaging the carrier
        engaged_first, presser_first = None, {}
        for r in range(i0, i1 + 1):
            f = int(self.frames[r])
            for other in self.by_frame.get(f, ()):
                if self.track_team.get(other) in (None, team) or other == tr:
                    continue
                dd = self.pdist(f, tr, other)
                if dd is None:
                    continue
                v, u = dd
                if (u == "m" and v <= cfg.pressure_radius_m) or (u == "norm" and v <= 2.0):
                    presser_first.setdefault(other, [r, r])[1] = r
                if (u == "m" and v <= cfg.duel_radius_m) or (u == "norm" and v <= 1.2):
                    engaged_first = r if engaged_first is None else engaged_first
        for other, (ra, rb) in presser_first.items():
            if self.ts[rb] - self.ts[ra] >= 0.3:
                self.add("pressure", None, ra, self.track_team.get(other), other, target=tr, conf=self.conf(0.55, ra), attrs={"duration_s": float(self.ts[rb] - self.ts[ra])})
        self._engaged = getattr(self, "_engaged", {})
        self._engaged[k] = engaged_first

    # ---- what happens between spell k and spell k+1 (and shots / clearances on release)
    def transition(self, k, tr, team, i0, i1, nxt):
        cfg = self.cfg
        tr2, j0, j1 = nxt
        team2 = self.track_team.get(tr2)
        if team2 not in ("home", "away") or self.ent.get(tr) == self.ent.get(tr2):
            return
        gap = float(self.ts[j0] - self.ts[i1])
        d, unit = self.disp(i1, j0, tr)
        far = d >= (cfg.min_pass_m if unit == "m" else cfg.min_pass_norm)
        a, c = self.bxy(i1), self.bxy(j0)
        sgn = self.sign(team, self.ts[i1])
        ev = {"gap_s": round(gap, 2), "ball_displacement": round(d, 2), "unit": unit}
        if gap > cfg.max_pass_gap_s:
            if team2 != team:
                self.add("recovery", "loose_ball", j0, team2, tr2, conf=self.conf(0.4, j0), evidence=ev)
            return
        if far and self.is_clearance(k, tr, team, i0, i1, a, c, sgn, team2):
            self.add("clearance", None, i1, team, tr, start=a, end=c, dist=d if unit == "m" else None, conf=0.4, review=True, evidence=ev)
            return
        if far:
            attrs = {}
            if a and c and sgn:
                d0, d1 = dist_to_goal(a[0], a[1], self.L, self.W, sgn), dist_to_goal(c[0], c[1], self.L, self.W, sgn)
                attrs["progressive"] = bool((d0 - d1) >= max(10.0, cfg.progressive_frac * d0)) if team == team2 else None
                attrs["final_third"] = attack_x(c[0], self.L, sgn) >= 2 * self.L / 3
                attrs["direction"] = "forward" if (c[0] - a[0]) * sgn > 2 else ("backward" if (c[0] - a[0]) * sgn < -2 else "lateral")
            if team == team2:
                self.add("pass", "completed", i1, team, tr, target=tr2, start=a, end=c, dist=d if unit == "m" else None,
                         conf=self.conf(0.65, i1, j0, need_cal=False), attrs=attrs, evidence=ev)
            else:
                self.add("pass", "intercepted", i1, team, tr, target=None, start=a, end=c, dist=d if unit == "m" else None,
                         conf=self.conf(0.55, i1, j0), attrs=attrs, evidence=ev)
                self.add("interception", None, j0, team2, tr2, target=tr, start=c, conf=self.conf(0.55, i1, j0), evidence=ev)
        elif team != team2 and gap <= cfg.tackle_gap_s:
            self.add("tackle", "won", j0, team2, tr2, target=tr, start=c, conf=self.conf(0.5, j0), evidence=ev)
            if self._engaged.get(k) is not None:
                self.add("dribble", "failed", i1, team, tr, target=tr2, conf=self.conf(0.5, i1), evidence=ev)
        elif team != team2:
            self.add("recovery", "regain", j0, team2, tr2, target=tr, start=c, conf=self.conf(0.45, j0), evidence=ev)
        # a carrier who was engaged and kept the ball through to a teammate/next action has dribbled past
        if team == team2 and self._engaged.get(k) is not None and self.ts[i1] - self.ts[self._engaged[k]] >= 1.0:
            self.add("dribble", "successful", self._engaged[k], team, tr, conf=self.conf(0.5, i1), evidence=ev)

    def is_clearance(self, k, tr, team, i0, i1, a, c, sgn, team2):
        if not (a and c and sgn) or attack_x(a[0], self.L, sgn) >= self.L / 3:
            return False
        fl = self.flight(i1)
        forward = (c[0] - a[0]) * sgn
        pressed = self._engaged.get(k) is not None
        return bool(fl and fl[2] >= 12.0 and forward >= 20.0 and pressed and (team2 != team))

    # ---- shots (need calibration + direction; height is unknowable so 'on target' only comes with save/goal evidence)
    def try_shot(self, k, tr, team, i1, nxt):
        cfg, L, W = self.cfg, self.L, self.W
        sgn = self.sign(team, self.ts[i1])
        fl = self.flight(i1)
        a = self.bxy(i1)
        if not (sgn and fl and a):
            return None
        vx, vy, spd = fl
        if spd < cfg.shot_min_speed or vx * sgn <= 0.5 * spd:
            return None
        dgoal = dist_to_goal(a[0], a[1], L, W, sgn)
        if dgoal > cfg.shot_max_dist_m:
            return None
        gx = L if sgn > 0 else 0.0
        t_cross = (gx - a[0]) / vx
        y_cross = a[1] + vy * t_cross
        off_centre = abs(y_cross - W / 2)
        if t_cross <= 0 or off_centre > GOAL_WIDTH / 2 + 3.0:
            return None
        aligned = off_centre <= GOAL_WIDTH / 2
        outcome, conf, review, blocker, saver = None, 0.5, False, None, None
        nx = None
        if nxt is not None:
            tr2, j0, _ = nxt
            team2 = self.track_team.get(tr2)
            dt = self.ts[j0] - self.ts[i1]
            if dt <= 4.0:
                nx = (tr2, team2, dt, j0)
        if nx and nx[1] == team:
            return None                                        # received by a team-mate: a pass, not a shot
        if nx and self.track_role.get(nx[0]) == "goalkeeper" and nx[1] != team:
            outcome, saver = ("saved" if aligned else "off_target"), (nx[0] if aligned else None)
            conf = 0.55
        elif nx and nx[1] != team and nx[2] <= 1.5 and aligned and (self.bxy(nx[3]) is None or math.hypot(self.bxy(nx[3])[0] - a[0], self.bxy(nx[3])[1] - a[1]) < 0.7 * dgoal):
            outcome, blocker, conf = "blocked", nx[0], 0.45
        elif nx is None:
            end_rows = [r for r in range(i1 + 1, min(len(self.ts), i1 + 25)) if np.isfinite(self.bX[r]) and self.ts[r] - self.ts[i1] <= 2.5]
            near_line = end_rows and abs(self.bX[end_rows[-1]] - gx) <= 4.0
            if aligned and near_line:
                outcome, conf, review = "goal", 0.5, True
            else:
                outcome, conf = "off_target", 0.5
        else:
            outcome, conf = "off_target", 0.45
        xg = xg_geometric(a[0], a[1], L, W, sgn)
        attrs = {"outcome": outcome, "on_target": outcome in ("saved", "goal"), "xg": round(xg, 3), "xg_model": XG_MODEL, "xg_source": "estimated",
                 "distance_to_goal_m": round(dgoal, 1), "ball_speed_ms": round(spd, 1), "crossing_offset_m": round(off_centre, 1)}
        self.add("shot", outcome, i1, team, tr, start=a, end=(gx, y_cross), dist=dgoal, conf=self.conf(conf, i1), attrs=attrs, review=review,
                 evidence={"velocity": [round(vx, 1), round(vy, 1)], "height": "unknown (single camera)"})
        if outcome == "goal":
            self.add("goal", None, i1, team, tr, start=a, dist=dgoal, conf=0.5, review=True,
                     attrs={"note": "ball not observed after crossing the line — confirm manually"}, evidence={})
        if saver is not None:
            self.add("save", None, nx[3], nx[1], saver, target=tr, conf=0.5)
        if blocker is not None:
            self.add("block", None, nx[3], nx[1], blocker, target=tr, conf=0.45)
        return {"track": tr, "team": team, "i": i1, "outcome": outcome, "ts": float(self.ts[i1])}

    def key_passes(self, shot_at):
        """A completed pass followed within 6 s by a shot from its receiver is a key pass; if that shot is a goal, an assist."""
        for ev in [e for e in self.events if e["type"] == "pass" and e["subtype"] == "completed"]:
            for sh in shot_at.values():
                if sh["track"] == ev["target_track_id"] and 0 <= sh["ts"] - ev["ts"] <= 6.0 and sh["team"] == ev["team"]:
                    self.events.append({**ev, "type": "key_pass", "subtype": None, "status": "auto", "attributes": {"shot_outcome": sh["outcome"]}})
                    if sh["outcome"] == "goal":
                        self.events.append({**ev, "type": "assist", "subtype": None, "status": "needs_review", "confidence": min(ev["confidence"], 0.45),
                                            "attributes": {"note": "follows an unconfirmed goal"}})
                    break

    # ---- ground duels: opposing players close together with the ball nearby
    def duels(self, segs):
        cfg, open_, done = self.cfg, {}, []
        for r in range(len(self.frames)):
            f = int(self.frames[r])
            ids = self.by_frame.get(f, ())
            if len(ids) < 2:
                continue
            seen = set()
            near = [t for t in ids if (self._to_ball(f, t, r) or 99) <= 2.5 and self.track_team.get(t) in ("home", "away")]
            for x in range(len(near)):
                for y in range(x + 1, len(near)):
                    a, b = near[x], near[y]
                    if self.track_team[a] == self.track_team[b]:
                        continue
                    dd = self.pdist(f, a, b)
                    if dd and ((dd[1] == "m" and dd[0] <= cfg.duel_radius_m) or (dd[1] == "norm" and dd[0] <= 1.0)):
                        key = (min(a, b), max(a, b))
                        seen.add(key)
                        open_.setdefault(key, [r, r])[1] = r
            for key in list(open_):
                if key not in seen and r - open_[key][1] > 2:
                    done.append((key, *open_.pop(key)))
        done += [(k, *v) for k, v in open_.items()]
        for (a, b), ra, rb in done:
            if self.ts[rb] - self.ts[ra] < 0.4:
                continue
            winner = None
            for tr, s0, s1 in segs:
                if self.ts[rb] <= self.ts[s0] <= self.ts[rb] + 1.5 and tr in (a, b):
                    winner = tr
                    break
            loser = (b if winner == a else a) if winner is not None else None
            self.add("duel", "won" if winner is not None else "contested", ra, self.track_team.get(winner if winner is not None else a),
                     winner if winner is not None else a, target=loser if loser is not None else b, conf=self.conf(0.5, ra),
                     attrs={"duration_s": float(self.ts[rb] - self.ts[ra]), "kind": "ground"})

    def _to_ball(self, f, t, r):
        p = self.P.get((f, t))
        if not p:
            return None
        if np.isfinite(p[3]) and np.isfinite(self.bX[r]):
            return math.hypot(p[3] - self.bX[r], p[4] - self.bY[r])
        return 3.0 * math.hypot(p[0] - self.bpx[r], p[1] - self.bpy[r]) / max(p[2], 1.0)   # ~1.8 m per player-height


def detect(tracks, ball, directions, L, W, cfg, step) -> Detector:
    d = Detector(tracks, ball, directions, L, W, cfg, step)
    d.run()
    return d


def possession_summary(d: Detector, dt: float) -> dict:
    """Ball-control time per team (share of *controlled* time), spells per track, controlled time per third."""
    spells, time_team, third = defaultdict(int), defaultdict(float), defaultdict(lambda: defaultdict(float))
    for tr, i0, i1 in d.segs:
        team = d.track_team.get(tr)
        spells[int(tr)] += 1
        if team not in ("home", "away"):
            continue
        dur = float(d.ts[i1] - d.ts[i0]) + dt
        time_team[team] += dur
        mid = (i0 + i1) // 2
        sgn = d.sign(team, d.ts[mid])
        if sgn and np.isfinite(d.bX[mid]):
            third[team][third_of(float(d.bX[mid]), d.L, sgn)] += dur
    return {"spells_by_track": dict(spells), "time_by_team_s": {k: round(v, 1) for k, v in time_team.items()},
            "by_third_s": {t: {k: round(v, 1) for k, v in g.items()} for t, g in third.items()}}


def run(ctx) -> None:
    cfg, conn, job_id = ctx.cfg, ctx.conn, ctx.job_id
    L, W = ctx.pitch
    execute(conn, "DELETE FROM events WHERE job_id=%s", (job_id,))
    tracks = fetch_df(conn, """SELECT track_id, frame, ts, role, team, player_id, y1, y2, px, py, pitch_x, pitch_y FROM player_tracks
                               WHERE job_id=%s AND role IN ('player','goalkeeper')""", (job_id,))
    ball = fetch_df(conn, """SELECT frame, ts, px, py, pitch_x, pitch_y, interpolated, nearest_track, nearest_dist, possession_track
                             FROM ball_tracks WHERE job_id=%s""", (job_id,))
    step = ctx.summary["video"]["analysis_step"]
    det = detect(tracks, ball, ctx.summary.get("directions") or [], L, W, cfg, step) if len(ball) and len(tracks) else None
    events = det.events if det else []
    ctx.summary["possession"] = possession_summary(det, step / ctx.summary["video"]["src_fps"]) if det else {}
    cols = ["job_id", "type", "subtype", "ts", "frame", "team", "track_id", "target_track_id", "sx", "sy", "ex", "ey", "distance_m", "confidence", "status", "attributes", "evidence"]
    copy_rows(conn, "events", cols, [(job_id, e["type"], e["subtype"], e["ts"], e["frame"], e["team"], e["track_id"], e["target_track_id"], e["sx"], e["sy"], e["ex"], e["ey"],
                                      e["distance_m"], e["confidence"], e["status"], to_json(e["attributes"]), to_json(e["evidence"])) for e in events])
    counts = defaultdict(int)
    for e in events:
        counts[e["type"]] += 1
    ctx.summary["events"] = {"total": len(events), "by_type": dict(counts), "needs_review": sum(e["status"] == "needs_review" for e in events),
                             "aerial_duels": "not detected (needs ball height, unavailable from a single camera)", "corners": "not detected"}
    if not ctx.summary.get("calibration", {}).get("method"):
        ctx.warn("events were derived without pitch calibration: pass/carry distances, progressive and final-third flags, shots and xG are unavailable")
    if counts.get("goal"):
        ctx.summary["review"] = {**(ctx.summary.get("review") or {}), "goals": f"{counts['goal']} goal candidate(s) need confirmation (PATCH /api/events/:id)"}
