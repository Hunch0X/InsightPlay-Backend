"""Synthetic match generator — TEST FIXTURE ONLY. It fabricates tracker output (positions, ball path) with a known
ground truth so the analytical stages can be verified end to end. It says nothing about real-world detector accuracy."""
from __future__ import annotations
import numpy as np
import pandas as pd

FPS, SRC, STEP = 10, 30, 3
BOXH = 33.0


def to_px(X, Y):
    return 40 + 17 * np.asarray(X, float), 60 + 14 * np.asarray(Y, float)


CALIB_POINTS = [{"px": float(40 + 17 * x), "py": float(60 + 14 * y), "x": float(x), "y": float(y)}
                for x, y in [(0, 0), (105, 0), (105, 68), (0, 68), (52.5, 34), (16.5, 13.8), (88.5, 54.2)]]

# tid -> (team, role, base x, base y, amplitude m)
LAYOUT = {1: ("home", "goalkeeper", 5, 34, 0.5)}
for i, y in enumerate([12, 26, 42, 56]):
    LAYOUT[2 + i] = ("home", "player", 25, y, 3)
for i, y in enumerate([14, 26, 42, 54]):
    LAYOUT[6 + i] = ("home", "player", 48, y, 4)
LAYOUT[10] = ("home", "player", 78, 28, 3)
LAYOUT[11] = ("home", "player", 78, 40, 3)
LAYOUT[21] = ("away", "goalkeeper", 100, 34, 0.0)
for i, y in enumerate([12, 26, 42, 56]):
    LAYOUT[22 + i] = ("away", "player", 80, y, 3)
for i, y in enumerate([14, 26, 42, 54]):
    LAYOUT[26 + i] = ("away", "player", 57, y, 4)
LAYOUT[30] = ("away", "player", 30, 28, 3)
LAYOUT[31] = ("away", "player", 30, 40, 3)
LAYOUT[40] = (None, "referee", 52, 34, 8)


def pos(tid, i):
    team, role, bx, by, amp = LAYOUT[tid]
    t = i / FPS
    ph = tid * 0.7
    return bx + amp * np.sin(2 * np.pi * t / 22 + ph), by + amp * np.cos(2 * np.pi * t / 22 + ph)


# ball script: ('hold', tid, frames) | ('fly', tid_to, frames)
SCRIPT = [("hold", 6, 15), ("fly", 7, 8), ("hold", 7, 15), ("fly", 9, 6), ("hold", 9, 12), ("fly", 10, 6), ("hold", 10, 12),
          ("fly", 21, 12), ("hold", 21, 15),                       # shot from 10 saved by the away goalkeeper
          ("fly", 26, 10), ("hold", 26, 15),                       # away GK -> away midfielder (completed)
          ("fly", 7, 12), ("hold", 7, 12), ("fly", 27, 10), ("hold", 27, 20)]   # home 7 -> away 27 (intercepted)


def build(seconds_pad=5):
    ball, i = [], 0
    last = None
    for kind, tid, n in SCRIPT:
        if kind == "hold":
            for k in range(n):
                x, y = pos(tid, i + k)
                ball.append((i + k, x, y, tid))
            last = (i + n - 1, *pos(tid, i + n - 1))
            i += n
        else:
            x0, y0 = ball[-1][1], ball[-1][2]
            x1, y1 = pos(tid, i + n)
            for k in range(n):
                a = (k + 1) / (n + 1)
                ball.append((i + k, x0 + a * (x1 - x0), y0 + a * (y1 - y0), None))
            i += n
    n_frames = i + seconds_pad * FPS
    tracks = []
    for tid, (team, role, *_r) in LAYOUT.items():
        for k in range(n_frames):
            X, Y = pos(tid, k)
            px, py = to_px(X, Y)
            tracks.append({"track_id": tid, "frame": k * STEP, "ts": k * STEP / SRC, "role": role, "team": team, "px": float(px), "py": float(py),
                           "x1": float(px - 8), "y1": float(py - BOXH), "x2": float(px + 8), "y2": float(py), "conf": 0.9,
                           "pitch_x": float(X), "pitch_y": float(Y)})
    balls = []
    for k, x, y, holder in ball:
        px, py = to_px(x, y)
        off = 2.0 if holder is not None else 0.0
        balls.append({"frame": k * STEP, "ts": k * STEP / SRC, "px": float(px) + off, "py": float(py) - off, "conf": 0.7, "interpolated": False,
                      "pitch_x": float(x), "pitch_y": float(y)})
    return tracks, balls, n_frames


def frames(uncalibrated=False):
    """DataFrames in the format the DB stages produce, with ball<->player linkage computed by the real ball stage code."""
    from pipeline.ball import link_to_players
    from config import Config
    tracks, balls, n = build()
    tdf, bdf = pd.DataFrame(tracks), pd.DataFrame(balls)
    tdf["player_id"] = None
    bdf["id"] = np.arange(len(bdf))
    link = link_to_players(bdf, tdf, Config().control_dist)
    bdf = bdf.merge(link, on="id")
    if uncalibrated:
        for df in (tdf, bdf):
            df["pitch_x"], df["pitch_y"] = np.nan, np.nan
    return tdf, bdf.drop(columns=["id"]), n
