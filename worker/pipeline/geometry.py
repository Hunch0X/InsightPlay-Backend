"""Pitch geometry shared by the physical, event and statistics stages.

Pitch coordinates: x along the length (0..L), y across (0..W), metres.
'home_attacks' is 'right' when the home team attacks towards x = L."""
from __future__ import annotations
import math


def attack_sign(team: str | None, home_attacks: str | None) -> int | None:
    """+1 if the team attacks towards increasing x, -1 towards decreasing x, None if unknown."""
    if team not in ("home", "away") or home_attacks not in ("left", "right"):
        return None
    home_sign = 1 if home_attacks == "right" else -1
    return home_sign if team == "home" else -home_sign


def period_index(ts: float, periods: list[dict]) -> int | None:
    for i, p in enumerate(periods):
        if p["start_s"] <= ts < p["end_s"]:
            return i
    return None


def home_attacks_at(ts: float, directions: list[dict]) -> str | None:
    i = period_index(ts, directions)
    return directions[i].get("home_attacks") if i is not None else None


def attack_x(x: float, L: float, sign: int) -> float:
    """x measured from the team's own goal line towards the opponent's goal (0..L)."""
    return x if sign > 0 else L - x


def opp_goal(L: float, W: float, sign: int) -> tuple[float, float]:
    return (L if sign > 0 else 0.0), W / 2.0


def dist_to_goal(x: float, y: float, L: float, W: float, sign: int) -> float:
    gx, gy = opp_goal(L, W, sign)
    return math.hypot(gx - x, gy - y)


def third_of(x: float, L: float, sign: int) -> str:
    ax = attack_x(x, L, sign)
    return "defensive_third" if ax < L / 3 else ("middle_third" if ax < 2 * L / 3 else "attacking_third")


GOAL_WIDTH = 7.32


def goal_angle(x: float, y: float, L: float, W: float, sign: int) -> float:
    """Angle (radians) subtended by the goalposts from the shot location."""
    gx, gy = opp_goal(L, W, sign)
    d = abs(gx - x)
    a = math.atan2(GOAL_WIDTH * d, d * d + (y - gy) ** 2 - (GOAL_WIDTH / 2) ** 2)
    return a if a >= 0 else a + math.pi


def xg_geometric(x: float, y: float, L: float, W: float, sign: int) -> float:
    """xG baseline v0: logistic on shot distance and goal-mouth angle.

    NOT fitted to a dataset — a transparent geometric baseline (about 0.04 from 25 m central, 0.22 from the penalty spot,
    0.7 from 6 m). Always reported with source='estimated'. Replace with a trained model when shot data is available."""
    d = dist_to_goal(x, y, L, W, sign)
    theta = goal_angle(x, y, L, W, sign)
    z = -3.8 + 4.5 * theta - 0.03 * d
    return 1.0 / (1.0 + math.exp(-z))


XG_MODEL = "xg_geometric_v0"
