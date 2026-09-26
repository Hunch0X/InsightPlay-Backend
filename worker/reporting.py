"""Job progress reporting: DB row (source of truth) + Redis pub/sub (for the WebSocket bridge in the API)."""
from __future__ import annotations
import json
import time

from db import execute, to_json

# relative cost of each stage, used to turn stage progress into overall progress
STAGE_WEIGHTS = {
    "detect_track": 0.50, "ball": 0.02, "teams": 0.01, "calibration": 0.14, "pose": 0.06,
    "identity": 0.02, "direction": 0.01, "physical": 0.06, "events": 0.08, "statistics": 0.04, "ai": 0.06,
}


class Cancelled(Exception):
    pass


class Reporter:
    def __init__(self, cfg, redis_client, status_conn, job_id: str, stages: list[str]):
        self.cfg, self.r, self.conn, self.job_id = cfg, redis_client, status_conn, job_id
        self.stages = stages
        self.t0 = time.time()
        self.done: set[str] = set()
        self._last_pub = 0.0
        self._last_cancel_check = 0.0

    def _publish(self, event: str, **kw):
        msg = {"job_id": self.job_id, "event": event, "ts": time.time(), **kw}
        try:
            self.r.publish(self.cfg.progress_channel, json.dumps(msg))
        except Exception:
            pass  # progress is best-effort; the DB row is authoritative

    def overall(self, stage: str | None, frac: float) -> float:
        total = sum(STAGE_WEIGHTS.get(s, 0.02) for s in self.stages) or 1.0
        acc = sum(STAGE_WEIGHTS.get(s, 0.02) for s in self.stages if s in self.done)
        if stage and stage not in self.done:
            acc += STAGE_WEIGHTS.get(stage, 0.02) * max(0.0, min(1.0, frac))
        return round(100.0 * acc / total, 1)

    def stage_started(self, stage: str):
        execute(self.conn, "UPDATE analysis_jobs SET stage=%s, heartbeat_at=now() WHERE id=%s", (stage, self.job_id))
        self._publish("stage_started", stage=stage, progress=self.overall(stage, 0))

    def progress(self, stage: str, frac: float, detail: dict | None = None, force: bool = False):
        now = time.time()
        if not force and now - self._last_pub < 1.0:
            return
        self._last_pub = now
        pct = self.overall(stage, frac)
        elapsed = now - self.t0
        eta = round(elapsed * (100 - pct) / pct) if pct > 1 else None
        execute(self.conn, "UPDATE analysis_jobs SET progress=%s, heartbeat_at=now() WHERE id=%s", (pct, self.job_id))
        self._publish("progress", stage=stage, progress=pct, eta_s=eta, detail=detail or {})

    def stage_completed(self, stage: str):
        self.done.add(stage)
        execute(self.conn,
                "UPDATE analysis_jobs SET stages_done = stages_done || %s::jsonb, progress=%s, heartbeat_at=now() WHERE id=%s",
                (json.dumps([stage]), self.overall(None, 0), self.job_id))
        self._publish("stage_completed", stage=stage, progress=self.overall(None, 0))

    def needs_review(self, detail: dict):
        self._publish("needs_review", **detail)

    def check_cancel(self, every: float = 2.0):
        now = time.time()
        if now - self._last_cancel_check < every:
            return
        self._last_cancel_check = now
        with self.conn.cursor() as cur:
            cur.execute("SELECT cancel_requested FROM analysis_jobs WHERE id=%s", (self.job_id,))
            row = cur.fetchone()
        if row and row[0]:
            raise Cancelled()

    def completed(self):
        self._publish("completed", progress=100)

    def failed(self, error: str):
        self._publish("failed", error=error[:500])
