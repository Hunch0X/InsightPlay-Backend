"""Runs the analysis pipeline stage by stage. Every stage is one DB transaction, so a failed stage leaves no partial output
and a retried job resumes at the first stage that has not completed."""
from __future__ import annotations
import importlib
import json
import logging
import traceback
from dataclasses import dataclass, field

from db import connect, execute, fetch_one, to_json
from reporting import Cancelled, Reporter

log = logging.getLogger("orchestrator")

STAGE_ORDER = ["detect_track", "ball", "teams", "calibration", "pose", "direction", "identity",
               "physical", "events", "statistics", "ai"]


@dataclass
class Ctx:
    cfg: object
    conn: object
    job: dict
    match: dict
    video: dict | None
    rep: Reporter
    summary: dict = field(default_factory=dict)
    home: dict | None = None
    away: dict | None = None

    @property
    def job_id(self) -> str:
        return str(self.job["id"])

    @property
    def pitch(self) -> tuple[float, float]:
        return float(self.match["pitch_length"]), float(self.match["pitch_width"])

    @property
    def job_config(self) -> dict:
        return self.job.get("config") or {}

    def set_model_versions(self, versions: dict) -> None:
        """Job-row writes MUST use the status connection: the stage transaction (self.conn) would hold a row lock that the
        progress reporter, on its own connection, then waits on forever."""
        execute(self.rep.conn, "UPDATE analysis_jobs SET model_versions = model_versions || %s::jsonb WHERE id=%s", (json.dumps(versions), self.job_id))

    def warn(self, msg: str) -> None:
        w = self.summary.setdefault("warnings", [])
        if msg not in w:
            w.append(msg)
            log.warning("[%s] %s", self.job_id, msg)

    def periods(self) -> list[dict]:
        """Time windows (video seconds) with optional user-supplied attack direction."""
        cfg = self.job_config
        if cfg.get("periods"):
            return [dict(p) for p in cfg["periods"]]
        start = float(cfg.get("start_s") or 0.0)
        end = float(cfg.get("end_s") or (self.video or {}).get("duration_s") or 1e9)
        return [{"start_s": start, "end_s": end}]


def _load_ctx(cfg, conn, rep, job) -> Ctx:
    match = fetch_one(conn, "SELECT * FROM matches WHERE id=%s", (job["match_id"],))
    video = fetch_one(conn, "SELECT * FROM video_assets WHERE id=%s", (job["video_id"],)) if job.get("video_id") else None
    ctx = Ctx(cfg=cfg, conn=conn, job=job, match=match, video=video, rep=rep, summary=dict(job.get("summary") or {}))
    if match.get("home_team_id"):
        ctx.home = fetch_one(conn, "SELECT * FROM teams WHERE id=%s", (match["home_team_id"],))
    if match.get("away_team_id"):
        ctx.away = fetch_one(conn, "SELECT * FROM teams WHERE id=%s", (match["away_team_id"],))
    return ctx


def _save_summary(status_conn, ctx: Ctx) -> None:
    execute(status_conn, "UPDATE analysis_jobs SET summary = summary || %s::jsonb WHERE id=%s", (to_json(ctx.summary), ctx.job_id))


def run_job(cfg, redis_client, job_id: str) -> None:
    job, current = None, None
    status = connect(cfg.database_url, autocommit=True)
    work = connect(cfg.database_url, autocommit=False)
    try:
        job = fetch_one(status, "SELECT * FROM analysis_jobs WHERE id=%s", (job_id,))
        if not job or job["status"] not in ("queued", "running", "recomputing"):
            log.info("job %s not runnable (%s)", job_id, job and job["status"])
            return
        recompute = (job.get("config") or {}).get("recompute")
        stages = STAGE_ORDER
        done = list(job.get("stages_done") or [])
        if recompute:
            idx = STAGE_ORDER.index(recompute["from_stage"])
            done = [s for s in done if STAGE_ORDER.index(s) < idx] if all(s in STAGE_ORDER for s in done) else []
            execute(status, """UPDATE analysis_jobs SET stages_done=%s::jsonb, heartbeat_at=now(),
                               attempts = CASE WHEN error LIKE 'retrying%%' THEN attempts+1 ELSE 1 END WHERE id=%s""", (json.dumps(done), job_id))
        else:
            execute(status, "UPDATE analysis_jobs SET status='running', attempts=attempts+1, started_at=COALESCE(started_at, now()), heartbeat_at=now() WHERE id=%s", (job_id,))
        job = fetch_one(status, "SELECT * FROM analysis_jobs WHERE id=%s", (job_id,))

        rep = Reporter(cfg, redis_client, status, job_id, stages)
        rep.done = set(done)
        ctx = _load_ctx(cfg, work, rep, job)
        if job["video_id"] is None or ctx.video is None:
            raise RuntimeError("video asset no longer exists")

        for name in stages:
            if name in rep.done:
                continue
            mod = importlib.import_module(f"pipeline.{name}")
            ok, reason = mod.enabled(ctx) if hasattr(mod, "enabled") else (True, "")
            if not ok:
                ctx.summary.setdefault("stages_skipped", {})[name] = reason
                rep.stage_completed(name)
                _save_summary(status, ctx)
                log.info("[%s] stage %s skipped: %s", job_id, name, reason)
                continue
            current = name
            rep.check_cancel(every=0)
            rep.stage_started(name)
            log.info("[%s] stage %s", job_id, name)
            ctx.summary.get("stages_skipped", {}).pop(name, None)
            mod.run(ctx)
            work.commit()
            rep.progress(name, 1.0, force=True)
            rep.stage_completed(name)
            _save_summary(status, ctx)
            ctx.summary = dict(fetch_one(status, "SELECT summary FROM analysis_jobs WHERE id=%s", (job_id,))["summary"])

        execute(status, "UPDATE analysis_jobs SET status='completed', stage=NULL, progress=100, finished_at=now(), error=NULL, summary = summary - 'recompute_failed' WHERE id=%s", (job_id,))
        execute(status, "UPDATE matches SET status='analyzed' WHERE id=%s", (job["match_id"],))
        if ctx.summary.get("review"):
            rep.needs_review(ctx.summary["review"])
        rep.completed()
        log.info("[%s] completed", job_id)

    except Cancelled:
        work.rollback()
        execute(status, "UPDATE analysis_jobs SET status='cancelled', finished_at=now() WHERE id=%s", (job_id,))
        Reporter(cfg, redis_client, status, job_id, STAGE_ORDER).failed("cancelled")
        log.info("[%s] cancelled", job_id)
    except Exception as e:
        work.rollback()
        tb = traceback.format_exc()
        log.error("[%s] failed: %s\n%s", job_id, e, tb)
        row = fetch_one(status, "SELECT attempts, config FROM analysis_jobs WHERE id=%s", (job_id,)) or {"attempts": 99, "config": {}}
        if row["attempts"] < cfg.max_attempts:
            nxt = "recomputing" if (row["config"] or {}).get("recompute") else "queued"
            execute(status, "UPDATE analysis_jobs SET status=%s, error=%s WHERE id=%s", (nxt, f"retrying after: {e}"[:1000], job_id))
            redis_client.rpush(cfg.queue_key, job_id)
        elif (row["config"] or {}).get("recompute"):
            # Each stage is atomic, so the data is intact but may mix old and new stages. Keep results visible and flag them.
            msg = f"recompute failed at stage '{current}': {type(e).__name__}: {e}"[:1000]
            execute(status, "UPDATE analysis_jobs SET status='completed', error=%s, finished_at=now(), summary = summary || %s::jsonb WHERE id=%s",
                    (msg, to_json({"recompute_failed": {"stage": current, "error": msg, "note": "results may combine old and new stages; recompute again"}}), job_id))
            Reporter(cfg, redis_client, status, job_id, STAGE_ORDER).failed(msg)
        else:
            execute(status, "UPDATE analysis_jobs SET status='failed', error=%s, finished_at=now() WHERE id=%s", (f"{type(e).__name__}: {e}"[:1000], job_id))
            if job:
                execute(status, "UPDATE matches SET status='failed' WHERE id=%s", (job["match_id"],))
            Reporter(cfg, redis_client, status, job_id, STAGE_ORDER).failed(str(e))
    finally:
        status.close()
        work.close()
