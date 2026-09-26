"""Worker entrypoint: consumes analysis job ids from the Redis queue (BLPOP) and runs the pipeline."""
from __future__ import annotations
import logging
import signal
import sys
import time

import redis

from config import Config
from db import connect, execute
from pipeline.orchestrator import run_job

log = logging.getLogger("worker")
_stop = False


def _sig(*_):
    global _stop
    _stop = True
    log.info("shutdown requested; finishing current job")


def requeue_stale(cfg: Config, r) -> None:
    """Re-queue jobs left 'running' by a crashed worker (no heartbeat for stale_job_seconds)."""
    conn = connect(cfg.database_url, autocommit=True)
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE analysis_jobs SET status = CASE WHEN type='recompute' OR config ? 'recompute' THEN 'recomputing' ELSE 'queued' END
               WHERE status IN ('running','recomputing')
                 AND COALESCE(heartbeat_at, started_at, created_at) < now() - make_interval(secs => %s)
               RETURNING id""", (cfg.stale_job_seconds,))
        ids = [str(row[0]) for row in cur.fetchall()]
    conn.close()
    for i in ids:
        r.rpush(cfg.queue_key, i)
        log.warning("re-queued stale job %s", i)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.from_env()
    # socket_timeout must exceed the BLPOP wait below, otherwise an idle queue raises a client-side timeout
    r = redis.Redis.from_url(cfg.redis_url, decode_responses=True, socket_timeout=30, socket_connect_timeout=10, health_check_interval=30)
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    log.info("worker ready; queue=%s detector=%s", cfg.queue_key, cfg.detector_weights)
    last_sweep = 0.0
    while not _stop:
        if time.time() - last_sweep > 60:
            requeue_stale(cfg, r)
            last_sweep = time.time()
        try:
            item = r.blpop(cfg.queue_key, timeout=5)
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            log.warning("redis unavailable (%s); retrying in 3s", e)
            time.sleep(3)
            continue
        if not item:
            continue
        job_id = item[1]
        try:
            run_job(cfg, r, job_id)
        except Exception:  # run_job records failures itself; this is a last-resort guard
            log.exception("unhandled error while running job %s", job_id)
    log.info("worker stopped")


if __name__ == "__main__":
    sys.exit(main())
