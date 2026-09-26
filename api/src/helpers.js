const config = require('./config');
const db = require('./db');

const asyncH = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

class HttpError extends Error {
  constructor(status, code, detail) { super(code); this.status = status; this.code = code; this.detail = detail; }
}

/** Fetch a match, enforcing ownership when auth is enabled. */
async function getMatch(req, id) {
  const { rows } = await db.query('SELECT * FROM matches WHERE id=$1', [id]);
  const m = rows[0];
  if (!m) throw new HttpError(404, 'match_not_found');
  if (config.authRequired && m.owner_id && m.owner_id !== req.user?.sub) throw new HttpError(404, 'match_not_found');
  return m;
}

/** Fetch a job and enforce ownership through its match. */
async function getJob(req, id) {
  const { rows } = await db.query('SELECT * FROM analysis_jobs WHERE id=$1', [id]);
  const j = rows[0];
  if (!j) throw new HttpError(404, 'job_not_found');
  await getMatch(req, j.match_id);
  return j;
}

/** The job whose results are served: an explicit job, else the latest job with results for the match. */
async function resolveJob(req, matchId, jobId) {
  await getMatch(req, matchId);
  const { rows } = jobId
    ? await db.query('SELECT * FROM analysis_jobs WHERE id=$1 AND match_id=$2', [jobId, matchId])
    : await db.query(`SELECT * FROM analysis_jobs WHERE match_id=$1 AND status IN ('completed','recomputing')
                      ORDER BY created_at DESC LIMIT 1`, [matchId]);
  if (!rows[0]) throw new HttpError(404, 'no_completed_analysis', 'Run POST /api/matches/:id/analyze first.');
  return rows[0];
}

const isUuid = (s) => /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(s || '');

/** entity_key <- URL param: a player uuid or "track:<n>" */
const entityKeyFromParam = (p) => (isUuid(p) ? `player:${p}` : p);

module.exports = { asyncH, HttpError, getMatch, getJob, resolveJob, isUuid, entityKeyFromParam };
