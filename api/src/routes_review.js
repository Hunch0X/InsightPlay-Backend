const express = require('express');
const { z } = require('zod');
const db = require('./db');
const { enqueue } = require('./queue');
const { asyncH, HttpError, getJob, resolveJob, getMatch } = require('./helpers');

const r = express.Router();
const RECOMPUTE_STAGES = ['direction', 'identity', 'physical', 'events', 'statistics', 'ai'];

// ---- tracks whose identity needs a human decision ----
async function pending(req, res, jobId) {
  const { rows } = await db.query(
    `SELECT ic.track_id, ic.team, ic.jersey_number, ic.ocr_conf, ic.confidence, ic.status, ic.evidence,
            ic.player_id AS suggested_player_id, p.name AS suggested_player,
            tf.first_frame, tf.last_frame, tf.n_obs
     FROM identity_candidates ic
     LEFT JOIN players p ON p.id=ic.player_id
     LEFT JOIN track_features tf ON tf.job_id=ic.job_id AND tf.track_id=ic.track_id
     WHERE ic.job_id=$1 AND ic.status IN ('needs_confirmation','unmatched')
     ORDER BY tf.n_obs DESC NULLS LAST LIMIT 500`, [jobId]);
  res.json({ job_id: jobId, count: rows.length, tracks: rows });
}
r.get('/jobs/:id/identity/pending', asyncH(async (req, res) => { const j = await getJob(req, req.params.id); await pending(req, res, j.id); }));
r.get('/matches/:id/identity/pending', asyncH(async (req, res) => { const j = await resolveJob(req, req.params.id, req.query.job_id); await pending(req, res, j.id); }));

// ---- confirm / correct / reject a track identity ----
const identitySchema = z.object({
  action: z.enum(['confirm', 'reject']).default('confirm'),
  player_id: z.string().uuid().optional(),
  learn: z.boolean().optional(),   // store this track's appearance as the player's reference embedding
});

r.post('/jobs/:jobId/tracks/:trackId/identity', asyncH(async (req, res) => {
  const job = await getJob(req, req.params.jobId);
  const trackId = Number(req.params.trackId);
  const b = identitySchema.parse(req.body);
  const out = await db.tx(async (c) => {
    const t = await c.query('SELECT team, role FROM player_tracks WHERE job_id=$1 AND track_id=$2 LIMIT 1', [job.id, trackId]);
    if (!t.rows[0]) throw new HttpError(404, 'track_not_found');
    if (b.action === 'reject') {
      await c.query(`INSERT INTO identity_candidates(job_id,track_id,team,status,confidence) VALUES ($1,$2,$3,'rejected',0)
                     ON CONFLICT (job_id,track_id) DO UPDATE SET status='rejected', player_id=NULL, confidence=0`, [job.id, trackId, t.rows[0].team]);
      await c.query('UPDATE player_tracks SET player_id=NULL WHERE job_id=$1 AND track_id=$2', [job.id, trackId]);
      return { track_id: trackId, status: 'rejected', player_id: null };
    }
    if (!b.player_id) throw new HttpError(422, 'player_id_required');
    const p = await c.query(
      `SELECT p.id, p.jersey_number, CASE WHEN p.team_id=m.home_team_id THEN 'home' WHEN p.team_id=m.away_team_id THEN 'away' END AS side
       FROM players p, matches m WHERE p.id=$1 AND m.id=$2`, [b.player_id, job.match_id]);
    if (!p.rows[0] || !p.rows[0].side) throw new HttpError(422, 'player_not_in_this_match');
    await c.query(
      `INSERT INTO identity_candidates(job_id,track_id,team,player_id,jersey_number,confidence,status,evidence)
       VALUES ($1,$2,$3,$4,$5,1,'confirmed','{"confirmed_by":"user"}')
       ON CONFLICT (job_id,track_id) DO UPDATE SET player_id=$4, team=$3, jersey_number=$5, confidence=1, status='confirmed',
         evidence = identity_candidates.evidence || '{"confirmed_by":"user"}'::jsonb`,
      [job.id, trackId, p.rows[0].side, b.player_id, p.rows[0].jersey_number]);
    await c.query('UPDATE player_tracks SET player_id=$3, team=$4 WHERE job_id=$1 AND track_id=$2', [job.id, trackId, b.player_id, p.rows[0].side]);
    if (b.learn) {
      await c.query(`UPDATE players SET reference_embedding=(SELECT appearance FROM track_features WHERE job_id=$1 AND track_id=$2) WHERE id=$3`, [job.id, trackId, b.player_id]);
    }
    return { track_id: trackId, status: 'confirmed', player_id: b.player_id, team: p.rows[0].side };
  });
  res.json({ ...out, recompute_recommended: true });
}));

// ---- review a detected event ----
const eventPatch = z.object({
  status: z.enum(['confirmed', 'rejected', 'needs_review']).optional(),
  subtype: z.string().max(40).optional(),
  track_id: z.number().int().optional(),
  target_track_id: z.number().int().nullable().optional(),
}).strict();

r.patch('/events/:id', asyncH(async (req, res) => {
  const { rows } = await db.query('SELECT job_id FROM events WHERE id=$1', [req.params.id]);
  if (!rows[0]) throw new HttpError(404, 'event_not_found');
  await getJob(req, rows[0].job_id);
  const b = eventPatch.parse(req.body);
  const keys = Object.keys(b);
  if (!keys.length) throw new HttpError(422, 'nothing_to_update');
  const sets = keys.map((k, i) => `${k}=$${i + 2}`).join(',');
  const up = await db.query(`UPDATE events SET ${sets} WHERE id=$1 RETURNING *`, [req.params.id, ...keys.map((k) => b[k])]);
  res.json({ event: up.rows[0], recompute_recommended: true });
}));

// ---- recompute (no video processing) ----
async function startRecompute(job, fromStage, skipAi) {
  if (!RECOMPUTE_STAGES.includes(fromStage)) throw new HttpError(422, 'invalid_from_stage', RECOMPUTE_STAGES.join(', '));
  if (job.status !== 'completed') throw new HttpError(409, 'job_not_completed', job.status);
  await db.query(
    `UPDATE analysis_jobs SET status='recomputing', error=NULL, cancel_requested=FALSE, progress=0,
       config = config || $2::jsonb WHERE id=$1`,
    [job.id, JSON.stringify({ recompute: { from_stage: fromStage, skip_ai: !!skipAi } })]);
  await enqueue(job.id);
}

r.post('/jobs/:id/recompute', asyncH(async (req, res) => {
  const job = await getJob(req, req.params.id);
  const b = z.object({ from_stage: z.string().default('statistics'), skip_ai: z.boolean().optional() }).parse(req.body || {});
  await startRecompute(job, b.from_stage, b.skip_ai);
  res.status(202).json({ job_id: job.id, status: 'recomputing', from_stage: b.from_stage });
}));
r.post('/matches/:id/recompute', asyncH(async (req, res) => {
  const job = await resolveJob(req, req.params.id, req.query.job_id);
  const b = z.object({ from_stage: z.string().default('statistics'), skip_ai: z.boolean().optional() }).parse(req.body || {});
  await startRecompute(job, b.from_stage, b.skip_ai);
  res.status(202).json({ job_id: job.id, status: 'recomputing', from_stage: b.from_stage });
}));

/**
 * The kit-cluster -> home/away mapping is a guess unless both teams have a kit colour.
 * If it is the wrong way round, swap it and re-run identity onward (identity matching depends on the roster of the team).
 */
r.post('/jobs/:id/swap-teams', asyncH(async (req, res) => {
  const job = await getJob(req, req.params.id);
  if (job.status !== 'completed') throw new HttpError(409, 'job_not_completed', job.status);
  await db.tx(async (c) => {
    const flip = `CASE team WHEN 'home' THEN 'away' WHEN 'away' THEN 'home' ELSE team END`;
    await c.query(`UPDATE player_tracks SET team=${flip}, player_id=NULL WHERE job_id=$1`, [job.id]);
    await c.query('DELETE FROM identity_candidates WHERE job_id=$1', [job.id]);
    await c.query(`UPDATE events SET team=${flip} WHERE job_id=$1`, [job.id]);
    await c.query(`UPDATE analysis_jobs SET summary = summary || '{"team_mapping":"swapped_by_user"}'::jsonb WHERE id=$1`, [job.id]);
  });
  await startRecompute(job, 'direction', req.body?.skip_ai);
  res.status(202).json({ job_id: job.id, status: 'recomputing', from_stage: 'direction' });
}));

module.exports = r;
