const express = require('express');
const db = require('./db');
const { asyncH, HttpError, resolveJob, entityKeyFromParam } = require('./helpers');

const r = express.Router();
const mv = (m, k) => (m && m[k] ? m[k].value ?? null : null);   // value of a metric object
const NA = (reason) => ({ value: null, status: 'not_available', reason });

const jobBlock = (j) => ({ id: j.id, status: j.status, finished_at: j.finished_at, summary: j.summary, model_versions: j.model_versions });

async function teamNames(matchId) {
  const { rows } = await db.query(
    `SELECT m.*, ht.name AS home_name, at.name AS away_name FROM matches m
     LEFT JOIN teams ht ON ht.id=m.home_team_id LEFT JOIN teams at ON at.id=m.away_team_id WHERE m.id=$1`, [matchId]);
  return rows[0];
}

// ---------------- match summary ----------------
r.get('/matches/:id/summary', asyncH(async (req, res) => {
  const job = await resolveJob(req, req.params.id, req.query.job_id);
  const m = await teamNames(req.params.id);
  const [ts, goals, ai] = await Promise.all([
    db.query('SELECT team, metrics FROM team_match_stats WHERE job_id=$1', [job.id]),
    db.query(`SELECT team, count(*)::int AS n, bool_or(status='needs_review') AS unreviewed FROM events
              WHERE job_id=$1 AND type='goal' AND status<>'rejected' GROUP BY team`, [job.id]),
    db.query(`SELECT subject_key, content, dropped, model, created_at FROM ai_analyses WHERE job_id=$1 AND scope='team'`, [job.id]),
  ]);
  const teams = {};
  for (const side of ['home', 'away']) {
    const row = ts.rows.find((x) => x.team === side);
    teams[side] = { name: side === 'home' ? m.home_name : m.away_name, metrics: row?.metrics || null };
  }
  const eventsRan = !!job.summary?.events;             // "0 goals" is only true if events were actually computed
  const g = (side) => (eventsRan ? (goals.rows.find((x) => x.team === side)?.n ?? 0) : null);
  res.json({
    match: { id: m.id, date: m.match_date, competition: m.competition, home: m.home_name, away: m.away_name },
    job: jobBlock(job),
    score: {
      detected: { home: g('home'), away: g('away'), source: 'derived', ...(eventsRan ? {} : { status: 'not_available', reason: 'events_not_computed' }),
        note: 'Counted from detected goal events. Goals are flagged for review; confirm them via PATCH /api/events/:id.',
        needs_review: goals.rows.some((x) => x.unreviewed) },
      reported: { home: m.reported_home_score, away: m.reported_away_score },
    },
    teams,
    ai_reports: Object.fromEntries(ai.rows.map((a) => [a.subject_key, { content: a.content, model: a.model, created_at: a.created_at }])),
  });
}));

// ---------------- players table ----------------
function playerRow(s) {
  const a = s.attributes || {};
  const m = s.metrics || {};
  return {
    key: s.player_id || s.entity_key,
    entity_key: s.entity_key,
    player_id: s.player_id,
    name: s.name || `Unidentified (${s.team || 'unknown'} track ${String(s.entity_key).replace('track:', '')})`,
    jersey_number: s.jersey_number ?? null,
    position: s.position || null,
    team: s.team,
    role: s.role,
    identity: { confidence: s.identity_confidence, status: s.identity_status },
    columns: {
      RAT: mv(m, 'rating'), G: mv(m, 'goals'), A: mv(m, 'assists'), 'PAS%': mv(m, 'pass_accuracy_pct'),
      DIS: mv(m, 'distance_km'), DEF: a.DEF?.value ?? null, PHY: a.PHY?.value ?? null,
      PAC: a.PAC?.value ?? null, SHO: a.SHO?.value ?? null, PAS: a.PAS?.value ?? null, DRI: a.DRI?.value ?? null,
      MIN: mv(m, 'minutes'),
    },
    metrics: m,
    attributes: a,
  };
}

r.get('/matches/:id/players', asyncH(async (req, res) => {
  const job = await resolveJob(req, req.params.id, req.query.job_id);
  const { rows } = await db.query(
    `SELECT s.job_id, s.entity_key, s.player_id, s.team, s.role, s.identity_confidence, s.identity_status, s.metrics, s.attributes,
            p.name, p.jersey_number, p.position
     FROM player_match_stats s LEFT JOIN players p ON p.id=s.player_id
     WHERE s.job_id=$1 AND s.role<>'referee' ${req.query.team ? 'AND s.team=$2' : ''}`,
    req.query.team ? [job.id, req.query.team] : [job.id]);
  let list = rows.map(playerRow);
  if (req.query.identified === 'true') list = list.filter((p) => p.player_id);
  list.sort((a, b) => (b.columns.RAT ?? -1) - (a.columns.RAT ?? -1));
  res.json({ job: jobBlock(job), players: list });
}));

// ---------------- player page ----------------
async function playerDetail(req, res) {
  const matchId = req.params.matchId || req.params.id;
  const key = entityKeyFromParam(req.params.playerKey || req.params.playerId);
  const job = await resolveJob(req, matchId, req.query.job_id);
  const { rows } = await db.query(
    `SELECT s.*, p.name, p.jersey_number, p.position, t.name AS team_name
     FROM player_match_stats s LEFT JOIN players p ON p.id=s.player_id LEFT JOIN teams t ON t.id=p.team_id
     WHERE s.job_id=$1 AND s.entity_key=$2`, [job.id, key]);
  const s = rows[0];
  if (!s) throw new HttpError(404, 'player_not_found_in_analysis');
  const [ev, ai, idc] = await Promise.all([
    db.query(`SELECT id,type,subtype,ts,frame,team,track_id,target_track_id,sx,sy,ex,ey,distance_m,confidence,status,attributes
              FROM events WHERE job_id=$1 AND track_id=ANY($2) AND status<>'rejected' ORDER BY ts`, [job.id, s.track_ids]),
    db.query(`SELECT content,dropped,model,created_at FROM ai_analyses WHERE job_id=$1 AND scope='player' AND subject_key=$2`, [job.id, key]),
    db.query('SELECT track_id,jersey_number,confidence,status,evidence FROM identity_candidates WHERE job_id=$1 AND track_id=ANY($2)', [job.id, s.track_ids]),
  ]);
  const by = (types) => ev.rows.filter((e) => types.includes(e.type));
  res.json({
    job: jobBlock(job),
    player: { ...playerRow(s), team_name: s.team_name },
    identity_tracks: idc.rows,
    heatmap: s.heatmap || NA('camera_not_calibrated'),
    average_position: s.avg_position || NA('camera_not_calibrated'),
    zone_time: s.zone_time || NA('attack_direction_unknown'),
    movement_path: s.movement_path || NA('camera_not_calibrated'),
    pass_map: by(['pass']),
    shot_map: by(['shot', 'goal']),
    carries: by(['carry', 'dribble']),
    defensive_actions: by(['tackle', 'interception', 'clearance', 'block', 'recovery', 'pressure', 'duel']),
    ai_analysis: ai.rows[0] ? { ...ai.rows[0].content, source: 'ai_inference', model: ai.rows[0].model, dropped_statements: ai.rows[0].dropped.length } : NA('ai_analysis_not_generated'),
  });
}
r.get('/matches/:matchId/players/:playerKey', asyncH(playerDetail));
r.get('/players/:playerId/matches/:matchId', asyncH(playerDetail));

// ---------------- team tactics ----------------
r.get('/matches/:id/teams/:side/tactics', asyncH(async (req, res) => {
  const side = req.params.side;
  if (!['home', 'away'].includes(side)) throw new HttpError(422, 'side_must_be_home_or_away');
  const job = await resolveJob(req, req.params.id, req.query.job_id);
  const [ts, ai] = await Promise.all([
    db.query('SELECT metrics, tactics FROM team_match_stats WHERE job_id=$1 AND team=$2', [job.id, side]),
    db.query(`SELECT content,dropped,model,created_at FROM ai_analyses WHERE job_id=$1 AND scope='team' AND subject_key=$2`, [job.id, side]),
  ]);
  if (!ts.rows[0]) throw new HttpError(404, 'team_stats_not_found');
  res.json({
    job: jobBlock(job), team: side, metrics: ts.rows[0].metrics, tactics: ts.rows[0].tactics,
    ai_analysis: ai.rows[0] ? { ...ai.rows[0].content, source: 'ai_inference', model: ai.rows[0].model, dropped_statements: ai.rows[0].dropped.length } : NA('ai_analysis_not_generated'),
  });
}));

// ---------------- events (video timeline) ----------------
r.get('/matches/:id/events', asyncH(async (req, res) => {
  const job = await resolveJob(req, req.params.id, req.query.job_id);
  const where = ['job_id=$1'];
  const params = [job.id];
  const add = (sql, v) => { params.push(v); where.push(sql.replace('?', `$${params.length}`)); };
  if (req.query.type) add('type = ANY(?)', String(req.query.type).split(','));
  if (req.query.team) add('team=?', req.query.team);
  if (req.query.status) add('status=?', req.query.status); else where.push(`status<>'rejected'`);
  if (req.query.from_ts) add('ts>=?', Number(req.query.from_ts));
  if (req.query.to_ts) add('ts<=?', Number(req.query.to_ts));
  if (req.query.track_id) add('track_id=?', Number(req.query.track_id));
  if (req.query.player) {
    const s = await db.query('SELECT track_ids FROM player_match_stats WHERE job_id=$1 AND entity_key=$2', [job.id, entityKeyFromParam(req.query.player)]);
    add('track_id = ANY(?)', s.rows[0]?.track_ids || []);
  }
  const limit = Math.min(Number(req.query.limit) || 500, 5000);
  const offset = Number(req.query.offset) || 0;
  const { rows } = await db.query(
    `SELECT id,type,subtype,ts,frame,team,track_id,target_track_id,sx,sy,ex,ey,distance_m,confidence,status,attributes
     FROM events WHERE ${where.join(' AND ')} ORDER BY ts LIMIT ${limit} OFFSET ${offset}`, params);
  res.json({ job_id: job.id, count: rows.length, events: rows });
}));

module.exports = r;
