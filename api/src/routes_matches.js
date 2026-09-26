const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const { execFile } = require('child_process');
const { z } = require('zod');
const db = require('./db');
const config = require('./config');
const { enqueue, dequeue } = require('./queue');
const { asyncH, HttpError, getMatch, getJob } = require('./helpers');

const r = express.Router();

// ---------- matches ----------
const matchSchema = z.object({
  home_team_id: z.string().uuid().optional(),
  away_team_id: z.string().uuid().optional(),
  match_date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).optional(),
  competition: z.string().max(120).optional(),
  pitch_length: z.number().min(90).max(120).optional(),
  pitch_width: z.number().min(45).max(90).optional(),
  reported_home_score: z.number().int().min(0).max(30).optional(),
  reported_away_score: z.number().int().min(0).max(30).optional(),
});

r.post('/matches', asyncH(async (req, res) => {
  const b = matchSchema.parse(req.body);
  if (b.home_team_id && b.home_team_id === b.away_team_id) throw new HttpError(422, 'home_and_away_must_differ');
  const { rows } = await db.query(
    `INSERT INTO matches(owner_id,home_team_id,away_team_id,match_date,competition,pitch_length,pitch_width,reported_home_score,reported_away_score)
     VALUES ($1,$2,$3,$4,$5,COALESCE($6,105),COALESCE($7,68),$8,$9) RETURNING *`,
    [req.user?.sub || null, b.home_team_id || null, b.away_team_id || null, b.match_date || null, b.competition || null,
     b.pitch_length ?? null, b.pitch_width ?? null, b.reported_home_score ?? null, b.reported_away_score ?? null]);
  res.status(201).json(rows[0]);
}));

r.patch('/matches/:id', asyncH(async (req, res) => {
  await getMatch(req, req.params.id);
  const b = matchSchema.partial().parse(req.body);
  const keys = Object.keys(b);
  if (!keys.length) throw new HttpError(422, 'nothing_to_update');
  const sets = keys.map((k, i) => `${k}=$${i + 2}`).join(',');
  const { rows } = await db.query(`UPDATE matches SET ${sets} WHERE id=$1 RETURNING *`, [req.params.id, ...keys.map((k) => b[k])]);
  res.json(rows[0]);
}));

r.get('/matches', asyncH(async (req, res) => {
  const params = [];
  let where = '';
  if (config.authRequired) { params.push(req.user.sub); where = 'WHERE m.owner_id=$1'; }
  const { rows } = await db.query(
    `SELECT m.*, ht.name AS home_team, at.name AS away_team,
            (SELECT status FROM analysis_jobs j WHERE j.match_id=m.id ORDER BY created_at DESC LIMIT 1) AS latest_job_status
     FROM matches m LEFT JOIN teams ht ON ht.id=m.home_team_id LEFT JOIN teams at ON at.id=m.away_team_id
     ${where} ORDER BY m.created_at DESC LIMIT 200`, params);
  res.json(rows);
}));

r.get('/matches/:id', asyncH(async (req, res) => {
  const m = await getMatch(req, req.params.id);
  const [videos, jobs] = await Promise.all([
    db.query('SELECT id,original_name,camera_type,static_camera,fps,duration_s,width,height,size_bytes,(calibration_points IS NOT NULL) AS calibrated_manually,created_at FROM video_assets WHERE match_id=$1 ORDER BY created_at', [m.id]),
    db.query('SELECT id,type,status,stage,progress,created_at,finished_at FROM analysis_jobs WHERE match_id=$1 ORDER BY created_at DESC', [m.id]),
  ]);
  res.json({ ...m, videos: videos.rows, jobs: jobs.rows });
}));

// ---------- video upload ----------
fs.mkdirSync(config.uploadDir, { recursive: true });
const ALLOWED_EXT = new Set(['.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.ts']);
const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, config.uploadDir),
    filename: (_req, file, cb) => cb(null, `${Date.now()}-${crypto.randomBytes(6).toString('hex')}${path.extname(file.originalname).toLowerCase()}`),
  }),
  limits: { fileSize: config.maxUploadBytes, files: 1 },
  fileFilter: (_req, file, cb) => (ALLOWED_EXT.has(path.extname(file.originalname).toLowerCase())
    ? cb(null, true) : cb(new HttpError(415, 'unsupported_video_type', [...ALLOWED_EXT].join(', ')))),
});

function probe(file) {
  return new Promise((resolve, reject) => {
    execFile('ffprobe', ['-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', file],
      { maxBuffer: 20 * 1024 * 1024 }, (err, stdout) => {
        if (err) return reject(err);
        try {
          const j = JSON.parse(stdout);
          const v = (j.streams || []).find((s) => s.codec_type === 'video');
          if (!v) return reject(new Error('no video stream'));
          const [n, d] = String(v.avg_frame_rate && v.avg_frame_rate !== '0/0' ? v.avg_frame_rate : v.r_frame_rate).split('/').map(Number);
          const fps = d ? n / d : null;
          const duration = Number(v.duration || j.format?.duration) || null;
          resolve({
            fps, duration, width: v.width, height: v.height, codec: v.codec_name,
            frames: Number(v.nb_frames) || (fps && duration ? Math.round(fps * duration) : null),
          });
        } catch (e) { reject(e); }
      });
  });
}

r.post('/matches/:id/videos', asyncH(async (req, _res, next) => { await getMatch(req, req.params.id); next(); }),
  upload.single('video'),
  asyncH(async (req, res) => {
    if (!req.file) throw new HttpError(422, 'video_file_required', 'Send multipart field "video".');
    const body = z.object({
      camera_type: z.enum(['tactical', 'broadcast', 'training']).default('broadcast'),
      static_camera: z.enum(['true', 'false']).default('false'),
    }).parse(req.body);
    let meta;
    try { meta = await probe(req.file.path); }
    catch (e) { fs.unlink(req.file.path, () => {}); throw new HttpError(422, 'unreadable_video', 'The file is not a readable video (ffprobe could not parse it).'); }
    const { rows } = await db.query(
      `INSERT INTO video_assets(match_id,path,original_name,camera_type,static_camera,fps,duration_s,width,height,frame_count,codec,size_bytes)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) RETURNING *`,
      [req.params.id, path.resolve(req.file.path), req.file.originalname, body.camera_type, body.static_camera === 'true',
       meta.fps, meta.duration, meta.width, meta.height, meta.frames, meta.codec, req.file.size]);
    res.status(201).json(rows[0]);
  }));

/**
 * Manual pitch calibration for a fixed camera: >= 4 pixel<->pitch correspondences.
 * Pitch coordinates are metres: x along the length (0..pitch_length), y across (0..pitch_width), origin at a corner.
 */
r.put('/videos/:id/calibration', asyncH(async (req, res) => {
  const { rows: vr } = await db.query('SELECT v.*, m.pitch_length, m.pitch_width FROM video_assets v JOIN matches m ON m.id=v.match_id WHERE v.id=$1', [req.params.id]);
  const v = vr[0];
  if (!v) throw new HttpError(404, 'video_not_found');
  await getMatch(req, v.match_id);
  const pt = z.object({ px: z.number().min(0), py: z.number().min(0), x: z.number(), y: z.number() });
  const { points } = z.object({ points: z.array(pt).min(4).max(40) }).parse(req.body);
  for (const p of points) {
    if (p.px > v.width || p.py > v.height) throw new HttpError(422, 'pixel_outside_frame', `${v.width}x${v.height}`);
    if (p.x < -1 || p.x > v.pitch_length + 1 || p.y < -1 || p.y > v.pitch_width + 1) throw new HttpError(422, 'point_outside_pitch', `${v.pitch_length}x${v.pitch_width} m`);
  }
  await db.query('UPDATE video_assets SET calibration_points=$2, static_camera=TRUE WHERE id=$1', [v.id, JSON.stringify(points)]);
  res.json({ video_id: v.id, points: points.length, static_camera: true });
}));

// ---------- analysis jobs ----------
const analyzeSchema = z.object({
  video_id: z.string().uuid().optional(),
  analysis_fps: z.number().min(1).max(30).optional(),
  start_s: z.number().min(0).optional(),
  end_s: z.number().min(1).optional(),
  periods: z.array(z.object({
    start_s: z.number().min(0), end_s: z.number().min(1),
    home_attacks: z.enum(['left', 'right']).optional(),   // pitch x=0 is "left"
  })).max(4).optional(),
  skip_ai: z.boolean().optional(),
}).strict();

r.post('/matches/:id/analyze', asyncH(async (req, res) => {
  const m = await getMatch(req, req.params.id);
  const cfg = analyzeSchema.parse(req.body || {});
  if (cfg.periods) for (const p of cfg.periods) if (p.end_s <= p.start_s) throw new HttpError(422, 'invalid_period');
  const { rows: vr } = cfg.video_id
    ? await db.query('SELECT id FROM video_assets WHERE id=$1 AND match_id=$2', [cfg.video_id, m.id])
    : await db.query('SELECT id FROM video_assets WHERE match_id=$1 ORDER BY created_at DESC LIMIT 1', [m.id]);
  if (!vr[0]) throw new HttpError(422, 'no_video', 'Upload a video first: POST /api/matches/:id/videos');
  const busy = await db.query(`SELECT id FROM analysis_jobs WHERE match_id=$1 AND status IN ('queued','running','recomputing')`, [m.id]);
  if (busy.rows[0]) throw new HttpError(409, 'analysis_already_running', busy.rows[0].id);
  const { video_id, ...config_ } = cfg;
  const { rows } = await db.query(
    `INSERT INTO analysis_jobs(match_id,video_id,config) VALUES ($1,$2,$3) RETURNING *`, [m.id, vr[0].id, JSON.stringify(config_)]);
  await db.query(`UPDATE matches SET status='processing' WHERE id=$1`, [m.id]);
  await enqueue(rows[0].id);
  res.status(202).json({ job_id: rows[0].id, status: 'queued', ws: `/ws/jobs/${rows[0].id}` });
}));

r.get('/jobs/:id', asyncH(async (req, res) => {
  const j = await getJob(req, req.params.id);
  const { config: _c, ...rest } = j;
  res.json({ ...rest, config: j.config });
}));

r.delete('/jobs/:id', asyncH(async (req, res) => {
  const j = await getJob(req, req.params.id);
  if (j.status === 'queued') {
    await dequeue(j.id);
    await db.query(`UPDATE analysis_jobs SET status='cancelled', finished_at=now() WHERE id=$1`, [j.id]);
    return res.json({ id: j.id, status: 'cancelled' });
  }
  if (['running', 'recomputing'].includes(j.status)) {
    await db.query('UPDATE analysis_jobs SET cancel_requested=TRUE WHERE id=$1', [j.id]);
    return res.status(202).json({ id: j.id, status: j.status, cancel_requested: true });
  }
  throw new HttpError(409, 'job_not_cancellable', j.status);
}));

module.exports = r;
