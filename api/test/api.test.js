// Run with: npm test   (needs Postgres + Redis; uses its own queue key so it never triggers a live worker)
process.env.QUEUE_KEY = 'insightplay:test-jobs';
process.env.AUTO_MIGRATE = 'false';
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const path = require('node:path');
const http = require('node:http');

const config = require('../src/config');
const db = require('../src/db');
const { redis } = require('../src/queue');
const { migrate } = require('../src/migrate');
const { buildApp } = require('../src/index');

let server, base;
const created = { teams: [], matches: [] };
const j = (method, url, body) => fetch(base + url, { method, headers: { 'content-type': 'application/json' }, body: body && JSON.stringify(body) })
  .then(async (r) => ({ status: r.status, body: await r.json().catch(() => null) }));

before(async () => {
  await migrate();
  await redis.del(config.queueKey);
  server = http.createServer(buildApp());
  await new Promise((r) => server.listen(0, r));
  base = `http://127.0.0.1:${server.address().port}/api`;
});

after(async () => {
  for (const m of created.matches) await db.query('DELETE FROM matches WHERE id=$1', [m]);
  for (const t of created.teams) await db.query('DELETE FROM teams WHERE id=$1', [t]);
  await redis.del(config.queueKey);
  await new Promise((r) => server.close(r));
  await db.pool.end();
  redis.disconnect();
});

async function newMatch(extra = {}) {
  const h = (await j('POST', '/teams', { name: 'T-Home', kit_color: '#D62828' })).body;
  const a = (await j('POST', '/teams', { name: 'T-Away', kit_color: '#FFFFFF' })).body;
  created.teams.push(h.id, a.id);
  const m = (await j('POST', '/matches', { home_team_id: h.id, away_team_id: a.id, ...extra })).body;
  created.matches.push(m.id);
  return { m, h, a };
}

test('health reports database and redis', async () => {
  const r = await fetch(base.replace('/api', '/health'));
  assert.equal(r.status, 200);
});

test('squad upload rejects duplicate jersey numbers and accepts a valid squad', async () => {
  const { h } = await newMatch();
  const dup = await j('POST', `/teams/${h.id}/players`, [{ name: 'A', jersey_number: 9 }, { name: 'B', jersey_number: 9 }]);
  assert.equal(dup.status, 422); assert.equal(dup.body.error, 'duplicate_jersey_numbers');
  const ok = await j('POST', `/teams/${h.id}/players`, [{ name: 'A', jersey_number: 9 }, { name: 'B', jersey_number: 10 }]);
  assert.equal(ok.status, 201); assert.equal(ok.body.length, 2);
  assert.equal((await j('GET', `/teams/${h.id}/players`)).body[0].jersey_number, 9);
});

test('validation: bad hex colour, home == away, out-of-range pitch', async () => {
  assert.equal((await j('POST', '/teams', { name: 'X', kit_color: 'red' })).status, 422);
  const { h } = await newMatch();
  assert.equal((await j('POST', '/matches', { home_team_id: h.id, away_team_id: h.id })).body.error, 'home_and_away_must_differ');
  assert.equal((await j('POST', '/matches', { pitch_length: 300 })).status, 422);
  assert.equal((await j('GET', '/matches/not-a-uuid')).status, 422);
});

test('results are 404 (not empty numbers) before any analysis exists', async () => {
  const { m } = await newMatch();
  for (const p of ['summary', 'players', 'events']) {
    const r = await j('GET', `/matches/${m.id}/${p}`);
    assert.equal(r.status, 404); assert.equal(r.body.error, 'no_completed_analysis');
  }
});

test('analyze needs a video, enqueues once, refuses a second concurrent job, and cancel dequeues', async () => {
  const { m } = await newMatch();
  assert.equal((await j('POST', `/matches/${m.id}/analyze`, {})).body.error, 'no_video');
  await db.query(`INSERT INTO video_assets(match_id,path,camera_type,fps,duration_s,width,height) VALUES ($1,'/x.mp4','tactical',25,60,1920,1080)`, [m.id]);
  assert.equal((await j('POST', `/matches/${m.id}/analyze`, { periods: [{ start_s: 10, end_s: 5 }] })).body.error, 'invalid_period');
  assert.equal((await j('POST', `/matches/${m.id}/analyze`, { unknown_option: 1 })).status, 422);          // strict schema
  const a = await j('POST', `/matches/${m.id}/analyze`, { periods: [{ start_s: 0, end_s: 2700, home_attacks: 'right' }] });
  assert.equal(a.status, 202);
  assert.deepEqual(await redis.lrange(config.queueKey, 0, -1), [a.body.job_id]);
  assert.equal((await j('POST', `/matches/${m.id}/analyze`, {})).status, 409);
  const c = await j('DELETE', `/jobs/${a.body.job_id}`);
  assert.equal(c.body.status, 'cancelled');
  assert.deepEqual(await redis.lrange(config.queueKey, 0, -1), []);
  assert.equal((await j('DELETE', `/jobs/${a.body.job_id}`)).status, 409);                                // already finished
});

test('manual calibration validates point count, frame bounds and pitch bounds', async () => {
  const { m } = await newMatch();
  const v = (await db.query(`INSERT INTO video_assets(match_id,path,fps,duration_s,width,height) VALUES ($1,'/x.mp4',25,60,1920,1080) RETURNING id`, [m.id])).rows[0].id;
  const pts = (n, over = {}) => ({ points: Array.from({ length: n }, (_, i) => ({ px: 100 + i * 200, py: 100 + i * 50, x: i * 20, y: 10 + i * 8, ...over })) });
  assert.equal((await j('PUT', `/videos/${v}/calibration`, pts(3))).status, 422);
  assert.equal((await j('PUT', `/videos/${v}/calibration`, pts(4, { px: 5000 }))).body.error, 'pixel_outside_frame');
  assert.equal((await j('PUT', `/videos/${v}/calibration`, pts(4, { x: 500 }))).body.error, 'point_outside_pitch');
  const ok = await j('PUT', `/videos/${v}/calibration`, pts(5));
  assert.equal(ok.status, 200); assert.equal(ok.body.static_camera, true);
});

test('video upload rejects non-video extensions and unreadable files without leaking server paths', async () => {
  const { m } = await newMatch();
  const send = (name, content) => { const f = new FormData(); f.append('video', new Blob([content]), name); return fetch(`${base}/matches/${m.id}/videos`, { method: 'POST', body: f }).then(async (r) => ({ status: r.status, body: await r.json() })); };
  const txt = await send('notes.txt', 'hello');
  assert.equal(txt.status, 415);
  const fake = await send('fake.mp4', 'this is not a video');
  assert.equal(fake.status, 422); assert.equal(fake.body.error, 'unreadable_video');
  assert.ok(!JSON.stringify(fake.body).includes('/'), 'error must not contain a filesystem path');
});

test('review endpoints: unknown ids and strict bodies', async () => {
  assert.equal((await j('PATCH', '/events/999999999', { status: 'confirmed' })).status, 404);
  const { m } = await newMatch();
  await db.query(`INSERT INTO video_assets(match_id,path,fps,duration_s,width,height) VALUES ($1,'/x.mp4',25,60,1920,1080)`, [m.id]);
  const jobId = (await db.query(`INSERT INTO analysis_jobs(match_id,status) VALUES ($1,'completed') RETURNING id`, [m.id])).rows[0].id;
  const ev = (await db.query(`INSERT INTO events(job_id,type,ts,frame) VALUES ($1,'pass',1,1) RETURNING id`, [jobId])).rows[0].id;
  assert.equal((await j('PATCH', `/events/${ev}`, { confidence: 1 })).status, 422);                       // cannot edit arbitrary columns
  assert.equal((await j('PATCH', `/events/${ev}`, { status: 'rejected' })).body.event.status, 'rejected');
  assert.equal((await j('POST', `/jobs/${jobId}/tracks/1/identity`, { player_id: '00000000-0000-0000-0000-000000000000' })).status, 404);   // track unknown
  assert.equal((await j('POST', `/jobs/${jobId}/recompute`, { from_stage: 'detect_track' })).body.error, 'invalid_from_stage');
  await db.query(`UPDATE analysis_jobs SET status='running' WHERE id=$1`, [jobId]);
  assert.equal((await j('POST', `/jobs/${jobId}/recompute`, {})).body.error, 'job_not_completed');
});

test('a completed job with no events reports the score as not_available, never 0-0', async () => {
  const { m } = await newMatch();
  const jobId = (await db.query(`INSERT INTO analysis_jobs(match_id,status,summary) VALUES ($1,'completed','{}') RETURNING id`, [m.id])).rows[0].id;
  let s = (await j('GET', `/matches/${m.id}/summary`)).body.score.detected;
  assert.equal(s.home, null); assert.equal(s.reason, 'events_not_computed');
  await db.query(`UPDATE analysis_jobs SET summary='{"events":{"total":0}}' WHERE id=$1`, [jobId]);
  s = (await j('GET', `/matches/${m.id}/summary`)).body.score.detected;
  assert.equal(s.home, 0); assert.equal(s.away, 0);                                                      // events ran and found no goals
});

// ---------- auth, in its own process because config is read at start-up ----------
function startServer(env) {
  return new Promise((resolve, reject) => {
    const p = spawn('node', ['src/index.js'], { cwd: path.join(__dirname, '..'), env: { ...process.env, AUTO_MIGRATE: 'false', ...env } });
    let out = '';
    p.stdout.on('data', (d) => { out += d; if (out.includes('listening')) resolve(p); });
    p.stderr.on('data', (d) => { out += d; });
    p.on('exit', (code) => reject(new Error(`exited ${code}: ${out}`)));
  });
}

test('AUTH_REQUIRED=true refuses to start with the default secret', async () => {
  await assert.rejects(startServer({ AUTH_REQUIRED: 'true', JWT_SECRET: '', PORT: '4131' }), /JWT_SECRET/);
});

test('auth: unauthenticated is 401 and one user cannot read another user\'s match', async () => {
  const p = await startServer({ AUTH_REQUIRED: 'true', JWT_SECRET: 'test-secret-123', PORT: '4132' });
  const b = 'http://127.0.0.1:4132/api';
  const call = async (method, url, body, token) => { const r = await fetch(b + url, { method, headers: { 'content-type': 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) }, body: body && JSON.stringify(body) }); return { status: r.status, body: await r.json() }; };
  try {
    assert.equal((await call('GET', '/matches')).status, 401);
    const mail = (n) => `t${Date.now()}${n}@example.com`;
    const u1 = (await call('POST', '/auth/register', { email: mail(1), password: 'password-one' })).body;
    const u2 = (await call('POST', '/auth/register', { email: mail(2), password: 'password-two' })).body;
    assert.equal((await call('POST', '/auth/login', { email: u1.user.email, password: 'wrong-password' })).status, 401);
    const m = (await call('POST', '/matches', {}, u1.token)).body;
    created.matches.push(m.id);
    assert.equal((await call('GET', `/matches/${m.id}`, null, u1.token)).status, 200);
    assert.equal((await call('GET', `/matches/${m.id}`, null, u2.token)).status, 404);
    assert.equal((await call('GET', '/matches', null, 'garbage')).status, 401);
    await db.query('DELETE FROM users WHERE id = ANY($1)', [[u1.user.id, u2.user.id]]);
  } finally { p.removeAllListeners('exit'); p.kill(); }
});
