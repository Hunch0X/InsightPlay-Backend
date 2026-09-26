const http = require('http');
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const { ZodError } = require('zod');
const config = require('./config');
const db = require('./db');
const { redis } = require('./queue');
const { migrate } = require('./migrate');
const { auth } = require('./auth');
const { attachWebSocket } = require('./ws');
const { HttpError } = require('./helpers');

function buildApp() {
  const app = express();
  app.use(helmet());
  app.use(cors({ origin: config.corsOrigin === '*' ? true : config.corsOrigin.split(','), credentials: true }));
  app.use(express.json({ limit: '2mb' }));

  app.get('/health', async (_req, res) => {
    try {
      await db.query('SELECT 1');
      await redis.ping();
      res.json({ status: 'ok' });
    } catch (e) { res.status(503).json({ status: 'unavailable', error: e.message }); }
  });

  app.use('/api/auth', require('./routes_auth'));
  app.use('/api', auth, require('./routes_library'), require('./routes_matches'), require('./routes_results'), require('./routes_review'));

  app.use((_req, res) => res.status(404).json({ error: 'not_found' }));
  // eslint-disable-next-line no-unused-vars
  app.use((err, _req, res, _next) => {
    if (err instanceof ZodError) return res.status(422).json({ error: 'validation_failed', issues: err.issues.map((i) => ({ path: i.path.join('.'), message: i.message })) });
    if (err instanceof HttpError) return res.status(err.status).json({ error: err.code, detail: err.detail });
    if (err.code === 'LIMIT_FILE_SIZE') return res.status(413).json({ error: 'file_too_large' });
    if (err.code === '22P02') return res.status(422).json({ error: 'invalid_identifier' });
    console.error('[error]', err);
    res.status(500).json({ error: 'internal_error' });
  });
  return app;
}

async function main() {
  if (config.autoMigrate) await migrate();
  const app = buildApp();
  const server = http.createServer(app);
  attachWebSocket(server);
  server.requestTimeout = 0;          // large uploads
  server.listen(config.port, () => console.log(`[api] listening on :${config.port}`));
  const stop = () => server.close(async () => { await db.pool.end(); redis.disconnect(); process.exit(0); });
  process.on('SIGTERM', stop);
  process.on('SIGINT', stop);
}

module.exports = { buildApp };
if (require.main === module) main().catch((e) => { console.error(e); process.exit(1); });
