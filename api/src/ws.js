const { WebSocketServer } = require('ws');
const jwt = require('jsonwebtoken');
const config = require('./config');
const db = require('./db');
const { subscribeProgress } = require('./queue');

/**
 * WebSocket endpoint:  ws://host/ws/jobs/:jobId[?token=JWT]
 * Sends the current job state immediately, then live worker events:
 * stage_started | progress | stage_completed | needs_review | completed | failed
 */
function attachWebSocket(server) {
  const wss = new WebSocketServer({ noServer: true });
  const rooms = new Map(); // jobId -> Set<ws>

  server.on('upgrade', async (req, socket, head) => {
    try {
      const url = new URL(req.url, 'http://x');
      const m = url.pathname.match(/^\/ws\/jobs\/([0-9a-f-]{36})$/i);
      if (!m) return socket.destroy();
      if (config.authRequired) jwt.verify(url.searchParams.get('token') || '', config.jwtSecret);
      wss.handleUpgrade(req, socket, head, (ws) => onConnect(ws, m[1]));
    } catch {
      socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n');
      socket.destroy();
    }
  });

  async function onConnect(ws, jobId) {
    if (!rooms.has(jobId)) rooms.set(jobId, new Set());
    rooms.get(jobId).add(ws);
    ws.on('close', () => { rooms.get(jobId)?.delete(ws); if (!rooms.get(jobId)?.size) rooms.delete(jobId); });
    const { rows } = await db.query(
      'SELECT id, status, stage, progress, error, summary FROM analysis_jobs WHERE id=$1', [jobId]);
    ws.send(JSON.stringify({ event: 'state', job: rows[0] || null }));
  }

  subscribeProgress((msg) => {
    const room = rooms.get(msg.job_id);
    if (!room) return;
    const data = JSON.stringify(msg);
    for (const ws of room) if (ws.readyState === 1) ws.send(data);
  });

  return wss;
}

module.exports = { attachWebSocket };
