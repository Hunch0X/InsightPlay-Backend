const Redis = require('ioredis');
const config = require('./config');

const redis = new Redis(config.redisUrl, { maxRetriesPerRequest: null });
redis.on('error', (e) => console.error('[redis]', e.message));

/** Push a job id onto the FIFO queue consumed by the Python worker (BLPOP). */
const enqueue = (jobId) => redis.rpush(config.queueKey, jobId);

/** Remove a still-queued job (used for cancel). */
const dequeue = (jobId) => redis.lrem(config.queueKey, 0, jobId);

/** Subscribe to worker progress messages (JSON) on a dedicated connection. */
function subscribeProgress(handler) {
  const sub = new Redis(config.redisUrl, { maxRetriesPerRequest: null });
  sub.on('error', (e) => console.error('[redis:sub]', e.message));
  sub.subscribe(config.progressChannel);
  sub.on('message', (_ch, raw) => {
    try { handler(JSON.parse(raw)); } catch (e) { console.error('[progress] bad message', e.message); }
  });
  return sub;
}

module.exports = { redis, enqueue, dequeue, subscribeProgress };
