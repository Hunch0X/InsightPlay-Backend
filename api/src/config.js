require('dotenv').config();
const env = process.env;

const config = {
  port: Number(env.PORT || 4000),
  databaseUrl: env.DATABASE_URL || 'postgres://insightplay:insightplay@localhost:5432/insightplay',
  redisUrl: env.REDIS_URL || 'redis://localhost:6379',
  jwtSecret: env.JWT_SECRET || 'dev-only-change-me',
  authRequired: (env.AUTH_REQUIRED || 'false') === 'true',
  uploadDir: env.UPLOAD_DIR || './uploads',
  maxUploadBytes: Number(env.MAX_UPLOAD_GB || 10) * 1024 ** 3,
  corsOrigin: env.CORS_ORIGIN || '*',
  autoMigrate: (env.AUTO_MIGRATE || 'true') === 'true',
  queueKey: env.QUEUE_KEY || 'insightplay:jobs',
  progressChannel: 'insightplay:progress',
};

if (config.authRequired && config.jwtSecret === 'dev-only-change-me') {
  throw new Error('AUTH_REQUIRED=true but JWT_SECRET is not set. Refusing to start with the default secret.');
}
module.exports = config;
