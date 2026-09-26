const { Pool, types } = require('pg');
const config = require('./config');

// Return BIGINT/COUNT as numbers (safe for our value ranges) and keep REAL as float.
types.setTypeParser(20, (v) => Number(v));
types.setTypeParser(700, (v) => parseFloat(v));

const pool = new Pool({ connectionString: config.databaseUrl, max: 10 });
pool.on('error', (err) => console.error('[pg] idle client error', err.message));

const query = (text, params) => pool.query(text, params);

async function tx(fn) {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    const out = await fn(client);
    await client.query('COMMIT');
    return out;
  } catch (e) {
    await client.query('ROLLBACK');
    throw e;
  } finally {
    client.release();
  }
}

module.exports = { pool, query, tx };
