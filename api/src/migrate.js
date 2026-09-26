const fs = require('fs');
const path = require('path');
const { pool } = require('./db');

async function migrate() {
  const dir = path.join(__dirname, '..', 'migrations');
  await pool.query(`CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())`);
  const done = new Set((await pool.query('SELECT name FROM schema_migrations')).rows.map((r) => r.name));
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.sql')).sort();
  for (const f of files) {
    if (done.has(f)) continue;
    const sql = fs.readFileSync(path.join(dir, f), 'utf8');
    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      await client.query(sql);
      await client.query('INSERT INTO schema_migrations(name) VALUES ($1)', [f]);
      await client.query('COMMIT');
      console.log(`[migrate] applied ${f}`);
    } catch (e) {
      await client.query('ROLLBACK');
      throw new Error(`Migration ${f} failed: ${e.message}`);
    } finally {
      client.release();
    }
  }
}

module.exports = { migrate };
if (require.main === module) {
  migrate().then(() => pool.end()).catch((e) => { console.error(e); process.exit(1); });
}
