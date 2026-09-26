const express = require('express');
const { z } = require('zod');
const db = require('./db');
const config = require('./config');
const { asyncH, HttpError } = require('./helpers');

const r = express.Router();
const teamSchema = z.object({ name: z.string().min(1).max(120), kit_color: z.string().regex(/^#[0-9a-fA-F]{6}$/).optional() });
const playerSchema = z.object({
  name: z.string().min(1).max(120),
  jersey_number: z.number().int().min(0).max(99).optional(),
  position: z.string().max(40).optional(),
});

r.post('/teams', asyncH(async (req, res) => {
  const b = teamSchema.parse(req.body);
  const { rows } = await db.query('INSERT INTO teams(name,kit_color,owner_id) VALUES ($1,$2,$3) RETURNING *', [b.name, b.kit_color || null, req.user?.sub || null]);
  res.status(201).json(rows[0]);
}));

r.get('/teams', asyncH(async (req, res) => {
  const params = [];
  let where = '';
  if (config.authRequired) { params.push(req.user.sub); where = 'WHERE owner_id=$1'; }
  const { rows } = await db.query(`SELECT * FROM teams ${where} ORDER BY name`, params);
  res.json(rows);
}));

r.patch('/teams/:id', asyncH(async (req, res) => {
  const b = teamSchema.partial().parse(req.body);
  const { rows } = await db.query('UPDATE teams SET name=COALESCE($2,name), kit_color=COALESCE($3,kit_color) WHERE id=$1 RETURNING *', [req.params.id, b.name || null, b.kit_color || null]);
  if (!rows[0]) throw new HttpError(404, 'team_not_found');
  res.json(rows[0]);
}));

r.get('/teams/:id/players', asyncH(async (req, res) => {
  const { rows } = await db.query('SELECT id,team_id,name,jersey_number,position FROM players WHERE team_id=$1 ORDER BY jersey_number NULLS LAST, name', [req.params.id]);
  res.json(rows);
}));

/** Accepts one player or an array (bulk squad upload). Jersey numbers must be unique inside a team. */
r.post('/teams/:id/players', asyncH(async (req, res) => {
  const list = z.union([playerSchema, z.array(playerSchema).min(1).max(80)]).parse(req.body);
  const items = Array.isArray(list) ? list : [list];
  const nums = items.map((p) => p.jersey_number).filter((n) => n != null);
  if (new Set(nums).size !== nums.length) throw new HttpError(422, 'duplicate_jersey_numbers');
  const out = await db.tx(async (c) => {
    const t = await c.query('SELECT id FROM teams WHERE id=$1', [req.params.id]);
    if (!t.rows[0]) throw new HttpError(404, 'team_not_found');
    const created = [];
    for (const p of items) {
      const { rows } = await c.query('INSERT INTO players(team_id,name,jersey_number,position) VALUES ($1,$2,$3,$4) RETURNING id,team_id,name,jersey_number,position',
        [req.params.id, p.name, p.jersey_number ?? null, p.position || null]);
      created.push(rows[0]);
    }
    return created;
  });
  res.status(201).json(out);
}));

module.exports = r;
