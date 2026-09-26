const express = require('express');
const bcrypt = require('bcryptjs');
const { z } = require('zod');
const db = require('./db');
const { sign } = require('./auth');
const { asyncH, HttpError } = require('./helpers');

const r = express.Router();
const creds = z.object({ email: z.string().email(), password: z.string().min(8).max(200) });

r.post('/register', asyncH(async (req, res) => {
  const { email, password } = creds.parse(req.body);
  const hash = await bcrypt.hash(password, 10);
  try {
    const { rows } = await db.query('INSERT INTO users(email,password_hash) VALUES ($1,$2) RETURNING id,email,role', [email.toLowerCase(), hash]);
    res.status(201).json({ user: rows[0], token: sign(rows[0]) });
  } catch (e) {
    if (e.code === '23505') throw new HttpError(409, 'email_taken');
    throw e;
  }
}));

r.post('/login', asyncH(async (req, res) => {
  const { email, password } = creds.parse(req.body);
  const { rows } = await db.query('SELECT * FROM users WHERE email=$1', [email.toLowerCase()]);
  const u = rows[0];
  if (!u || !(await bcrypt.compare(password, u.password_hash))) throw new HttpError(401, 'invalid_credentials');
  res.json({ user: { id: u.id, email: u.email, role: u.role }, token: sign(u) });
}));

module.exports = r;
