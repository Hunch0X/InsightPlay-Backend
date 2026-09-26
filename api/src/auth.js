const jwt = require('jsonwebtoken');
const config = require('./config');

/** Attaches req.user. When AUTH_REQUIRED=false requests pass through as an anonymous user. */
function auth(req, res, next) {
  const header = req.headers.authorization || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : null;
  if (token) {
    try { req.user = jwt.verify(token, config.jwtSecret); return next(); }
    catch { return res.status(401).json({ error: 'invalid_token' }); }
  }
  if (config.authRequired) return res.status(401).json({ error: 'authentication_required' });
  req.user = null;
  next();
}

const sign = (user) => jwt.sign({ sub: user.id, email: user.email, role: user.role }, config.jwtSecret, { expiresIn: '7d' });

module.exports = { auth, sign };
