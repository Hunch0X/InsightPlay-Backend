# InsightPlay backend

Football video analysis: upload a match video, get tracked players, a ball track, events, physical and tactical
statistics, and AI commentary grounded in those numbers.

```
frontend ──HTTP/WS──▶ api (Node/Express) ──Redis queue──▶ worker (Python: YOLO26 + BoT-SORT + analytics)
                          │                                   │
                          └──────────── PostgreSQL ◀──────────┘        Gemini (interpretation only)
```

* **api/** — auth, teams/rosters, uploads, jobs, results, review endpoints, WebSocket progress.
* **worker/** — the CV + analytics pipeline. One job = one video. Each stage is one DB transaction, so a failed stage leaves
  no partial output and a retry resumes at the first unfinished stage.

## Design rules (enforced in code and tests)

1. **Never invent a number.** A metric that cannot be measured is `{"value": null, "status": "not_available", "reason": "..."}`.
   Unknown is never `0`. A score of `0 - 0` is only reported if the event stage actually ran.
2. Every metric carries `source` (`derived` / `estimated`), `confidence` and `coverage_pct`.
3. Uncertain identities and events are **not applied silently**: they are queued for human review.
4. Gemini receives numbered facts and must cite them. Statements whose numbers don't match the cited facts are dropped.
5. Pixel distances are never presented as metres. Without pitch calibration there is no distance, speed, heatmap or zone data.

## Quick start

```bash
cp .env.example .env            # set JWT_SECRET and AUTH_REQUIRED=true for anything but local dev
docker compose up --build       # postgres, redis, api :4000, worker
```

Without Docker: run Postgres 16 and Redis, then

```bash
cd api    && npm install && npm start                     # migrates the schema on start
cd worker && pip install -r requirements.txt && python main.py
```

### Typical flow

```bash
A=http://localhost:4000/api
curl -X POST $A/teams   -H 'content-type: application/json' -d '{"name":"Gor Mahia","kit_color":"#009A44"}'
curl -X POST $A/teams/$HOME/players -H 'content-type: application/json' -d '[{"name":"…","jersey_number":9}]'
curl -X POST $A/matches -H 'content-type: application/json' -d '{"home_team_id":"…","away_team_id":"…"}'
curl -X POST $A/matches/$M/videos -F camera_type=tactical -F video=@match.mp4
# fixed camera? give >=4 pixel<->pitch point pairs (metres, origin at a corner):
curl -X PUT  $A/videos/$V/calibration -H 'content-type: application/json' \
     -d '{"points":[{"px":40,"py":60,"x":0,"y":0}, … ]}'
curl -X POST $A/matches/$M/analyze -H 'content-type: application/json' \
     -d '{"periods":[{"start_s":0,"end_s":2700,"home_attacks":"right"},{"start_s":3300,"end_s":6000,"home_attacks":"left"}]}'
# progress: ws://localhost:4000/ws/jobs/$JOB   (stage_started | progress | stage_completed | needs_review | completed | failed)
curl $A/matches/$M/summary ; curl $A/matches/$M/players ; curl $A/matches/$M/events?type=shot,goal
```

## API

| Area | Endpoints |
|---|---|
| Auth | `POST /api/auth/register`, `POST /api/auth/login` (JWT; enforced when `AUTH_REQUIRED=true`) |
| Library | `POST/GET /api/teams`, `PATCH /api/teams/:id`, `GET/POST /api/teams/:id/players` (bulk squad upload) |
| Matches | `POST/GET /api/matches`, `GET/PATCH /api/matches/:id`, `POST /api/matches/:id/videos`, `PUT /api/videos/:id/calibration` |
| Jobs | `POST /api/matches/:id/analyze`, `GET /api/jobs/:id`, `DELETE /api/jobs/:id` (cancel), `WS /ws/jobs/:id` |
| Results | `GET /api/matches/:id/summary`, `/players`, `/players/:playerKey`, `/teams/:home\|away/tactics`, `/events` (filters: type, team, status, from_ts, to_ts, track_id, player, limit, offset). Alias: `GET /api/players/:playerId/matches/:matchId` |
| Review | `GET /api/jobs/:id/identity/pending`, `POST /api/jobs/:jobId/tracks/:trackId/identity`, `PATCH /api/events/:id` |
| Recompute (no video reprocessing) | `POST /api/jobs/:id/recompute {from_stage, skip_ai}`, `POST /api/jobs/:id/swap-teams` |

Recompute stages: `direction, identity, physical, events, statistics, ai`. User identity decisions survive a recompute
from `identity` or later. `from_stage: "events"` regenerates events, discarding event reviews. If a recompute fails, the
job stays `completed` with `summary.recompute_failed` set — results remain visible and are flagged as possibly mixed.
`swap-teams` (use it when the kit-cluster → home/away guess was reversed) discards identity decisions and re-runs from `direction`.

## Pipeline

`detect_track → ball → teams → calibration → pose (optional) → direction → identity → physical → events → statistics → ai`

| Stage | What it does | Needs |
|---|---|---|
| detect_track | YOLO26 detection, BoT-SORT with camera-motion compensation, camera-cut handling, per-track kit colour / jersey OCR votes | weights |
| ball | outlier rejection, gap interpolation (flagged `interpolated`), nearest-player / possession candidate (in player-heights, no calibration needed) | — |
| teams | kit-colour clustering into two teams; outliers left unaffiliated; mapped to home/away via team `kit_color` (else flagged for review) | — |
| calibration | pixel → pitch metres: manual points (fixed camera) or a pitch-landmark keypoint model (per frame) | points or model |
| direction | attack direction per period (user-supplied, else heuristic); goalkeepers/referees inferred if the detector lacks those classes | calibration |
| identity | jersey votes ↔ roster; low confidence or conflicting matches go to `needs_confirmation` | OCR, roster |
| physical | smoothed speed, distance, high-speed running, sprints, accelerations, heatmap, thirds, path, each with coverage | calibration |
| events | pass (completed / intercepted), interception, tackle, recovery, carry, dribble, pressure, duel, shot + geometric xG, save, block, goal (always `needs_review`), clearance, key pass, assist | ball |
| statistics | per-player and per-team metrics, `rating_v1`, PAC/SHO/PAS/DRI/DEF/PHY (`attr_v1`), PPDA, estimated formation | — |
| ai | Gemini team and player reports over numbered facts, verified before storing | `GEMINI_API_KEY` |

Ratings and attributes are transparent formulas over measured values, gated by minimum evidence (e.g. PAC needs ≥ 10 observed
minutes, SHO needs ≥ 2 shots). They are not calibrated against expert ratings.

## Configuration

See `.env.example`. Any threshold in `worker/config.py` can be overridden with `THRESHOLDS_JSON`, e.g. `{"sprint_kmh": 24, "control_dist": 0.8}`.

## Tests

```bash
cd worker && pip install -r requirements-dev.txt && python -m pytest tests -q     # 29 tests
cd api    && npm test                                                              # 11 tests; needs Postgres + Redis
```

The worker's integration test loads a **synthetic** match (`worker/tests/synth.py`, a test fixture with known ground truth)
into Postgres and runs every analytical stage through the real orchestrator. It verifies the analytics — distances against
ground truth, events, identity, statistics, recompute — not detector accuracy on real footage.

## What you must supply / known limits

* **Football-trained detector weights** (classes such as `player, goalkeeper, referee, ball`; set `DETECTOR_WEIGHTS`, and
  `DETECTOR_CLASS_MAP` if the class names aren't recognisable). Generic COCO weights detect people but barely detect the ball
  and have no goalkeeper/referee classes; events will be sparse.
* **Pitch calibration for panning/broadcast footage** needs a pitch-landmark keypoint model that emits the 29 landmarks in
  `worker/pipeline/calibration.py::default_layout` (or supply your own layout with `PITCH_LAYOUT_PATH`). Fixed cameras can use
  manual points. With neither, all metric-space statistics are `not_available`.
* **Jersey OCR** (`JERSEY_OCR=easyocr`, install `easyocr`) is optional. Without it, players stay "Unidentified" until confirmed by a user.
* Single-camera limits: ball height is unknown, so "shot on target" means saved or scored; aerial duels and corners are not detected.
  Goals are never auto-confirmed. Replays are not detected (camera cuts are).
* Halves: pass `periods[]` with `home_attacks` for multi-period videos, otherwise direction is assumed constant.
* Not tested against real football footage, real football-trained weights, jersey OCR, the pose model, a keypoint calibration model,
  live Gemini calls, or the Docker builds. Detection and tracking were run against a real YOLO26 model on a generic test clip only.
* Checkpointing is per stage, not per video chunk: a crash during `detect_track` restarts that stage.
