-- InsightPlay schema v1
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'analyst',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE teams (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id UUID REFERENCES users(id) ON DELETE SET NULL,
  name TEXT NOT NULL,
  kit_color TEXT,                     -- hex, e.g. #009A44 (used to map kit clusters to teams)
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE players (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  team_id UUID NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  jersey_number INT,
  position TEXT,
  reference_embedding JSONB,          -- appearance embedding learned from confirmed tracks
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX players_team_idx ON players(team_id);

CREATE TABLE matches (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id UUID REFERENCES users(id) ON DELETE SET NULL,
  home_team_id UUID REFERENCES teams(id),
  away_team_id UUID REFERENCES teams(id),
  match_date DATE,
  competition TEXT,
  pitch_length REAL NOT NULL DEFAULT 105,
  pitch_width REAL NOT NULL DEFAULT 68,
  reported_home_score INT,            -- optional user-entered result (kept separate from detected goals)
  reported_away_score INT,
  status TEXT NOT NULL DEFAULT 'created',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE video_assets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  match_id UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  original_name TEXT,
  camera_type TEXT NOT NULL DEFAULT 'broadcast' CHECK (camera_type IN ('tactical','broadcast','training')),
  static_camera BOOLEAN NOT NULL DEFAULT FALSE,
  fps REAL, duration_s REAL, width INT, height INT, frame_count BIGINT, codec TEXT,
  size_bytes BIGINT,
  calibration_points JSONB,           -- [{px,py,x,y}] pixel -> pitch metre correspondences (static camera)
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX video_assets_match_idx ON video_assets(match_id);

CREATE TABLE analysis_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  match_id UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
  video_id UUID REFERENCES video_assets(id) ON DELETE SET NULL,
  type TEXT NOT NULL DEFAULT 'analyze' CHECK (type IN ('analyze','recompute')),
  status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','completed','failed','cancelled','recomputing')),
  stage TEXT,
  progress REAL NOT NULL DEFAULT 0,
  stages_done JSONB NOT NULL DEFAULT '[]',
  config JSONB NOT NULL DEFAULT '{}',
  summary JSONB NOT NULL DEFAULT '{}',   -- coverage, warnings, direction, calibration info, review counters
  model_versions JSONB NOT NULL DEFAULT '{}',
  error TEXT,
  attempts INT NOT NULL DEFAULT 0,
  cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
  heartbeat_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
);
CREATE INDEX analysis_jobs_match_idx ON analysis_jobs(match_id, created_at DESC);
CREATE INDEX analysis_jobs_status_idx ON analysis_jobs(status);

CREATE TABLE player_tracks (
  id BIGSERIAL PRIMARY KEY,
  job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
  track_id INT NOT NULL,
  frame INT NOT NULL,
  ts REAL NOT NULL,
  role TEXT NOT NULL,                 -- player | goalkeeper | referee
  px REAL NOT NULL, py REAL NOT NULL, -- foot point, pixels
  x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
  conf REAL NOT NULL,
  team TEXT,                          -- home | away | NULL
  player_id UUID REFERENCES players(id) ON DELETE SET NULL,
  pitch_x REAL, pitch_y REAL,         -- metres, NULL when the frame is not calibrated
  calib_conf REAL,
  vx REAL, vy REAL, speed REAL, accel REAL   -- m/s, m/s^2 (filled by the physical stage)
);
CREATE INDEX player_tracks_job_track_idx ON player_tracks(job_id, track_id, frame);
CREATE INDEX player_tracks_job_frame_idx ON player_tracks(job_id, frame);

CREATE TABLE ball_tracks (
  id BIGSERIAL PRIMARY KEY,
  job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
  frame INT NOT NULL,
  ts REAL NOT NULL,
  px REAL NOT NULL, py REAL NOT NULL,
  conf REAL,
  interpolated BOOLEAN NOT NULL DEFAULT FALSE,
  pitch_x REAL, pitch_y REAL, calib_conf REAL,
  vx REAL, vy REAL, speed REAL,
  nearest_track INT, nearest_dist REAL,   -- distance to nearest player, in player-heights
  possession_track INT                     -- track judged to be controlling the ball (candidate)
);
CREATE INDEX ball_tracks_job_frame_idx ON ball_tracks(job_id, frame);

CREATE TABLE track_features (
  job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
  track_id INT NOT NULL,
  role TEXT NOT NULL,
  first_frame INT, last_frame INT, n_obs INT,
  colour JSONB,                       -- median Lab kit colour
  appearance JSONB,                   -- appearance embedding
  jersey_votes JSONB,                 -- {"22": {"w": 2.7, "n": 3}, ...}
  cluster INT,                        -- kit cluster (0/1) or -1 for outliers
  cluster_dist REAL,
  PRIMARY KEY (job_id, track_id)
);

CREATE TABLE calibration_segments (
  id BIGSERIAL PRIMARY KEY,
  job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
  start_frame INT NOT NULL, end_frame INT NOT NULL,
  method TEXT NOT NULL,               -- manual | keypoint_model
  matrix JSONB NOT NULL,              -- 3x3 pixel -> pitch homography
  confidence REAL, error_m REAL
);
CREATE INDEX calibration_segments_job_idx ON calibration_segments(job_id, start_frame);

CREATE TABLE pose_features (
  job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
  track_id INT NOT NULL,
  frame INT NOT NULL,
  features JSONB NOT NULL,
  PRIMARY KEY (job_id, track_id, frame)
);

CREATE TABLE identity_candidates (
  id BIGSERIAL PRIMARY KEY,
  job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
  track_id INT NOT NULL,
  team TEXT,
  player_id UUID REFERENCES players(id) ON DELETE SET NULL,
  jersey_number INT,
  ocr_conf REAL, appearance_sim REAL,
  confidence REAL NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'needs_confirmation' CHECK (status IN ('auto','needs_confirmation','confirmed','rejected','unmatched')),
  evidence JSONB NOT NULL DEFAULT '{}',
  UNIQUE (job_id, track_id)
);

CREATE TABLE events (
  id BIGSERIAL PRIMARY KEY,
  job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
  type TEXT NOT NULL,                 -- pass, shot, goal, carry, dribble, tackle, interception, clearance, block, recovery, pressure, duel, key_pass, assist
  subtype TEXT,
  ts REAL NOT NULL, frame INT NOT NULL,
  team TEXT,
  track_id INT, target_track_id INT,
  sx REAL, sy REAL, ex REAL, ey REAL,  -- pitch metres (NULL when uncalibrated)
  distance_m REAL,
  confidence REAL NOT NULL DEFAULT 0.5,
  status TEXT NOT NULL DEFAULT 'auto' CHECK (status IN ('auto','needs_review','confirmed','rejected')),
  attributes JSONB NOT NULL DEFAULT '{}',
  evidence JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX events_job_type_idx ON events(job_id, type, ts);

CREATE TABLE entity_physical (
  job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
  entity_key TEXT NOT NULL,           -- player:<uuid> | track:<n>
  metrics JSONB NOT NULL,
  heatmap JSONB, avg_position JSONB, zone_time JSONB, movement_path JSONB,
  PRIMARY KEY (job_id, entity_key)
);

CREATE TABLE player_match_stats (
  job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
  entity_key TEXT NOT NULL,
  player_id UUID REFERENCES players(id) ON DELETE SET NULL,
  team TEXT,
  role TEXT,
  track_ids INT[] NOT NULL DEFAULT '{}',
  identity_confidence REAL,
  identity_status TEXT,
  metrics JSONB NOT NULL,
  attributes JSONB NOT NULL DEFAULT '{}',   -- PAC/SHO/PAS/DRI/DEF/PHY
  heatmap JSONB, avg_position JSONB, zone_time JSONB, movement_path JSONB,
  PRIMARY KEY (job_id, entity_key)
);

CREATE TABLE team_match_stats (
  job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
  team TEXT NOT NULL,                 -- home | away
  metrics JSONB NOT NULL,
  tactics JSONB NOT NULL DEFAULT '{}',
  PRIMARY KEY (job_id, team)
);

CREATE TABLE ai_analyses (
  id BIGSERIAL PRIMARY KEY,
  job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
  scope TEXT NOT NULL,                -- team | player
  subject_key TEXT NOT NULL,          -- home | away | player:<uuid> | track:<n>
  content JSONB NOT NULL,
  dropped JSONB NOT NULL DEFAULT '[]', -- statements discarded because their numbers did not match the input data
  model TEXT NOT NULL,
  input_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (job_id, scope, subject_key)
);
