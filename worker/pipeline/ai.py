"""Stage — Gemini interpretation of already-computed statistics.

The model never sees video or raw tracking and never computes anything. It receives a list of numbered facts and must cite
fact ids for every statement. Each statement is then verified: cited ids must exist and every number in the text must match a
cited value. Statements that fail are dropped (and recorded), so an invented figure never reaches the product."""
from __future__ import annotations
import hashlib
import json
import logging
import re

from db import execute, fetch_df, to_json

log = logging.getLogger("ai")
NAME = "ai"

SYSTEM = """You are a football analyst writing for a coach. You are given a numbered list of FACTS computed from match video by a tracking system.
Rules:
- Use ONLY the facts provided. Never invent, estimate or recall statistics, players, scores or events.
- Every statement must list the fact ids it relies on in "facts_used". Every number you write must equal (or round to) a cited fact value.
- If the facts are insufficient for a point, omit it. If a limitation listed under "limitations" affects a conclusion, say so plainly.
- Be specific, practical and concise. No praise padding. Do not mention fact ids in the text.
Return JSON only, matching the schema you are given."""

TEAM_SCHEMA = {"headline": "STATEMENT", "attacking": ["STATEMENT"], "defending": ["STATEMENT"], "transitions": ["STATEMENT"], "weaknesses": ["STATEMENT"], "coach_notes": ["STATEMENT"]}
PLAYER_SCHEMA = {"headline": "STATEMENT", "strengths": ["STATEMENT"], "areas_to_improve": ["STATEMENT"], "coach_notes": ["STATEMENT"]}
STATEMENT = {"text": "string", "facts_used": ["fact id", "..."]}

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def enabled(ctx):
    if ctx.job_config.get("skip_ai") or ((ctx.job_config.get("recompute") or {}).get("skip_ai")):
        return False, "skipped by request"
    if not ctx.cfg.gemini_api_key:
        return False, "GEMINI_API_KEY not set"
    return True, ""


def flatten_facts(obj, prefix: str, out: dict) -> dict:
    """{'possession_pct': {'value': 57.1, ...}} -> {'home.possession_pct': 57.1}. Only measured values become facts."""
    if isinstance(obj, dict):
        if "value" in obj and not isinstance(obj["value"], (dict, list)):
            if obj["value"] is not None and not isinstance(obj["value"], bool):
                out[prefix] = obj["value"]
            return out
        for k, v in obj.items():
            if k in ("attack_normalised", "basis", "contributions", "note", "model", "method", "source", "confidence", "unit", "coverage_pct"):
                continue
            flatten_facts(v, f"{prefix}.{k}" if prefix else k, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = obj
    return out


def verify_statement(st, facts: dict) -> tuple[bool, str]:
    """Pure function, unit-tested."""
    if not isinstance(st, dict) or not isinstance(st.get("text"), str) or not st["text"].strip():
        return False, "malformed"
    ids = st.get("facts_used")
    if not isinstance(ids, list) or not ids:
        return False, "no_facts_cited"
    if any(i not in facts for i in ids):
        return False, "unknown_fact_id"
    vals = [float(facts[i]) for i in ids if isinstance(facts[i], (int, float)) and not isinstance(facts[i], bool)]
    for tok in _NUM.findall(st["text"]):
        x = float(tok)
        if not any(abs(x - v) <= max(0.06, 0.01 * abs(v)) or abs(x - round(v)) < 1e-9 or abs(x - round(v, 1)) < 1e-9 for v in vals):
            return False, f"number_{tok}_not_in_cited_facts"
    return True, ""


def verify_content(content: dict, facts: dict) -> tuple[dict, list]:
    kept, dropped = {}, []
    for key, val in content.items():
        if isinstance(val, list):
            good = []
            for st in val:
                ok, why = verify_statement(st, facts)
                (good.append(st) if ok else dropped.append({"section": key, "statement": st, "reason": why}))
            kept[key] = good
        elif isinstance(val, dict):
            ok, why = verify_statement(val, facts)
            if ok:
                kept[key] = val
            else:
                dropped.append({"section": key, "statement": val, "reason": why})
    return kept, dropped


def analyse(payload: dict, schema: dict, generate) -> tuple[dict, list]:
    """generate(system, user) -> str. Retries once on malformed JSON."""
    facts = payload["facts_map"]
    user = json.dumps({"context": payload["context"], "facts": payload["facts"], "not_available": payload["not_available"],
                       "limitations": payload["limitations"], "output_schema": schema, "statement": STATEMENT}, ensure_ascii=False)
    last = None
    for _ in range(2):
        raw = generate(SYSTEM, user)
        try:
            txt = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
            content = json.loads(txt)
            if isinstance(content, dict):
                return verify_content(content, facts)
        except Exception as e:  # malformed JSON
            last = e
    raise RuntimeError(f"model returned unusable JSON: {last}")


def build_payload(context: dict, metrics: dict, extra: dict | None, limitations: list[str]) -> dict:
    facts_map = flatten_facts(metrics, "", {})
    if extra:
        flatten_facts(extra, "", facts_map)
    na = [{"id": k, "reason": v.get("reason")} for k, v in (metrics or {}).items() if isinstance(v, dict) and v.get("status") == "not_available"]
    return {"context": context, "facts": [{"id": k, "value": v} for k, v in facts_map.items()], "facts_map": facts_map, "not_available": na, "limitations": limitations}


def make_generate(cfg):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=cfg.gemini_api_key)

    def generate(system: str, user: str) -> str:
        r = client.models.generate_content(model=cfg.gemini_model, contents=user,
                                           config=types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json", temperature=0.2))
        return r.text
    return generate


def run(ctx) -> None:
    cfg, conn, job_id = ctx.cfg, ctx.conn, ctx.job_id
    generate = make_generate(cfg)
    ctx.set_model_versions({"interpretation": cfg.gemini_model})
    limits = list(ctx.summary.get("warnings") or [])
    names = {"home": (ctx.home or {}).get("name", "Home"), "away": (ctx.away or {}).get("name", "Away")}
    n_done = n_dropped = 0

    def store(scope, key, payload, schema):
        nonlocal n_done, n_dropped
        h = hashlib.sha256(json.dumps(payload["facts"], sort_keys=True).encode()).hexdigest()
        prev = fetch_df(conn, "SELECT input_hash FROM ai_analyses WHERE job_id=%s AND scope=%s AND subject_key=%s", (job_id, scope, key))
        if len(prev) and prev["input_hash"].iloc[0] == h:
            return
        if len(payload["facts"]) < 3:
            return
        content, dropped = analyse(payload, schema, generate)
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO ai_analyses(job_id,scope,subject_key,content,dropped,model,input_hash) VALUES (%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s)
                           ON CONFLICT (job_id,scope,subject_key) DO UPDATE SET content=EXCLUDED.content, dropped=EXCLUDED.dropped, model=EXCLUDED.model,
                           input_hash=EXCLUDED.input_hash, created_at=now()""", (job_id, scope, key, to_json(content), to_json(dropped), cfg.gemini_model, h))
        n_done += 1
        n_dropped += len(dropped)
        ctx.rep.progress(NAME, min(0.95, n_done / 20.0))

    teams = {r.team: r for r in fetch_df(conn, "SELECT team, metrics, tactics FROM team_match_stats WHERE job_id=%s", (job_id,)).itertuples()}
    for side in ("home", "away"):
        r = teams.get(side)
        if r is None:
            continue
        opp = teams.get("away" if side == "home" else "home")
        extra = {"opponent": opp.metrics} if opp is not None else None
        tactics = {k: v for k, v in (r.tactics or {}).items() if k in ("formation", "shape", "possession_by_third_s")}
        payload = build_payload({"team": names[side], "side": side, "match_analysed_by": "computer vision tracking"}, {**r.metrics, **{f"tactics_{k}": v for k, v in tactics.items()}}, extra, limits)
        store("team", side, payload, TEAM_SCHEMA)

    players = fetch_df(conn, "SELECT s.entity_key, s.team, s.metrics, s.attributes, p.name, p.jersey_number FROM player_match_stats s LEFT JOIN players p ON p.id=s.player_id WHERE s.job_id=%s AND s.role<>'referee'", (job_id,))
    players["minutes"] = players["metrics"].map(lambda m: ((m.get("minutes") or {}).get("value")) or 0)
    for r in players[players["minutes"] >= cfg.ai_min_minutes].sort_values("minutes", ascending=False).head(cfg.ai_max_players).itertuples():
        label = r.name or f"unidentified {r.team} player"
        payload = build_payload({"player": label, "team": names.get(r.team, ""), "jersey_number": None if r.jersey_number != r.jersey_number else r.jersey_number},
                                {**r.metrics, **{f"attr_{k}": v for k, v in (r.attributes or {}).items()}}, None, limits)
        store("player", r.entity_key, payload, PLAYER_SCHEMA)
    ctx.summary["ai"] = {"analyses": n_done, "statements_dropped": n_dropped, "model": cfg.gemini_model}
    if n_dropped:
        ctx.warn(f"{n_dropped} AI statement(s) were dropped because their numbers did not match the computed facts")
