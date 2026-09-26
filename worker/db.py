"""Small psycopg3 helpers: dataframes in, COPY bulk inserts, bulk updates."""
from __future__ import annotations
import json
import uuid
import datetime as _dt
import numpy as np
import pandas as pd
import psycopg
from psycopg.rows import tuple_row


def connect(url: str, autocommit: bool = False) -> psycopg.Connection:
    return psycopg.connect(url, autocommit=autocommit, row_factory=tuple_row)


def fetch_df(conn, sql: str, params=None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    return df


def fetch_one(conn, sql: str, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d.name for d in cur.description], row))


def execute(conn, sql: str, params=None) -> None:
    with conn.cursor() as cur:
        cur.execute(sql, params or ())


def _clean(v):
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (dict, list)):
        return to_json(v)
    return v


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (uuid.UUID, _dt.date, _dt.datetime)):
        return str(o)
    raise TypeError(type(o))


def _sanitize(o):
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.generic):
        return o.item()
    return o


def to_json(obj) -> str:
    """JSON-safe dump: NaN/inf become null, numpy scalars become python scalars."""
    return json.dumps(_sanitize(obj), default=_json_default)


def copy_rows(conn, table: str, cols: list[str], rows) -> int:
    n = 0
    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} ({','.join(cols)}) FROM STDIN") as cp:
            for r in rows:
                cp.write_row([_clean(v) for v in r])
                n += 1
    return n


def bulk_update(conn, table: str, key: str, df: pd.DataFrame, cols: list[str], types: dict[str, str]) -> None:
    """UPDATE table SET cols = df values WHERE table.key = df.key, via a temp table + COPY."""
    if df.empty:
        return
    tmp = f"tmp_{table}_upd"
    coldefs = ", ".join([f"{key} {types.get(key, 'BIGINT')}"] + [f"{c} {types[c]}" for c in cols])
    with conn.cursor() as cur:
        cur.execute(f"CREATE TEMP TABLE {tmp} ({coldefs}) ON COMMIT DROP")
    copy_rows(conn, tmp, [key] + cols, df[[key] + cols].itertuples(index=False, name=None))
    sets = ", ".join(f"{c}=v.{c}" for c in cols)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {table} t SET {sets} FROM {tmp} v WHERE t.{key}=v.{key}")
