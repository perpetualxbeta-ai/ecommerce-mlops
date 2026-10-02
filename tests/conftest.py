"""Shared test setup.

Tests that touch Postgres TRUNCATE tables, so they must never run against your
working database. Every test session is redirected to `<POSTGRES_DB>_test`,
which is created on the fly if Postgres is reachable.
"""

import psycopg2
from psycopg2 import sql

from src import config

_REAL_DB = config.POSTGRES_DB
TEST_DB = _REAL_DB if _REAL_DB.endswith("_test") else f"{_REAL_DB}_test"
config.POSTGRES_DB = TEST_DB  # config.postgres_dsn() reads this at call time


def _ensure_test_db() -> None:
    admin_dsn = config.postgres_dsn().replace(f"dbname={TEST_DB}", f"dbname={_REAL_DB}")
    try:
        conn = psycopg2.connect(admin_dsn, connect_timeout=3)
    except psycopg2.OperationalError:
        return  # no Postgres: DB-backed tests skip themselves
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,))
        if not cur.fetchone():
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TEST_DB)))
    conn.close()


_ensure_test_db()
