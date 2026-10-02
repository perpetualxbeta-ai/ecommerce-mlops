"""Postgres access for the API: user history, category averages, prediction log."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager

import pandas as pd
from psycopg2.pool import ThreadedConnectionPool

log = logging.getLogger("api.store")

HISTORY_SQL = """
    SELECT transaction_id, user_id, transaction_amount::float AS transaction_amount,
           merchant_category, timestamp, payment_method, country, device_type
    FROM transactions
    WHERE user_id = %s AND timestamp < %s AND transaction_id <> %s
    ORDER BY timestamp DESC
    LIMIT %s
"""
CATEGORY_MEANS_SQL = """
    SELECT merchant_category, AVG(transaction_amount)::float
    FROM transactions GROUP BY merchant_category
"""
PREDICTIONS_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id               SERIAL PRIMARY KEY,
    transaction_id   VARCHAR(64),
    user_id          VARCHAR(64),
    model_name       VARCHAR(64) NOT NULL,
    model_version    VARCHAR(32),
    score            DOUBLE PRECISION NOT NULL,
    label            SMALLINT,          -- ground truth, back-filled later for monitoring
    predicted_at     TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS decision VARCHAR(16);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS latency_ms DOUBLE PRECISION;
CREATE INDEX IF NOT EXISTS idx_predictions_txn ON predictions (transaction_id);
"""
INSERT_PREDICTION_SQL = """
    INSERT INTO predictions (transaction_id, user_id, model_name, model_version, score,
                             decision, latency_ms)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


class FeatureStore:
    def __init__(self, dsn: str, history_limit: int = 500, category_ttl_s: int = 300,
                 min_conn: int = 1, max_conn: int = 10):
        self.pool = ThreadedConnectionPool(min_conn, max_conn, dsn)
        self.history_limit = history_limit
        self.category_ttl_s = category_ttl_s
        self._category_means: dict[str, float] = {}
        self._category_loaded_at = 0.0
        self._cat_lock = threading.Lock()
        with self.conn() as c, c.cursor() as cur:
            cur.execute(PREDICTIONS_DDL)

    @contextmanager
    def conn(self):
        c = self.pool.getconn()
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            self.pool.putconn(c)

    def ping(self) -> bool:
        try:
            with self.conn() as c, c.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    def user_history(self, user_id: str, before, exclude_txn_id: str) -> pd.DataFrame:
        with self.conn() as c, c.cursor() as cur:
            cur.execute(HISTORY_SQL, (user_id, before, exclude_txn_id, self.history_limit))
            rows = cur.fetchall()
            cols = [d.name for d in cur.description]
        return pd.DataFrame(rows, columns=cols).iloc[::-1].reset_index(drop=True)

    def category_means(self) -> dict[str, float]:
        """Global average amount per category, cached for `category_ttl_s` seconds."""
        if time.monotonic() - self._category_loaded_at > self.category_ttl_s:
            with self._cat_lock:
                if time.monotonic() - self._category_loaded_at > self.category_ttl_s:
                    with self.conn() as c, c.cursor() as cur:
                        cur.execute(CATEGORY_MEANS_SQL)
                        self._category_means = dict(cur.fetchall())
                    self._category_loaded_at = time.monotonic()
        return self._category_means

    def log_prediction(self, txn_id: str, user_id: str, model_name: str, model_version: str,
                       score: float, decision: str, latency_ms: float) -> None:
        try:
            with self.conn() as c, c.cursor() as cur:
                cur.execute(INSERT_PREDICTION_SQL, (txn_id, user_id, model_name, model_version, score,
                                                    decision, latency_ms))
        except Exception:
            log.exception("Failed to log prediction for %s", txn_id)

    def close(self) -> None:
        self.pool.closeall()
