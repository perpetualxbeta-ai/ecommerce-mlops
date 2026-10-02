"""Consume the transactions topic and persist raw events into PostgreSQL.

Delivery semantics: at-least-once + idempotent writes.
  * Offsets are committed to Kafka only AFTER the batch is committed to Postgres.
  * Inserts use ON CONFLICT (transaction_id) DO NOTHING, so a replayed batch
    after a crash doesn't create duplicates.

Usage:
    python -m src.streaming.consumer
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from typing import Iterable

import psycopg2
from psycopg2.extras import execute_values
from kafka import KafkaConsumer
from kafka.errors import KafkaError  # base class; NoBrokersAvailable was removed in kafka-python 3.x

from src import config

log = logging.getLogger("consumer")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id     VARCHAR(64) PRIMARY KEY,
    user_id            VARCHAR(64)    NOT NULL,
    transaction_amount NUMERIC(12, 2) NOT NULL,
    merchant_category  VARCHAR(32)    NOT NULL,
    timestamp          TIMESTAMPTZ    NOT NULL,
    is_fraud           SMALLINT       NOT NULL,
    payment_method     VARCHAR(32),
    country            VARCHAR(8),
    device_type        VARCHAR(32),
    fraud_type         VARCHAR(32),
    kafka_partition    INT,
    kafka_offset       BIGINT,
    ingested_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_transactions_user_ts ON transactions (user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_ts ON transactions (timestamp);
"""

COLUMNS = (
    "transaction_id", "user_id", "transaction_amount", "merchant_category", "timestamp",
    "is_fraud", "payment_method", "country", "device_type", "fraud_type",
    "kafka_partition", "kafka_offset",
)
INSERT_SQL = (
    f"INSERT INTO transactions ({', '.join(COLUMNS)}) VALUES %s "
    "ON CONFLICT (transaction_id) DO NOTHING"
)
REQUIRED = ("transaction_id", "user_id", "transaction_amount", "merchant_category", "timestamp", "is_fraud")


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()


def to_row(event: dict, partition: int | None = None, offset: int | None = None) -> tuple | None:
    """Validate one event and turn it into a DB row. Returns None for malformed events."""
    if any(event.get(k) is None for k in REQUIRED):
        return None
    return (
        event["transaction_id"], event["user_id"], float(event["transaction_amount"]),
        event["merchant_category"], event["timestamp"], int(event["is_fraud"]),
        event.get("payment_method"), event.get("country"), event.get("device_type"),
        event.get("fraud_type"), partition, offset,
    )


def write_batch(conn, rows: Iterable[tuple]) -> int:
    """Insert rows in one transaction. Returns the number of NEW rows written."""
    rows = list(rows)
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, INSERT_SQL, rows, page_size=len(rows))  # one statement, so rowcount covers the whole batch
        inserted = cur.rowcount
    conn.commit()
    return inserted


def connect_postgres(dsn: str, retries: int = 10):
    for attempt in range(1, retries + 1):
        try:
            return psycopg2.connect(dsn)
        except psycopg2.OperationalError as e:
            log.warning("Postgres not ready (attempt %d/%d): %s", attempt, retries, str(e).strip())
            time.sleep(3)
    raise SystemExit("Could not connect to Postgres")


def build_consumer(bootstrap: str, topic: str, group: str, retries: int = 10) -> KafkaConsumer:
    for attempt in range(1, retries + 1):
        try:
            return KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap,
                group_id=group,
                enable_auto_commit=False,          # we commit after the DB write
                auto_offset_reset="earliest",
                value_deserializer=lambda b: b,    # decode ourselves so bad JSON can't crash the poll loop
                bootstrap_timeout_ms=5000,         # fail fast so the retry loop can report progress
            )
        except KafkaError:
            log.warning("Kafka not ready at %s (attempt %d/%d), retrying in 3s", bootstrap, attempt, retries)
            time.sleep(3)
    raise SystemExit(f"Could not connect to Kafka at {bootstrap}")


def run(consumer: KafkaConsumer, conn, batch_size: int = 200, poll_timeout_ms: int = 1000,
        max_polls: int | None = None) -> int:
    """Poll-write-commit loop. Runs until SIGINT/SIGTERM (or `max_polls`, used by tests)."""
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    total = skipped = polls = 0
    try:
        while not stop and (max_polls is None or polls < max_polls):
            polls += 1
            polled = consumer.poll(timeout_ms=poll_timeout_ms, max_records=batch_size)
            if not polled:
                continue
            rows = []
            batch_start = {tp: messages[0].offset for tp, messages in polled.items() if messages}
            for tp, messages in polled.items():
                for msg in messages:
                    try:
                        row = to_row(json.loads(msg.value), msg.partition, msg.offset)
                    except (ValueError, TypeError):
                        row = None
                    if row is None:
                        skipped += 1
                        log.warning("Skipping malformed message at %s:%d", tp, msg.offset)
                        continue
                    rows.append(row)
            try:
                inserted = write_batch(conn, rows)
            except psycopg2.Error:
                conn.rollback()
                log.exception("DB write failed; rewinding to retry the batch")
                for tp, offset in batch_start.items():
                    consumer.seek(tp, offset)
                time.sleep(2)
                continue
            consumer.commit()
            total += inserted
            log.info("batch=%d inserted=%d total=%d skipped=%d", len(rows), inserted, total, skipped)
    finally:
        consumer.close()
        conn.close()
        log.info("Stopped. total inserted=%d skipped=%d", total, skipped)
    return total


def main() -> None:
    p = argparse.ArgumentParser(description="Kafka -> Postgres sink for raw transactions")
    p.add_argument("--bootstrap", default=config.KAFKA_BOOTSTRAP_SERVERS)
    p.add_argument("--topic", default=config.KAFKA_TRANSACTIONS_TOPIC)
    p.add_argument("--group", default=config.KAFKA_CONSUMER_GROUP)
    p.add_argument("--batch-size", type=int, default=200)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    conn = connect_postgres(config.postgres_dsn())
    ensure_schema(conn)
    consumer = build_consumer(args.bootstrap, args.topic, args.group)
    log.info("Consuming %s from %s into Postgres", args.topic, args.bootstrap)
    run(consumer, conn, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
