"""Producer/consumer tests.

Kafka is replaced with in-memory fakes. The Postgres tests use a real database
(the `postgres` service in CI, or `docker compose up postgres` locally) and are
skipped if none is reachable.
"""

import json
from collections import namedtuple

import psycopg2
import pytest

from src import config
from src.data_simulator import TransactionSimulator
from src.streaming import consumer as consumer_mod
from src.streaming import producer as producer_mod

TP = namedtuple("TP", "topic partition")
Record = namedtuple("Record", "partition offset value")


class FakeProducer:
    def __init__(self):
        self.sent = []
        self.flushed = self.closed = False

    def send(self, topic, key, value):
        # mimic the real serializers so we know messages are JSON-serialisable
        self.sent.append((topic, key.encode(), json.dumps(value).encode()))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


class FakeConsumer:
    """Replays the given batches from poll(), then returns nothing."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.commits = 0
        self.seeks = []
        self.closed = False

    def poll(self, timeout_ms, max_records):
        return self.batches.pop(0) if self.batches else {}

    def commit(self):
        self.commits += 1

    def seek(self, tp, offset):
        self.seeks.append((tp, offset))

    def close(self):
        self.closed = True


def test_producer_sends_keyed_json():
    fake = FakeProducer()
    n = producer_mod.run(fake, TransactionSimulator(n_users=10, seed=1), "transactions", rate=0, max_messages=25)
    assert n == 25 and len(fake.sent) == 25
    assert fake.flushed and fake.closed
    topic, key, value = fake.sent[0]
    event = json.loads(value)
    assert topic == "transactions"
    assert key.decode() == event["user_id"]


def test_to_row_rejects_malformed():
    assert consumer_mod.to_row({"user_id": "u1"}) is None
    good = TransactionSimulator(seed=1).next_transaction().to_dict()
    row = consumer_mod.to_row(good, partition=0, offset=5)
    assert row[0] == good["transaction_id"] and row[-2:] == (0, 5)


# ---- Postgres-backed tests -------------------------------------------------

@pytest.fixture
def pg_conn():
    try:
        conn = psycopg2.connect(config.postgres_dsn(), connect_timeout=3)
    except psycopg2.OperationalError:
        pytest.skip("Postgres not reachable")
    consumer_mod.ensure_schema(conn)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE transactions")
    conn.commit()
    yield conn
    if not conn.closed:
        conn.close()


def _count(dsn=None):
    with psycopg2.connect(dsn or config.postgres_dsn()) as c, c.cursor() as cur:
        cur.execute("SELECT COUNT(*), SUM(is_fraud) FROM transactions")
        return cur.fetchone()


def _batch(txns, tp=TP("transactions", 0), start_offset=0):
    return {tp: [Record(tp.partition, start_offset + i, json.dumps(t.to_dict()).encode())
                 for i, t in enumerate(txns)]}


def test_consumer_writes_to_postgres(pg_conn):
    sim = TransactionSimulator(n_users=50, seed=2)
    txns = [sim.next_transaction() for _ in range(300)]
    batches = [_batch(txns[:150]), _batch(txns[150:], start_offset=150)]
    fake = FakeConsumer(batches)

    inserted = consumer_mod.run(fake, pg_conn, max_polls=3)

    assert inserted == 300
    assert fake.commits == 2 and fake.closed
    count, frauds = _count()
    assert count == 300
    assert frauds == sum(t.is_fraud for t in txns)


def test_consumer_is_idempotent_and_skips_bad_messages(pg_conn):
    sim = TransactionSimulator(n_users=10, seed=4)
    txns = [sim.next_transaction() for _ in range(20)]
    tp = TP("transactions", 0)
    first = _batch(txns)
    replay = _batch(txns)  # same messages redelivered after a "crash"
    replay[tp].append(Record(0, 99, b"not json"))
    replay[tp].append(Record(0, 100, json.dumps({"user_id": "missing_fields"}).encode()))

    inserted = consumer_mod.run(FakeConsumer([first, replay]), pg_conn, max_polls=2)

    assert inserted == 20
    assert _count()[0] == 20


def test_consumer_rewinds_and_does_not_commit_on_db_error(pg_conn):
    tp = TP("transactions", 0)
    sim = TransactionSimulator(n_users=5, seed=9)
    bad = sim.next_transaction().to_dict()
    bad["transaction_amount"] = 1e15  # overflows NUMERIC(12, 2) -> Postgres rejects the batch
    batch = {tp: [Record(0, 40, json.dumps(sim.next_transaction().to_dict()).encode()),
                  Record(0, 41, json.dumps(bad).encode())]}
    fake = FakeConsumer([batch])

    inserted = consumer_mod.run(fake, pg_conn, max_polls=1)

    assert inserted == 0
    assert fake.commits == 0                 # offsets NOT committed
    assert fake.seeks == [(tp, 40)]          # rewound to the start of the failed batch
    assert _count()[0] == 0                  # whole batch rolled back, no partial write
