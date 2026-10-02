"""Stream synthetic transactions into Kafka in real time.

Messages are JSON, keyed by user_id so all of a user's transactions land on the
same partition (keeps per-user ordering — needed later for velocity features).

Usage:
    python -m src.streaming.producer --rate 5            # 5 txns/sec, forever
    python -m src.streaming.producer --rate 50 --max 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time

from kafka import KafkaProducer
from kafka.errors import KafkaError  # base class; NoBrokersAvailable was removed in kafka-python 3.x

from src import config
from src.data_simulator import TransactionSimulator

log = logging.getLogger("producer")


def build_producer(bootstrap: str, retries: int = 10) -> KafkaProducer:
    """Connect to Kafka, retrying while the broker is still starting up."""
    for attempt in range(1, retries + 1):
        try:
            return KafkaProducer(
                bootstrap_servers=bootstrap,
                key_serializer=lambda k: k.encode("utf-8"),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                linger_ms=20,
                retries=5,
                bootstrap_timeout_ms=5000,         # fail fast so the retry loop can report progress
            )
        except KafkaError:
            log.warning("Kafka not ready at %s (attempt %d/%d), retrying in 3s", bootstrap, attempt, retries)
            time.sleep(3)
    raise SystemExit(f"Could not connect to Kafka at {bootstrap}")


def run(producer: KafkaProducer, sim: TransactionSimulator, topic: str,
        rate: float, max_messages: int | None = None) -> int:
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    interval = 1.0 / rate if rate > 0 else 0
    sent = frauds = 0
    try:
        while not stop and (max_messages is None or sent < max_messages):
            txn = sim.next_transaction()
            producer.send(topic, key=txn.user_id, value=txn.to_dict())
            sent += 1
            frauds += txn.is_fraud
            if sent % 100 == 0:
                log.info("sent=%d fraud=%d (%.1f%%)", sent, frauds, 100 * frauds / sent)
            if interval:
                time.sleep(interval)
    finally:
        producer.flush()
        producer.close()
    log.info("Done. sent=%d fraud=%d", sent, frauds)
    return sent


def main() -> None:
    p = argparse.ArgumentParser(description="Kafka producer for synthetic transactions")
    p.add_argument("--bootstrap", default=config.KAFKA_BOOTSTRAP_SERVERS)
    p.add_argument("--topic", default=config.KAFKA_TRANSACTIONS_TOPIC)
    p.add_argument("--rate", type=float, default=5.0, help="transactions per second (0 = as fast as possible)")
    p.add_argument("--max", type=int, default=None, help="stop after N messages")
    p.add_argument("--users", type=int, default=1000)
    p.add_argument("--fraud-rate", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    sim = TransactionSimulator(n_users=args.users, fraud_rate=args.fraud_rate, seed=args.seed)
    log.info("Producing to %s / topic=%s at %.1f msg/s", args.bootstrap, args.topic, args.rate)
    run(build_producer(args.bootstrap), sim, args.topic, args.rate, args.max)


if __name__ == "__main__":
    main()
