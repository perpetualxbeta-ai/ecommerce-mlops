from datetime import datetime

from src.data_simulator import MERCHANT_CATEGORIES, TransactionSimulator

REQUIRED = {"transaction_id", "user_id", "transaction_amount", "merchant_category", "timestamp", "is_fraud"}


def test_schema_and_types():
    sim = TransactionSimulator(n_users=50, seed=1)
    for t in sim.generate_batch(500):
        d = t.to_dict()
        assert REQUIRED <= d.keys()
        assert d["transaction_amount"] > 0
        assert d["merchant_category"] in MERCHANT_CATEGORIES
        assert d["is_fraud"] in (0, 1)
        datetime.fromisoformat(d["timestamp"])
        assert (d["fraud_type"] is None) == (d["is_fraud"] == 0)


def test_deterministic_with_seed():
    a = [t.to_dict() for t in TransactionSimulator(n_users=20, seed=7).generate_batch(100,
         start=datetime(2026, 1, 1).astimezone())]
    b = [t.to_dict() for t in TransactionSimulator(n_users=20, seed=7).generate_batch(100,
         start=datetime(2026, 1, 1).astimezone())]
    assert a == b


def test_fraud_rate_and_signal():
    sim = TransactionSimulator(n_users=500, fraud_rate=0.02, seed=3)
    txns = sim.generate_batch(20_000)
    fraud = [t for t in txns if t.is_fraud]
    legit = [t for t in txns if not t.is_fraud]
    rate = len(fraud) / len(txns)
    assert 0.015 < rate < 0.025, rate
    # Fraud is shifted toward high-risk categories, so a model has something to learn
    risky = {"electronics", "gift_cards", "jewelry", "digital_goods"}
    assert sum(t.merchant_category in risky for t in fraud) / len(fraud) > \
           sum(t.merchant_category in risky for t in legit) / len(legit)


def test_batch_sorted_by_time():
    ts = [t.timestamp for t in TransactionSimulator(seed=5).generate_batch(300)]
    assert ts == sorted(ts)
