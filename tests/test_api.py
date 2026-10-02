"""API tests: real Postgres (test DB) for history, a throwaway SQLite MLflow registry."""

import sys
import time

import mlflow
import pandas as pd
import psycopg2
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from src import config
from src.api import main
from src.api.schemas import Decision
from src.data_simulator import TransactionSimulator
from src.models import train
from src.streaming.consumer import ensure_schema, to_row, write_batch
from tests.conftest import DATA_START


def _pg():
    try:
        return psycopg2.connect(config.postgres_dsn(), connect_timeout=3)
    except psycopg2.OperationalError:
        pytest.skip("Postgres not reachable")


def _train(mp, raw, *args):
    mp.setattr(train, "load_transactions", lambda limit: raw)
    mp.setattr(sys, "argv", ["train", "--n-trials", "1", *args])
    train.main()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    conn = _pg()
    ensure_schema(conn)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE transactions")
        cur.execute("DROP TABLE IF EXISTS predictions")
    conn.commit()

    sim = TransactionSimulator(n_users=150, seed=21)
    txns = sim.generate_batch(15000, start=DATA_START, days=30)
    write_batch(conn, [to_row(t.to_dict()) for t in txns])
    raw = pd.DataFrame([t.to_dict() for t in txns])

    uri = f"sqlite:///{tmp}/mlflow.db"
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        mp.setenv("MLFLOW_TRACKING_URI", uri)
        mlflow.set_tracking_uri(uri)
        _train(mp, raw)
        mp.setattr(main, "TRACKING_URI", uri)
        mp.setattr(main, "REFRESH_SECONDS", 3600.0)  # tests trigger reloads explicitly
        with TestClient(main.app) as client:
            for _ in range(100):
                if client.get("/ready").json()["ready"]:
                    break
                time.sleep(0.2)
            # a user with plenty of normal history to compare against
            legit = raw[raw.is_fraud == 0]
            user = legit.user_id.value_counts().index[0]
            u = legit[legit.user_id == user]
            habits = dict(user_id=user, country=u.country.mode()[0], device_type=u.device_type.mode()[0],
                          payment_method=u.payment_method.mode()[0],
                          merchant_category=u.merchant_category.mode()[0],
                          avg=float(u.transaction_amount.mean()))
            yield dict(client=client, raw=raw, conn=conn, mp=mp, uri=uri, habits=habits)
    conn.close()


def _txn(h, **over):
    body = {k: h[k] for k in ("user_id", "country", "device_type", "payment_method", "merchant_category")}
    body["transaction_amount"] = round(h["avg"] * 0.5, 2)
    body["timestamp"] = "2026-10-01T06:00:00Z"  # daytime in Asia
    body.update(over)
    return body


def test_ready_and_model_info(env):
    c = env["client"]
    assert c.get("/health").json() == {"status": "ok"}
    r = c.get("/ready").json()
    assert r["ready"] and r["model_version"] == "1"
    info = c.get("/model").json()
    assert info["thresholds"]["review"] <= info["thresholds"]["block"]
    assert info["n_features"] > 20


def test_normal_transaction_allowed(env):
    r = env["client"].post("/predict", json=_txn(env["habits"]))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["decision"] == "ALLOW"
    assert 0 <= d["fraud_probability"] < 0.5
    assert d["user_history_size"] > 10 and d["model_version"] == "1"


def test_account_takeover_blocked_with_reasons(env):
    h = env["habits"]
    foreign = "GB" if h["country"] != "GB" else "US"
    device = "web_desktop" if h["device_type"] != "web_desktop" else "android"
    d = env["client"].post("/predict", json=_txn(
        h, transaction_amount=round(h["avg"] * 30, 2), merchant_category="jewelry",
        country=foreign, device_type=device, payment_method="credit_card")).json()
    assert d["decision"] == "BLOCK", d
    assert d["fraud_probability"] >= d["thresholds"]["block"]
    assert d["top_factors"] and all(f["contribution"] > 0 for f in d["top_factors"])


def test_validation_errors(env):
    r = env["client"].post("/predict", json={"user_id": "u", "transaction_amount": -1,
                                             "merchant_category": "weapons", "payment_method": "cash",
                                             "country": "SG", "device_type": "ios"})
    assert r.status_code == 422
    bad = {e["loc"][-1] for e in r.json()["detail"]}
    assert {"transaction_amount", "merchant_category", "payment_method"} <= bad


def test_predictions_are_logged(env):
    body = _txn(env["habits"], transaction_id="logged-txn-1")
    env["client"].post("/predict", json=body)
    with env["conn"].cursor() as cur:
        cur.execute("SELECT decision, model_version, latency_ms FROM predictions WHERE transaction_id=%s",
                    ("logged-txn-1",))
        row = cur.fetchone()
    env["conn"].commit()
    assert row and row[0] in {d.value for d in Decision} and row[1] == "1" and row[2] > 0


def test_decide_bands():
    t = {"block": 0.9, "review": 0.5}
    assert main.decide(0.95, t) == Decision.BLOCK
    assert main.decide(0.6, t) == Decision.REVIEW
    assert main.decide(0.1, t) == Decision.ALLOW


def test_hot_swap_and_feature_version_guard(env):
    c, mp = env["client"], env["mp"]

    # 1) A newly promoted version is picked up without restarting the API
    _train(mp, env["raw"], "--force-promote", "--seed", "5")
    assert c.post("/model/reload").json() == {"swapped": True, "version": "2"}
    assert c.post("/predict", json=_txn(env["habits"])).json()["model_version"] == "2"

    # 2) A model trained with different feature logic is refused; v2 keeps serving
    _train(mp, env["raw"], "--force-promote", "--seed", "6")
    client = MlflowClient(env["uri"])
    client.set_model_version_tag(train.MODEL_NAME, "3", "feature_version", "999")
    r = c.post("/model/reload").json()
    assert r == {"swapped": False, "version": "2"}
    assert "feature_version" in c.get("/model").json()["last_error"]
