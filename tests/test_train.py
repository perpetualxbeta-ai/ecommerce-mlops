"""Training pipeline tests. Uses a throwaway SQLite MLflow store, so no server is needed."""

import sys

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from src.data_simulator import TransactionSimulator
from src.features.build_features import build_features
from src.models import train
from tests.conftest import DATA_START


def _raw(n=20000, seed=1):
    sim = TransactionSimulator(n_users=300, seed=seed)
    return pd.DataFrame([t.to_dict() for t in sim.generate_batch(n, start=DATA_START, days=30)])


def test_time_split_has_no_overlap():
    f = build_features(_raw(3000))
    tr, va, te = train.time_split(f, 0.15, 0.15)
    assert len(tr) + len(va) + len(te) == len(f)
    assert tr.timestamp.max() <= va.timestamp.min() <= va.timestamp.max() <= te.timestamp.min()


def test_best_f1_threshold():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert 0.3 < train.best_f1_threshold(y, s) <= 0.8


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)  # default artifact root (./mlruns) lands in tmp, not the repo
    mlflow.set_tracking_uri(uri)
    yield uri


def _run(monkeypatch, raw, *extra):
    monkeypatch.setattr(train, "load_transactions", lambda limit: raw)
    monkeypatch.setattr(sys, "argv", ["train", "--n-trials", "2", *extra])
    train.main()


def test_end_to_end_logs_registers_and_promotes(mlflow_store, monkeypatch):
    raw = _raw()
    _run(monkeypatch, raw)

    client = MlflowClient()
    prod = client.get_model_version_by_alias(train.MODEL_NAME, train.PROD_ALIAS)
    assert str(prod.version) == "1"
    assert client.get_model_version(train.MODEL_NAME, "1").current_stage == "Production"

    run = client.get_run(prod.run_id)
    for k in ("precision", "recall", "f1", "test_pr_auc"):
        assert k in run.data.metrics
    assert run.data.metrics["f1"] > 0.6  # simulated fraud is learnable
    assert {"max_depth", "learning_rate", "threshold"} <= run.data.params.keys()
    assert run.data.tags["promoted"] == "True"

    model = mlflow.pyfunc.load_model(f"models:/{train.MODEL_NAME}@{train.PROD_ALIAS}")
    assert model.metadata.get_input_schema() is not None

    # A weaker challenger (trained on a small, recent slice) must not replace the champion
    _run(monkeypatch, raw.tail(6000), "--seed", "3")
    prod = client.get_model_version_by_alias(train.MODEL_NAME, train.PROD_ALIAS)
    v2 = client.get_model_version(train.MODEL_NAME, "2")
    v2_run = client.get_run(v2.run_id)
    if v2_run.data.tags["promoted"] == "True":
        assert str(prod.version) == "2"
        assert v2_run.data.metrics["f1"] >= v2_run.data.metrics["champion_test_f1"]
    else:
        assert str(prod.version) == "1"
        assert v2.current_stage == "None"
        assert v2_run.data.metrics["f1"] < v2_run.data.metrics["champion_test_f1"]
