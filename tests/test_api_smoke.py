"""Smoke test: the API application imports, starts and serves its basic routes.

Needs no MLflow server, no trained model and no Postgres. It points the API at an
empty throwaway MLflow registry, so this test only proves the app boots and
degrades gracefully (healthy, but not ready) when no model is available.
"""

import time

import pytest
from fastapi.testclient import TestClient

from src.api import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "TRACKING_URI", f"sqlite:///{tmp_path}/empty.db")
    monkeypatch.setattr(main, "REFRESH_SECONDS", 3600.0)
    with TestClient(main.app) as c:
        # let the background loader make its first (failing) attempt
        for _ in range(50):
            if main.state.models.last_checked:
                break
            time.sleep(0.1)
        yield c


def test_app_metadata():
    assert main.app.title == "E-commerce Fraud Scoring API"
    paths = {r.path for r in main.app.routes}
    assert {"/predict", "/health", "/ready", "/model", "/model/reload"} <= paths


def test_health_ok_without_model(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_missing_model(client):
    r = client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["ready"] is False and body["model_loaded"] is False
    assert "fraud-detector" in (body["last_error"] or "")


def test_predict_returns_503_not_500_without_model(client):
    r = client.post("/predict", json={
        "user_id": "user_00001", "transaction_amount": 42.0, "merchant_category": "grocery",
        "payment_method": "e_wallet", "country": "SG", "device_type": "ios",
    })
    assert r.status_code == 503


def test_openapi_schema_documents_predict(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "TransactionIn" in schema["components"]["schemas"]
