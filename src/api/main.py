"""Real-time fraud scoring API.

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    open http://localhost:8000/docs

Request flow for POST /predict
  1. Validate the transaction payload
  2. Fetch the user's earlier transactions from Postgres
  3. Build features with the SAME code used in training (build_online_features)
  4. Score with the in-memory Production model from MLflow
  5. Map the probability to BLOCK / REVIEW / ALLOW and explain the top drivers
  6. Log the prediction to Postgres (after the response is sent)

The Production model is resolved from the MLflow registry (`production` alias,
falling back to the legacy 'Production' stage) at startup and re-checked every
MODEL_REFRESH_SECONDS; a newly promoted version is hot-swapped in without restart.

Configuration (env vars)
  MLFLOW_TRACKING_URI      default http://localhost:5000
  MLFLOW_MODEL_NAME        default fraud-detector
  MODEL_REFRESH_SECONDS    default 60
  BLOCK_THRESHOLD          default: the model's own F1-tuned threshold (stored on the version)
  REVIEW_THRESHOLD         default 0.5
  HISTORY_LIMIT            max past transactions per user used for features, default 500
  POSTGRES_*               see src/config.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

# Fail fast when MLflow is down instead of the client's default multi-minute retry backoff
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")

import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402
from fastapi import BackgroundTasks, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from src import config  # noqa: E402
from src.api.model_manager import LoadedModel, ModelManager  # noqa: E402
from src.api.schemas import Decision, Factor, PredictionOut, TransactionIn  # noqa: E402
from src.api.store import FeatureStore  # noqa: E402
from src.features.build_features import FEATURE_VERSION, build_online_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("api")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraud-detector")
REFRESH_SECONDS = float(os.getenv("MODEL_REFRESH_SECONDS", "60"))
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.5"))
BLOCK_THRESHOLD_OVERRIDE = os.getenv("BLOCK_THRESHOLD")
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "500"))
TOP_FACTORS = 3


class State:
    models: ModelManager
    store: FeatureStore | None = None


state = State()


def _connect_store() -> None:
    if state.store is None:
        try:
            state.store = FeatureStore(config.postgres_dsn(), history_limit=HISTORY_LIMIT)
            log.info("Connected to Postgres at %s:%s/%s", config.POSTGRES_HOST, config.POSTGRES_PORT,
                     config.POSTGRES_DB)
        except Exception as e:
            log.warning("Postgres unavailable: %s", e)


async def _refresh_loop() -> None:
    """Initial load right away, then re-check the registry every REFRESH_SECONDS.

    Runs in the background so the API comes up immediately even if MLflow or
    Postgres is down; /ready reports 503 until both are available.
    """
    while True:
        await asyncio.to_thread(_connect_store)
        await asyncio.to_thread(state.models.refresh)
        await asyncio.sleep(REFRESH_SECONDS if state.models.current else min(REFRESH_SECONDS, 5))


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.models = ModelManager(TRACKING_URI, MODEL_NAME)
    task = asyncio.create_task(_refresh_loop())
    yield
    task.cancel()
    if state.store:
        state.store.close()


app = FastAPI(
    title="E-commerce Fraud Scoring API",
    version="1.0.0",
    description="Scores transactions in real time with the Production model from the MLflow registry.",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- helpers
def thresholds_for(m: LoadedModel) -> dict[str, float]:
    block = float(BLOCK_THRESHOLD_OVERRIDE) if BLOCK_THRESHOLD_OVERRIDE else m.threshold
    review = min(REVIEW_THRESHOLD, block)  # review band can never sit above the block line
    return {"block": round(block, 6), "review": round(review, 6)}


def decide(prob: float, thresholds: dict[str, float]) -> Decision:
    if prob >= thresholds["block"]:
        return Decision.BLOCK
    if prob >= thresholds["review"]:
        return Decision.REVIEW
    return Decision.ALLOW


def top_factors(m: LoadedModel, X) -> list[Factor]:
    """Per-feature SHAP contributions from XGBoost (exact for tree models, ~1 ms)."""
    booster = m.model.get_booster()
    best = getattr(m.model, "best_iteration", None)
    kwargs = {"iteration_range": (0, best + 1)} if best is not None else {}
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True, **kwargs)[0][:-1]  # drop bias term
    order = np.argsort(contribs)[::-1][:TOP_FACTORS]
    return [Factor(feature=m.feature_columns[i], value=round(float(X.iloc[0, i]), 4),
                   contribution=round(float(contribs[i]), 4))
            for i in order if contribs[i] > 0]


# --------------------------------------------------------------------------- endpoints
@app.get("/health", tags=["ops"])
def health():
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready():
    """Readiness: a model is loaded and Postgres is reachable."""
    m = state.models.current
    db_ok = state.store is not None and state.store.ping()
    body = {"ready": bool(m and db_ok), "model_loaded": m is not None, "database": db_ok,
            "model_version": m.version if m else None, "last_error": state.models.last_error}
    return JSONResponse(body, status_code=200 if body["ready"] else 503)


@app.get("/model", tags=["ops"])
def model_info():
    m = state.models.current
    if m is None:
        raise HTTPException(503, detail=f"No model loaded: {state.models.last_error}")
    return {"name": m.name, "version": m.version, "run_id": m.run_id, "loaded_at": m.loaded_at,
            "feature_version": FEATURE_VERSION, "last_error": state.models.last_error,
            "last_checked": state.models.last_checked, "thresholds": thresholds_for(m),
            "n_features": len(m.feature_columns), "refresh_seconds": REFRESH_SECONDS}


@app.post("/model/reload", tags=["ops"])
def reload_model():
    """Check the registry now instead of waiting for the next refresh tick."""
    swapped = state.models.refresh()
    m = state.models.current
    if m is None:
        raise HTTPException(503, detail=f"No model loaded: {state.models.last_error}")
    return {"swapped": swapped, "version": m.version}


@app.post("/predict", response_model=PredictionOut, tags=["scoring"])
def predict(txn: TransactionIn, background: BackgroundTasks):
    t0 = time.perf_counter()
    m = state.models.current  # take one reference so a hot-swap mid-request can't mix models
    if m is None:
        raise HTTPException(503, detail=f"Model not loaded yet: {state.models.last_error}")
    if state.store is None:
        raise HTTPException(503, detail="Database unavailable")

    payload = txn.model_dump()
    history = state.store.user_history(txn.user_id, txn.timestamp, txn.transaction_id)
    feats = build_online_features(history, payload, state.store.category_means())
    X = feats[m.feature_columns]

    prob = float(m.model.predict_proba(X)[0, 1])
    thresholds = thresholds_for(m)
    decision = decide(prob, thresholds)
    factors = top_factors(m, X) if decision != Decision.ALLOW else []
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    background.add_task(state.store.log_prediction, txn.transaction_id, txn.user_id, m.name,
                        m.version, prob, decision.value, latency_ms)
    return PredictionOut(
        transaction_id=txn.transaction_id, fraud_probability=round(prob, 6), decision=decision,
        thresholds=thresholds, top_factors=factors, model_name=m.name, model_version=m.version,
        user_history_size=len(history), latency_ms=latency_ms,
    )
