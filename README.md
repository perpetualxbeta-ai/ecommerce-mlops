# ecommerce-mlops

Real-time fraud and churn detection for an e-commerce platform — an end-to-end MLOps practice project.

## Stack

| Service    | Purpose                                  | Port  |
|------------|------------------------------------------|-------|
| Zookeeper  | Kafka coordination                       | 2181  |
| Kafka      | Streaming transaction / event ingestion  | 9092  |
| PostgreSQL | App data + MLflow backend store          | 5432  |
| MLflow     | Experiment tracking & model registry     | 5000  |
| Fraud API  | Real-time scoring (FastAPI)              | 8000  |

## Layout

```
data/               raw and processed datasets (git-ignored)
notebooks/          exploration and prototyping
src/data_simulator.py   synthetic transaction generator
src/streaming/      Kafka producer + Kafka -> Postgres consumer
src/features/       point-in-time feature engineering
src/models/train.py XGBoost training, MLflow logging, registry promotion
src/api/            FastAPI real-time scoring service
tests/              pytest suite
docker/             service build files (MLflow image, Postgres init)
.github/workflows/  CI
```

## Quickstart

```bash
cp .env.example .env
docker compose up -d --build

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

- MLflow UI: http://localhost:5000
- Kafka bootstrap (from host): `localhost:9092` — from other containers: `kafka:29092`
- Postgres: `postgresql://mlops:mlops@localhost:5432/ecommerce`

## Streaming pipeline

```
data_simulator  ->  producer  ->  Kafka topic "transactions"  ->  consumer  ->  Postgres table "transactions"
```

Run each in its own terminal (with the venv activated and `docker compose up -d` running):

```bash
# 1. Consumer: reads the topic and writes raw events to Postgres
python -m src.streaming.consumer

# 2. Producer: streams synthetic transactions (5/sec by default)
python -m src.streaming.producer --rate 5
python -m src.streaming.producer --rate 100 --max 5000 --fraud-rate 0.05   # quick burst

# 3. Check what landed
docker compose exec postgres psql -U mlops -d ecommerce \
  -c "SELECT merchant_category, COUNT(*), ROUND(AVG(is_fraud)*100,2) AS fraud_pct
      FROM transactions GROUP BY 1 ORDER BY 2 DESC;"
```

Offline batch for notebooks / training:

```bash
python -m src.data_simulator --n 50000 --days 60 --out data/raw/transactions.csv
```

### Transaction schema

| Field | Notes |
|---|---|
| `transaction_id` | UUID, primary key (makes consumer writes idempotent) |
| `user_id` | Kafka message key, so one user's events stay ordered on one partition |
| `transaction_amount` | USD, log-normal per category, scaled by the user's spend profile |
| `merchant_category` | grocery, fashion, electronics, travel, gift_cards, ... |
| `timestamp` | ISO-8601 UTC |
| `is_fraud` | 0/1 label (~2% by default) |
| `payment_method`, `country`, `device_type` | context fields; fraud often deviates from the user's usual values |
| `fraud_type` | which pattern generated it (card_testing, account_takeover, high_risk_spend, odd_hours). **For analysis only — never use as a model feature, it leaks the label.** |

### Delivery guarantees

The consumer commits Kafka offsets only after the Postgres transaction commits (at-least-once), and inserts with `ON CONFLICT (transaction_id) DO NOTHING`, so replays after a crash don't create duplicates. Malformed messages are logged and skipped.

## Training the fraud model

```bash
# Need history first: either let the producer/consumer run for a while, or backfill
python -m src.data_simulator --n 60000 --days 60 --to-postgres

python -m src.models.train                    # loads latest 200k rows, 6 hyper-param trials
python -m src.models.train --n-trials 12 --limit 100000
```

What it does:

1. **Features** (`src/features/build_features.py`): each transaction only sees the user's *earlier* transactions:
   - velocity: count in the past 10 min / 1 h / 24 h, seconds since the last transaction
   - amount vs. the user's running mean, max and z-score, and vs. the category's running mean
   - first-time country, device or payment method; one-hot categories
   - hour of day, night flag and weekday, in the transaction country's **local** time
2. **Time-based split**: oldest 70% train, next 15% validation, newest 15% test (no shuffling, so no peeking at the future).
3. **Search**: each hyper-parameter candidate is a nested MLflow run. The best candidate is chosen on validation PR-AUC, and its decision threshold is tuned on validation for max F1.
4. **Logging**: params, `precision` / `recall` / `f1` (test), PR-AUC, ROC-AUC, confusion counts, PR-curve and feature-importance plots, and the model with its input signature.
5. **Registry**: every run registers a new version of `fraud-detector`. The current Production model is re-scored on the *same* test set, and the new version is promoted only if its F1 is at least as good (`--force-promote` overrides).
   - Promotion sets the **`production` alias**, which is MLflow's current mechanism; load the model with `models:/fraud-detector@production`.
   - It also moves the version to the legacy **"Production" stage**, which still shows in the UI.
   - The decision threshold and feature list are stored as tags on each model version.

Results on 60k simulated transactions: test precision ≈ 0.89, recall ≈ 0.84, F1 ≈ 0.87.

## Real-time scoring API

```bash
uvicorn src.api.main:app --port 8000      # local, or: docker compose up -d --build api
open http://localhost:8000/docs           # interactive Swagger UI
```

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "user_id": "user_00111", "transaction_amount": 2900, "merchant_category": "jewelry",
  "payment_method": "credit_card", "country": "GB", "device_type": "web_desktop"
}'
```

```json
{
  "transaction_id": "5c1e…",
  "fraud_probability": 0.997741,
  "decision": "BLOCK",
  "thresholds": {"block": 0.915123, "review": 0.5},
  "top_factors": [
    {"feature": "amount_to_category_avg", "value": 9.425,  "contribution": 2.31},
    {"feature": "amount_to_user_avg",     "value": 31.028, "contribution": 1.12},
    {"feature": "is_new_country",         "value": 1.0,    "contribution": 0.64}
  ],
  "model_name": "fraud-detector", "model_version": "5",
  "user_history_size": 82, "latency_ms": 88.4
}
```

How a request is scored:

1. The user's earlier transactions (up to `HISTORY_LIMIT`) are read from Postgres.
2. Features are built with the **same code as training** (`build_online_features`); a test checks that online and offline features match exactly.
3. The in-memory Production model scores the transaction:
   - **BLOCK** if the score is at or above the model's own F1-tuned threshold (stored on the model version)
   - **REVIEW** if the score is at or above `REVIEW_THRESHOLD` (default 0.5)
   - **ALLOW** otherwise
4. `top_factors` lists the features that pushed the score up most (XGBoost SHAP contributions); it is filled for BLOCK and REVIEW decisions.
5. Each prediction is written to the `predictions` table after the response is sent.

On the simulated test set, BLOCK catches 84% of fraud at 89% precision, and adding REVIEW raises that to about 95% of fraud while sending about 2% of traffic to review.

**Model lifecycle**
- The API resolves `models:/fraud-detector@production`, falling back to the legacy "Production" stage, and re-checks every `MODEL_REFRESH_SECONDS` (default 60). A newly promoted model is hot-swapped in without a restart. `POST /model/reload` forces a check.
- **Feature-version guard:** each model version is tagged with the `FEATURE_VERSION` it was trained with. If the feature code changes, the API refuses to load a model that doesn't match and keeps serving the previous one. Without this, swapping in a model trained on new feature logic would silently skew predictions. Bump `FEATURE_VERSION` in `build_features.py` whenever feature logic changes.
- If MLflow or Postgres is down, the API still starts. `/health` returns 200 and `/ready` returns 503 until both are available, at which point the model loads automatically.

| Endpoint | Purpose |
|---|---|
| `POST /predict` | score a transaction |
| `GET /health` | liveness |
| `GET /ready` | readiness (model loaded + DB reachable) |
| `GET /model` | serving model version, thresholds, last refresh |
| `POST /model/reload` | check the registry now |

> **Tests use a separate database.** `tests/conftest.py` points every test at `<POSTGRES_DB>_test`, so running `pytest` never touches your working data.

> If you started the stack before the `transactions` schema changed, reset the database volume once: `docker compose down -v && docker compose up -d --build`.
