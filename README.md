# ecommerce-mlops

Real-time fraud and churn detection for an e-commerce platform — an end-to-end MLOps practice project.

## Stack

| Service    | Purpose                                  | Port  |
|------------|------------------------------------------|-------|
| Zookeeper  | Kafka coordination                       | 2181  |
| Kafka      | Streaming transaction / event ingestion  | 9092  |
| PostgreSQL | App data + MLflow backend store          | 5432  |
| MLflow     | Experiment tracking & model registry     | 5000  |

## Layout

```
data/               raw and processed datasets (git-ignored)
notebooks/          exploration and prototyping
src/data_simulator.py   synthetic transaction generator
src/streaming/      Kafka producer + Kafka -> Postgres consumer
src/features/       point-in-time feature engineering
src/models/train.py XGBoost training, MLflow logging, registry promotion
src/api/            FastAPI scoring service
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
   - first-time country, device or payment method; hour of day / night flag; one-hot categories
2. **Time-based split**: oldest 70% train, next 15% validation, newest 15% test (no shuffling, so no peeking at the future).
3. **Search**: each hyper-parameter candidate is a nested MLflow run. The best candidate is chosen on validation PR-AUC, and its decision threshold is tuned on validation for max F1.
4. **Logging**: params, `precision` / `recall` / `f1` (test), PR-AUC, ROC-AUC, confusion counts, PR-curve and feature-importance plots, and the model with its input signature.
5. **Registry**: every run registers a new version of `fraud-detector`. The current Production model is re-scored on the *same* test set, and the new version is promoted only if its F1 is at least as good (`--force-promote` overrides).
   - Promotion sets the **`production` alias**, which is MLflow's current mechanism; load the model with `models:/fraud-detector@production`.
   - It also moves the version to the legacy **"Production" stage**, which still shows in the UI.
   - The decision threshold and feature list are stored as tags on each model version.

Results on 60k simulated transactions: test precision ≈ 0.89, recall ≈ 0.86, F1 ≈ 0.87.

> If you started the stack before the `transactions` schema changed, reset the database volume once: `docker compose down -v && docker compose up -d --build`.
