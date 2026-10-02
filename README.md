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
src/features/       feature engineering
src/models/         training, evaluation, registry
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
