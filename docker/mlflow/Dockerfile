FROM ghcr.io/mlflow/mlflow:v3.16.1
# The official image doesn't ship a Postgres driver; MLflow needs it for the backend store
RUN pip install --no-cache-dir psycopg2-binary
