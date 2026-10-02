-- Separate database for MLflow tracking metadata
CREATE DATABASE mlflow;

-- Application tables live in the default "ecommerce" database
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id   VARCHAR(64) PRIMARY KEY,
    customer_id      VARCHAR(64) NOT NULL,
    amount           NUMERIC(12, 2) NOT NULL,
    currency         VARCHAR(8) DEFAULT 'USD',
    payment_method   VARCHAR(32),
    country          VARCHAR(8),
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    id               SERIAL PRIMARY KEY,
    transaction_id   VARCHAR(64),
    customer_id      VARCHAR(64),
    model_name       VARCHAR(64) NOT NULL,
    model_version    VARCHAR(32),
    score            DOUBLE PRECISION NOT NULL,
    label            SMALLINT,
    predicted_at     TIMESTAMPTZ DEFAULT NOW()
);
