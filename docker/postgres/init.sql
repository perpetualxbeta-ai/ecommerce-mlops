-- Separate database for MLflow tracking metadata
CREATE DATABASE mlflow;

-- Application tables live in the default "ecommerce" database
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id     VARCHAR(64) PRIMARY KEY,
    user_id            VARCHAR(64)    NOT NULL,
    transaction_amount NUMERIC(12, 2) NOT NULL,
    merchant_category  VARCHAR(32)    NOT NULL,
    timestamp          TIMESTAMPTZ    NOT NULL,
    is_fraud           SMALLINT       NOT NULL,
    payment_method     VARCHAR(32),
    country            VARCHAR(8),
    device_type        VARCHAR(32),
    fraud_type         VARCHAR(32),
    kafka_partition    INT,
    kafka_offset       BIGINT,
    ingested_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_transactions_user_ts ON transactions (user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_ts ON transactions (timestamp);

CREATE TABLE IF NOT EXISTS predictions (
    id               SERIAL PRIMARY KEY,
    transaction_id   VARCHAR(64),
    user_id          VARCHAR(64),
    model_name       VARCHAR(64) NOT NULL,
    model_version    VARCHAR(32),
    score            DOUBLE PRECISION NOT NULL,
    label            SMALLINT,
    predicted_at     TIMESTAMPTZ DEFAULT NOW()
);
