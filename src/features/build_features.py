"""Point-in-time feature engineering for fraud detection.

Every feature for a transaction uses ONLY that user's earlier transactions
(never the current row or anything later), so training matches what a
real-time scorer could know at decision time — no look-ahead leakage.

Feature groups
  * amount     - raw/log amount, ratio and z-score vs the user's history,
                 ratio vs the merchant category's running average
  * velocity   - # of the user's transactions in the past 10 min / 1 h / 24 h,
                 seconds since their previous transaction
  * behaviour  - first time seen with this country / device / payment method
  * time       - hour of day, night flag, day of week, in the LOCAL time of the
                 transaction's country (timestamps are stored in UTC)
  * categorical one-hots for merchant_category, payment_method, device_type
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_simulator import COUNTRY_TZ, DEVICES, MERCHANT_CATEGORIES, PAYMENT_METHODS

# Bump whenever feature LOGIC changes. Training stamps it on each model version and
# the API refuses to load a model built with a different version (training/serving skew).
FEATURE_VERSION = "2"  # v2: hour / is_night / day_of_week in the country's local time

# Fixed category lists keep the one-hot columns identical across training runs and serving
CATEGORICALS = {
    "merchant_category": list(MERCHANT_CATEGORIES),
    "payment_method": PAYMENT_METHODS,
    "device_type": DEVICES,
}
VELOCITY_WINDOWS = {"10min": "txn_count_10m", "1h": "txn_count_1h", "24h": "txn_count_24h"}
NO_HISTORY_SECONDS = 30 * 24 * 3600  # stand-in for "no previous transaction"

# Columns that must never be used as features (labels / label leaks / identifiers)
NON_FEATURES = {"transaction_id", "user_id", "timestamp", "is_fraud", "fraud_type",
                "country", "kafka_partition", "kafka_offset", "ingested_at"}


def _prior_rolling_count(df: pd.DataFrame, window: str) -> pd.Series:
    """Number of the user's transactions in the `window` before each transaction (excluding itself)."""
    counts = (
        df.set_index("timestamp")
        .groupby("user_id", sort=False)["transaction_amount"]
        .rolling(window, closed="left")   # 'left' => window excludes the current row
        .count()
    )
    # rolling output is ordered by (user_id, timestamp) and our df is sorted the same way
    return pd.Series(counts.to_numpy(), index=df.index).fillna(0)


def _first_time_seen(df: pd.DataFrame, col: str, has_history: pd.Series) -> pd.Series:
    """1 if the user has history but has never used this value of `col` before."""
    seen_before = df.groupby(["user_id", col], sort=False).cumcount() > 0
    return (has_history & ~seen_before).astype(int)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame aligned with `raw` (same index order after sorting by user/time).

    The returned frame keeps `transaction_id`, `timestamp` and `is_fraud` (if present)
    alongside the features so callers can split by time and pull out labels.
    """
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["transaction_amount"] = df["transaction_amount"].astype(float)
    df = df.sort_values(["user_id", "timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)

    g = df.groupby("user_id", sort=False)
    amt = df["transaction_amount"]
    out = pd.DataFrame(index=df.index)

    # ---- amount ------------------------------------------------------------
    out["amount"] = amt
    out["log_amount"] = np.log1p(amt)
    prior_n = g.cumcount()
    has_history = prior_n > 0
    out["user_prior_txn_count"] = prior_n
    out["is_first_txn"] = (~has_history).astype(int)

    prior_mean = g["transaction_amount"].transform(lambda s: s.shift(1).expanding().mean())
    prior_std = g["transaction_amount"].transform(lambda s: s.shift(1).expanding().std())
    prior_max = g["transaction_amount"].transform(lambda s: s.shift(1).expanding().max())
    out["user_avg_amount"] = prior_mean.fillna(-1)
    out["amount_to_user_avg"] = (amt / prior_mean).fillna(-1)
    out["amount_to_user_max"] = (amt / prior_max).fillna(-1)
    out["amount_zscore_user"] = ((amt - prior_mean) / prior_std.replace(0, np.nan)).clip(-10, 50).fillna(0)

    # Category baseline: running mean over ALL earlier transactions in that category
    by_time = df.sort_values(["timestamp", "transaction_id"], kind="mergesort")
    cat_prior_mean = by_time.groupby("merchant_category")["transaction_amount"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    out["amount_to_category_avg"] = (amt / cat_prior_mean.reindex(df.index)).fillna(1.0)

    # ---- velocity ----------------------------------------------------------
    for window, name in VELOCITY_WINDOWS.items():
        out[name] = _prior_rolling_count(df, window)
    secs = g["timestamp"].diff().dt.total_seconds()
    out["secs_since_last_txn"] = secs.fillna(NO_HISTORY_SECONDS)
    out["log_secs_since_last_txn"] = np.log1p(out["secs_since_last_txn"])

    # ---- behaviour change ---------------------------------------------------
    out["is_new_country"] = _first_time_seen(df, "country", has_history)
    out["is_new_device"] = _first_time_seen(df, "device_type", has_history)
    out["is_new_payment_method"] = _first_time_seen(df, "payment_method", has_history)

    # ---- time (local to the transaction's country) --------------------------
    hour = pd.Series(df["timestamp"].dt.hour, index=df.index)
    dow = pd.Series(df["timestamp"].dt.dayofweek, index=df.index)
    for country, idx in df.groupby("country", sort=False).groups.items():
        tz = COUNTRY_TZ.get(country)
        if tz:
            local = df.loc[idx, "timestamp"].dt.tz_convert(tz)
            hour.loc[idx] = local.dt.hour
            dow.loc[idx] = local.dt.dayofweek
    out["hour"] = hour
    out["is_night"] = hour.between(1, 5).astype(int)
    out["day_of_week"] = dow

    # ---- categoricals (fixed vocab) -----------------------------------------
    for col, vocab in CATEGORICALS.items():
        cat = pd.Categorical(df[col], categories=vocab)
        dummies = pd.get_dummies(cat, prefix=col, dtype=int)
        dummies.index = df.index
        out = pd.concat([out, dummies], axis=1)

    # All features as float64: keeps the MLflow input schema tolerant of missing values at serving time
    out = out.astype("float64")

    # ---- passthrough keys / label -------------------------------------------
    out.insert(0, "transaction_id", df["transaction_id"])
    out.insert(1, "timestamp", df["timestamp"])
    if "is_fraud" in df:
        out["is_fraud"] = df["is_fraud"].astype(int)
    return out


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in NON_FEATURES]


def build_online_features(history: pd.DataFrame, txn: dict,
                          category_means: dict[str, float] | None = None) -> pd.DataFrame:
    """Features for ONE incoming transaction, for real-time scoring.

    Runs the exact same `build_features` used in training over the user's earlier
    transactions plus the new one, and returns the new transaction's row. That
    guarantees training/serving parity for every per-user feature.

    The one cross-user feature, `amount_to_category_avg`, can't be derived from a
    single user's history, so the caller supplies category averages (the API
    caches them from Postgres). Without them it falls back to 1.0, the same value
    training uses for a category's first-ever transaction.
    """
    cols = ["transaction_id", "user_id", "transaction_amount", "merchant_category",
            "timestamp", "payment_method", "country", "device_type"]
    hist = history[cols] if len(history) else pd.DataFrame(columns=cols)
    frame = pd.concat([hist, pd.DataFrame([{c: txn[c] for c in cols}])], ignore_index=True)
    feats = build_features(frame)
    row = feats.loc[feats["transaction_id"] == txn["transaction_id"]].copy()

    cat_avg = (category_means or {}).get(txn["merchant_category"])
    row["amount_to_category_avg"] = float(txn["transaction_amount"]) / cat_avg if cat_avg else 1.0
    return row.reset_index(drop=True)
