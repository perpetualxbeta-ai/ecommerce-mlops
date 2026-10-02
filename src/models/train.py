"""Train an XGBoost fraud classifier from Postgres data and track it in MLflow.

Pipeline
  1. Load the most recent N transactions from Postgres
  2. Build point-in-time features (src/features/build_features.py)
  3. Split by TIME: oldest 70% train / next 15% validation / newest 15% test
  4. Small hyper-parameter search; each candidate is a nested MLflow run
  5. Pick the best candidate on validation PR-AUC, tune its decision threshold
     on validation for max F1, then evaluate once on the untouched test set
  6. Log params, metrics (precision / recall / F1 / PR-AUC / ROC-AUC), plots
     and the model to MLflow, and register it as a new model version
  7. Champion vs challenger: score the current Production model on the SAME
     test set; promote the new version only if its F1 is at least as good.
     Promotion sets the `production` alias (MLflow's current mechanism) and
     moves the version to the legacy "Production" stage for UI visibility.

Usage:
    python -m src.models.train                          # defaults
    python -m src.models.train --limit 100000 --n-trials 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import tempfile
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier  # noqa: E402

from src import config  # noqa: E402
from src.features.build_features import build_features, feature_columns  # noqa: E402

log = logging.getLogger("train")

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "fraud-detection")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraud-detector")
PROD_ALIAS = "production"
PROD_STAGE = "Production"

SEARCH_SPACE = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 400],
    "min_child_weight": [1, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


# --------------------------------------------------------------------------- data
def load_transactions(limit: int) -> pd.DataFrame:
    sql = """
        SELECT transaction_id, user_id, transaction_amount, merchant_category, timestamp,
               is_fraud, payment_method, country, device_type
        FROM transactions
        ORDER BY timestamp DESC
        LIMIT %s
    """
    with psycopg2.connect(config.postgres_dsn()) as conn:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # pandas prefers SQLAlchemy; psycopg2 is fine here
            df = pd.read_sql(sql, conn, params=(limit,))
    if df.empty:
        raise SystemExit("No transactions in Postgres. Run the producer+consumer, or backfill with:\n"
                         "  python -m src.data_simulator --n 50000 --to-postgres")
    return df.sort_values("timestamp").reset_index(drop=True)


def time_split(features: pd.DataFrame, val_frac: float, test_frac: float):
    f = features.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    n = len(f)
    i_val, i_test = int(n * (1 - val_frac - test_frac)), int(n * (1 - test_frac))
    return f.iloc[:i_val], f.iloc[i_val:i_test], f.iloc[i_test:]


# --------------------------------------------------------------------------- metrics
def best_f1_threshold(y_true, scores) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])]) if len(thresholds) else 0.5


def evaluate(y_true, scores, threshold: float, prefix: str) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}pr_auc": average_precision_score(y_true, scores),
        f"{prefix}roc_auc": roc_auc_score(y_true, scores),
        f"{prefix}tp": int(tp), f"{prefix}fp": int(fp), f"{prefix}fn": int(fn), f"{prefix}tn": int(tn),
    }


def plot_pr_curve(y_true, scores, threshold: float, path: str) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    idx = np.searchsorted(thresholds, threshold)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    if idx < len(thresholds):
        ax.scatter([recall[idx]], [precision[idx]], color="red", zorder=3, label=f"threshold={threshold:.3f}")
        ax.legend()
    ax.set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curve", xlim=(0, 1), ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance(model: XGBClassifier, cols: list[str], path: str, top: int = 20) -> None:
    imp = pd.Series(model.feature_importances_, index=cols).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(6, 6))
    imp.plot.barh(ax=ax)
    ax.set_title("Feature importance (gain)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- training
def sample_params(n_trials: int, seed: int) -> list[dict]:
    grid = [dict(zip(SEARCH_SPACE, combo)) for combo in itertools.product(*SEARCH_SPACE.values())]
    return random.Random(seed).sample(grid, k=min(n_trials, len(grid)))


def fit(params: dict, X_train, y_train, X_val, y_val, seed: int) -> XGBClassifier:
    pos = max(int(y_train.sum()), 1)
    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=(len(y_train) - pos) / pos,  # counter the ~2% class imbalance
        early_stopping_rounds=30,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


# --------------------------------------------------------------------------- registry
def current_production(client: MlflowClient):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, PROD_ALIAS)
    except mlflow.exceptions.MlflowException:
        return None


def score_production_on(prod_version, X_test: pd.DataFrame, y_test) -> float | None:
    """F1 of the current Production model on this run's test set (None if it can't be scored)."""
    try:
        model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@{PROD_ALIAS}")
        threshold = float(prod_version.tags.get("threshold", 0.5))
        cols = json.loads(prod_version.tags["feature_columns"])
        scores = model.predict_proba(X_test[cols])[:, 1]
        return f1_score(y_test, (scores >= threshold).astype(int), zero_division=0)
    except Exception as e:  # e.g. feature set changed between versions
        log.warning("Could not score current Production model on new test set (%s); treating as no champion", e)
        return None


def promote(client: MlflowClient, version: str) -> None:
    client.set_registered_model_alias(MODEL_NAME, PROD_ALIAS, version)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # stages are deprecated since MLflow 2.9 but still shown in the UI
        client.transition_model_version_stage(MODEL_NAME, version, PROD_STAGE, archive_existing_versions=True)


# --------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description="Train XGBoost fraud model and log to MLflow")
    p.add_argument("--limit", type=int, default=200_000, help="most recent N transactions to load")
    p.add_argument("--n-trials", type=int, default=6, help="hyper-parameter candidates to try")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    p.add_argument("--force-promote", action="store_true", help="promote even if it doesn't beat Production")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    raw = load_transactions(args.limit)
    feats = build_features(raw)
    cols = feature_columns(feats)
    train, val, test = time_split(feats, args.val_frac, args.test_frac)
    X_train, y_train = train[cols], train["is_fraud"]
    X_val, y_val = val[cols], val["is_fraud"]
    X_test, y_test = test[cols], test["is_fraud"]
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        log.info("%-5s rows=%d fraud=%d (%.2f%%)", name, len(y), y.sum(), 100 * y.mean())
    if min(y_train.sum(), y_val.sum(), y_test.sum()) < 5:
        raise SystemExit("Too few fraud examples in a split — load more data (see --limit) and retry.")

    with mlflow.start_run(run_name="xgb-fraud") as parent:
        mlflow.set_tags({"model_type": "xgboost", "task": "fraud_detection"})
        mlflow.log_params({
            "rows": len(feats), "n_features": len(cols), "n_trials": args.n_trials,
            "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
            "train_start": str(train.timestamp.min()), "test_end": str(test.timestamp.max()),
            "train_fraud_rate": round(float(y_train.mean()), 4),
        })

        # ---- hyper-parameter search (nested runs) -------------------------------
        best = None
        for i, params in enumerate(sample_params(args.n_trials, args.seed)):
            with mlflow.start_run(run_name=f"trial-{i}", nested=True):
                model = fit(params, X_train, y_train, X_val, y_val, args.seed)
                val_scores = model.predict_proba(X_val)[:, 1]
                thr = best_f1_threshold(y_val, val_scores)
                m = evaluate(y_val, val_scores, thr, "val_")
                mlflow.log_params({**params, "best_iteration": model.best_iteration})
                mlflow.log_metrics(m)
                log.info("trial %d  val_pr_auc=%.4f val_f1=%.4f  %s", i, m["val_pr_auc"], m["val_f1"], params)
                if best is None or m["val_pr_auc"] > best[2]["val_pr_auc"]:
                    best = (params, model, m, thr)

        params, model, val_metrics, threshold = best
        test_scores = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(y_test, test_scores, threshold, "test_")

        mlflow.log_params({**params, "best_iteration": model.best_iteration, "threshold": round(threshold, 6)})
        mlflow.log_metrics({**val_metrics, **test_metrics})
        # Headline metrics without prefix, so they're easy to compare in the MLflow UI
        mlflow.log_metrics({k: test_metrics[f"test_{k}"] for k in ("precision", "recall", "f1")})

        with tempfile.TemporaryDirectory() as tmp:
            plot_pr_curve(y_test, test_scores, threshold, f"{tmp}/pr_curve.png")
            plot_importance(model, cols, f"{tmp}/feature_importance.png")
            with open(f"{tmp}/feature_columns.json", "w") as fh:
                json.dump(cols, fh, indent=2)
            mlflow.log_artifacts(tmp)

        signature = infer_signature(X_test.head(100), test_scores[:100])
        info = mlflow.xgboost.log_model(
            model, name="model", signature=signature, input_example=X_test.head(5),
            registered_model_name=MODEL_NAME,
        )
        version = str(info.registered_model_version)
        client.set_model_version_tag(MODEL_NAME, version, "threshold", f"{threshold:.6f}")
        client.set_model_version_tag(MODEL_NAME, version, "feature_columns", json.dumps(cols))
        client.set_model_version_tag(MODEL_NAME, version, "test_f1", f"{test_metrics['test_f1']:.4f}")

        # ---- champion vs challenger ---------------------------------------------
        new_f1 = test_metrics["test_f1"]
        prod = current_production(client)
        prod_f1 = score_production_on(prod, X_test, y_test) if prod else None
        promoted = args.force_promote or prod_f1 is None or new_f1 >= prod_f1
        if prod_f1 is not None:
            mlflow.log_metric("champion_test_f1", prod_f1)
        if promoted:
            promote(client, version)
            reason = "forced" if args.force_promote else (
                "no existing Production model" if prod_f1 is None
                else f"F1 {new_f1:.4f} >= champion v{prod.version} F1 {prod_f1:.4f}")
        else:
            reason = f"F1 {new_f1:.4f} < champion v{prod.version} F1 {prod_f1:.4f}"
        mlflow.set_tags({"registered_version": version, "promoted": str(promoted), "promotion_reason": reason})

    log.info("Test  precision=%.4f recall=%.4f f1=%.4f pr_auc=%.4f (threshold=%.3f)",
             test_metrics["test_precision"], test_metrics["test_recall"], new_f1,
             test_metrics["test_pr_auc"], threshold)
    log.info("Registered %s v%s — %s: %s", MODEL_NAME, version,
             "PROMOTED to Production" if promoted else "kept as challenger", reason)
    log.info("Run: %s/#/experiments/%s/runs/%s", args.tracking_uri, parent.info.experiment_id, parent.info.run_id)


if __name__ == "__main__":
    main()
