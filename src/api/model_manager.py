"""Keeps the current Production fraud model in memory and hot-swaps it on promotion.

Loading a model from MLflow takes seconds, so it is never done per request.
Instead a background loop asks the registry every MODEL_REFRESH_SECONDS which
version holds the `production` alias (a cheap metadata call) and only downloads
when that version changes. The new model is fully loaded before it replaces the
old one, so in-flight requests never see a half-loaded model.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import mlflow
from mlflow import MlflowClient
from xgboost import XGBClassifier

from src.features.build_features import FEATURE_VERSION

log = logging.getLogger("api.model")


@dataclass(frozen=True)
class LoadedModel:
    model: XGBClassifier
    name: str
    version: str
    run_id: str
    threshold: float            # F1-optimal threshold tuned at training time
    feature_columns: list[str]
    loaded_at: str


class IncompatibleModelError(RuntimeError):
    pass


class ModelManager:
    def __init__(self, tracking_uri: str, model_name: str, alias: str = "production",
                 fallback_stage: str = "Production"):
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.alias = alias
        self.fallback_stage = fallback_stage
        self._current: LoadedModel | None = None
        self._lock = threading.Lock()
        self.last_error: str | None = None
        self.last_checked: str | None = None
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)

    @property
    def current(self) -> LoadedModel | None:
        return self._current

    def _resolve_production_version(self):
        """Version behind the `production` alias; falls back to the legacy 'Production' stage."""
        try:
            return self._client.get_model_version_by_alias(self.model_name, self.alias)
        except mlflow.exceptions.MlflowException:
            versions = self._client.search_model_versions(f"name='{self.model_name}'")
            staged = [v for v in versions if v.current_stage == self.fallback_stage]
            if not staged:
                raise LookupError(f"No '{self.alias}' alias or '{self.fallback_stage}' stage "
                                  f"for model '{self.model_name}'. Train one: python -m src.models.train") from None
            return max(staged, key=lambda v: int(v.version))

    def refresh(self, force: bool = False) -> bool:
        """Load the Production model if it changed. Returns True if a new model was swapped in."""
        self.last_checked = datetime.now(UTC).isoformat()
        try:
            mv = self._resolve_production_version()
            version = str(mv.version)
            if not force and self._current and self._current.version == version:
                self.last_error = None
                return False
            model_fv = mv.tags.get("feature_version")
            if model_fv != FEATURE_VERSION:
                raise IncompatibleModelError(
                    f"{self.model_name} v{version} was trained with feature_version={model_fv} but this API "
                    f"computes feature_version={FEATURE_VERSION}. Redeploy the API with matching feature "
                    f"code (or retrain); keeping the current model.")
            t0 = time.perf_counter()
            model = mlflow.xgboost.load_model(f"models:/{self.model_name}/{version}")
            loaded = LoadedModel(
                model=model, name=self.model_name, version=version, run_id=mv.run_id,
                threshold=float(mv.tags.get("threshold", 0.5)),
                feature_columns=json.loads(mv.tags["feature_columns"]),
                loaded_at=datetime.now(UTC).isoformat(),
            )
            with self._lock:
                previous = self._current
                self._current = loaded
            self.last_error = None
            log.info("Loaded %s v%s in %.1fs (previous: %s)", self.model_name, version,
                     time.perf_counter() - t0, f"v{previous.version}" if previous else "none")
            return True
        except Exception as e:  # keep serving the old model if the registry is unreachable
            self.last_error = f"{type(e).__name__}: {e}"
            log.warning("Model refresh failed: %s", self.last_error)
            return False
