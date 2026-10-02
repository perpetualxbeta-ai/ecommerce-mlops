from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.data_simulator import COUNTRIES, DEVICES, MERCHANT_CATEGORIES, PAYMENT_METHODS

MerchantCategory = Literal[tuple(MERCHANT_CATEGORIES)]  # type: ignore[valid-type]
PaymentMethod = Literal[tuple(PAYMENT_METHODS)]  # type: ignore[valid-type]
DeviceType = Literal[tuple(DEVICES)]  # type: ignore[valid-type]


class Decision(StrEnum):
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"
    ALLOW = "ALLOW"


class TransactionIn(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=64)
    user_id: str = Field(..., min_length=1, max_length=64, examples=["user_00042"])
    transaction_amount: float = Field(..., gt=0, lt=1_000_000, examples=[1899.0])
    merchant_category: MerchantCategory = Field(..., examples=["electronics"])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC),
                                description="Defaults to now (UTC). Naive datetimes are treated as UTC.")
    payment_method: PaymentMethod = Field(..., examples=["credit_card"])
    country: str = Field(..., min_length=2, max_length=8, examples=["SG"],
                         description=f"ISO country code; simulator uses {', '.join(COUNTRIES)}")
    device_type: DeviceType = Field(..., examples=["web_desktop"])

    @field_validator("timestamp")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=UTC) if v.tzinfo is None else v.astimezone(UTC)

    @field_validator("country")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class Factor(BaseModel):
    feature: str
    value: float
    contribution: float = Field(description="Push toward fraud in log-odds (XGBoost SHAP value)")


class PredictionOut(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: Decision
    thresholds: dict[str, float]
    top_factors: list[Factor]
    model_name: str
    model_version: str
    user_history_size: int
    latency_ms: float
