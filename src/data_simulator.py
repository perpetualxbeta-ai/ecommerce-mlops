"""Synthetic e-commerce transaction generator.

Each simulated user has a stable profile (home country, typical spend, preferred
categories, usual device), so normal behaviour is consistent per user and fraud
shows up as a *deviation* from it — which is what makes the data learnable.

Fraud patterns injected (~2% of transactions by default, set with --fraud-rate):
  * card_testing     - burst of tiny amounts, often at digital_goods
  * account_takeover - large purchase from a new device in a foreign country
  * high_risk_spend  - unusually large amount in electronics / gift_cards / jewelry
  * odd_hours        - large purchase between 01:00 and 05:00 local time

Usage:
    python -m src.data_simulator --n 10000 --out data/raw/transactions.csv
    python -m src.data_simulator --n 50000 --days 60 --to-postgres   # backfill training history
"""

from __future__ import annotations

import argparse
import random
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

MERCHANT_CATEGORIES: dict[str, tuple[float, float]] = {
    # category: (median amount USD, spread) — amounts are log-normal around the median
    "grocery": (45.0, 0.5),
    "fashion": (70.0, 0.6),
    "electronics": (250.0, 0.8),
    "home_garden": (90.0, 0.7),
    "beauty": (35.0, 0.5),
    "sports": (60.0, 0.6),
    "books_media": (20.0, 0.5),
    "travel": (400.0, 0.7),
    "digital_goods": (15.0, 0.6),
    "gift_cards": (50.0, 0.4),
    "jewelry": (180.0, 0.8),
}
HIGH_RISK_CATEGORIES = ["electronics", "gift_cards", "jewelry"]
COUNTRIES = ["SG", "MY", "ID", "TH", "PH", "VN", "US", "GB", "AU", "JP"]
# Shopping-hour patterns are in each country's LOCAL time; timestamps are stored in UTC
COUNTRY_TZ = {
    "SG": "Asia/Singapore", "MY": "Asia/Kuala_Lumpur", "ID": "Asia/Jakarta", "TH": "Asia/Bangkok",
    "PH": "Asia/Manila", "VN": "Asia/Ho_Chi_Minh", "US": "America/New_York", "GB": "Europe/London",
    "AU": "Australia/Sydney", "JP": "Asia/Tokyo",
}
PAYMENT_METHODS = ["credit_card", "debit_card", "e_wallet", "bnpl", "bank_transfer"]
DEVICES = ["ios", "android", "web_desktop", "web_mobile"]
FRAUD_TYPES = ["card_testing", "account_takeover", "high_risk_spend", "odd_hours"]
FRAUD_TYPE_WEIGHTS = [25, 30, 30, 15]
CARD_TESTING_BURST = (3, 6)  # min/max transactions per card-testing attack


@dataclass
class UserProfile:
    user_id: str
    home_country: str
    spend_multiplier: float
    favourite_categories: list[str]
    usual_device: str
    usual_payment: str


@dataclass
class Transaction:
    transaction_id: str
    user_id: str
    transaction_amount: float
    merchant_category: str
    timestamp: str  # ISO-8601, UTC
    is_fraud: int
    payment_method: str
    country: str
    device_type: str
    fraud_type: str | None = field(default=None)

    def to_dict(self) -> dict:
        return asdict(self)


class TransactionSimulator:
    def __init__(self, n_users: int = 1000, fraud_rate: float = 0.02, seed: int | None = None):
        self.rng = random.Random(seed)
        self.fraud_rate = fraud_rate
        # `fraud_rate` is the target share of fraudulent *transactions*. Card-testing
        # attacks emit several transactions each, so start attacks less often to compensate.
        share = FRAUD_TYPE_WEIGHTS[0] / sum(FRAUD_TYPE_WEIGHTS)
        txns_per_attack = share * sum(CARD_TESTING_BURST) / 2 + (1 - share)
        self._attack_prob = fraud_rate / (txns_per_attack * (1 - fraud_rate) + fraud_rate)
        self.users = [self._make_user(i) for i in range(n_users)]
        # A card-testing attack emits several transactions; queue the rest here
        self._pending: list[Transaction] = []

    # ---- profiles -------------------------------------------------------
    def _make_user(self, i: int) -> UserProfile:
        r = self.rng
        return UserProfile(
            user_id=f"user_{i:05d}",
            home_country=r.choices(COUNTRIES, weights=[30, 15, 12, 10, 8, 8, 6, 4, 4, 3])[0],
            spend_multiplier=r.lognormvariate(0, 0.4),
            favourite_categories=r.sample(list(MERCHANT_CATEGORIES), k=3),
            usual_device=r.choice(DEVICES),
            usual_payment=r.choices(PAYMENT_METHODS, weights=[35, 25, 25, 10, 5])[0],
        )

    # ---- helpers --------------------------------------------------------
    def _amount(self, category: str, multiplier: float = 1.0) -> float:
        median, spread = MERCHANT_CATEGORIES[category]
        return round(max(0.5, median * multiplier * self.rng.lognormvariate(0, spread)), 2)

    def _normal_hour(self) -> int:
        # Most shopping happens 08:00-23:00
        return self.rng.choices(range(24), weights=[2, 1, 1, 1, 1, 1, 2, 4, 6, 7, 7, 8,
                                                    9, 8, 7, 7, 8, 9, 10, 11, 11, 10, 7, 4])[0]

    def _ts(self, base: datetime, local_hour: int, country: str) -> datetime:
        """`base`'s date at `local_hour` in `country`'s timezone (returned tz-aware)."""
        local = base.astimezone(ZoneInfo(COUNTRY_TZ.get(country, "UTC")))
        return local.replace(hour=local_hour, minute=self.rng.randint(0, 59), second=self.rng.randint(0, 59))

    def _txn(self, user: UserProfile, ts: datetime, **kw) -> Transaction:
        return Transaction(
            transaction_id=str(uuid.UUID(int=self.rng.getrandbits(128))),
            user_id=user.user_id,
            timestamp=ts.astimezone(UTC).isoformat(),
            **kw,
        )

    # ---- generators -----------------------------------------------------
    def _legit(self, user: UserProfile, ts: datetime) -> Transaction:
        r = self.rng
        category = r.choice(user.favourite_categories) if r.random() < 0.75 else r.choice(list(MERCHANT_CATEGORIES))
        return self._txn(
            user, ts,
            transaction_amount=self._amount(category, user.spend_multiplier),
            merchant_category=category,
            is_fraud=0,
            payment_method=user.usual_payment if r.random() < 0.85 else r.choice(PAYMENT_METHODS),
            country=user.home_country if r.random() < 0.95 else r.choice(COUNTRIES),
            device_type=user.usual_device if r.random() < 0.9 else r.choice(DEVICES),
        )

    def _fraud(self, user: UserProfile, ts: datetime) -> list[Transaction]:
        r = self.rng
        kind = r.choices(FRAUD_TYPES, weights=FRAUD_TYPE_WEIGHTS)[0]
        foreign = r.choice([c for c in COUNTRIES if c != user.home_country])
        other_device = r.choice([d for d in DEVICES if d != user.usual_device])

        if kind == "card_testing":
            out = []
            for i in range(r.randint(*CARD_TESTING_BURST)):
                out.append(self._txn(
                    user, ts + timedelta(seconds=20 * i + r.randint(0, 15)),
                    transaction_amount=round(r.uniform(0.5, 5.0), 2),
                    merchant_category="digital_goods",
                    is_fraud=1, payment_method="credit_card",
                    country=foreign, device_type=other_device, fraud_type=kind,
                ))
            return out

        if kind == "account_takeover":
            category = r.choice(HIGH_RISK_CATEGORIES + ["travel"])
            return [self._txn(
                user, ts,
                transaction_amount=self._amount(category, user.spend_multiplier * r.uniform(3, 8)),
                merchant_category=category, is_fraud=1,
                payment_method=r.choice(PAYMENT_METHODS),
                country=foreign, device_type=other_device, fraud_type=kind,
            )]

        if kind == "high_risk_spend":
            category = r.choice(HIGH_RISK_CATEGORIES)
            return [self._txn(
                user, ts,
                transaction_amount=self._amount(category, user.spend_multiplier * r.uniform(4, 10)),
                merchant_category=category, is_fraud=1,
                payment_method=user.usual_payment,
                country=user.home_country if r.random() < 0.6 else foreign,
                device_type=user.usual_device if r.random() < 0.5 else other_device,
                fraud_type=kind,
            )]

        # odd_hours
        category = r.choice(HIGH_RISK_CATEGORIES + ["travel", "fashion"])
        return [self._txn(
            user, self._ts(ts, r.randint(1, 4), user.home_country),
            transaction_amount=self._amount(category, user.spend_multiplier * r.uniform(2, 6)),
            merchant_category=category, is_fraud=1,
            payment_method=r.choice(PAYMENT_METHODS),
            country=user.home_country, device_type=other_device, fraud_type=kind,
        )]

    def next_transaction(self, now: datetime | None = None, local_hour: int | None = None) -> Transaction:
        """Return one transaction.

        `now` defaults to the current UTC time (real-time mode). For historical batches,
        `local_hour` places the transaction at that hour in the user's home timezone.
        """
        if self._pending:
            return self._pending.pop(0)
        now = now or datetime.now(UTC)
        user = self.rng.choice(self.users)
        if local_hour is not None:
            now = self._ts(now, local_hour, user.home_country)
        if self.rng.random() < self._attack_prob:
            txns = self._fraud(user, now)
            self._pending.extend(txns[1:])
            return txns[0]
        return self._legit(user, now)

    def generate_batch(self, n: int, start: datetime | None = None, days: int = 30) -> list[Transaction]:
        """Generate `n` historical transactions spread across `days`, sorted by time (for training data)."""
        start = start or (datetime.now(UTC) - timedelta(days=days))
        out: list[Transaction] = []
        while len(out) < n:
            day = start + timedelta(days=self.rng.randrange(days))
            out.append(self.next_transaction(day, local_hour=self._normal_hour()))
        out = out[:n]
        out.sort(key=lambda t: t.timestamp)
        return out

    def stream(self) -> Iterator[Transaction]:
        while True:
            yield self.next_transaction()


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic e-commerce transactions")
    p.add_argument("--n", type=int, default=10_000, help="number of transactions")
    p.add_argument("--users", type=int, default=1000)
    p.add_argument("--fraud-rate", type=float, default=0.02)
    p.add_argument("--days", type=int, default=30, help="history window to spread transactions over")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/raw/transactions.csv")
    p.add_argument("--to-postgres", action="store_true",
                   help="insert directly into the Postgres transactions table instead of writing a CSV")
    args = p.parse_args()

    sim = TransactionSimulator(n_users=args.users, fraud_rate=args.fraud_rate, seed=args.seed)
    txns = sim.generate_batch(args.n, days=args.days)
    fraud_rate = sum(t.is_fraud for t in txns) / len(txns)

    if args.to_postgres:
        from src import config
        from src.streaming.consumer import connect_postgres, ensure_schema, to_row, write_batch

        conn = connect_postgres(config.postgres_dsn())
        ensure_schema(conn)
        inserted = 0
        for i in range(0, len(txns), 5000):
            inserted += write_batch(conn, [to_row(t.to_dict()) for t in txns[i:i + 5000]])
        conn.close()
        print(f"Inserted {inserted:,} transactions into Postgres (fraud rate {fraud_rate:.2%})")
        return

    import pandas as pd

    pd.DataFrame([t.to_dict() for t in txns]).to_csv(args.out, index=False)
    print(f"Wrote {len(txns):,} transactions to {args.out} (fraud rate {fraud_rate:.2%})")


if __name__ == "__main__":
    main()
