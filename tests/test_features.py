import numpy as np
import pandas as pd
import pytest

from src.data_simulator import TransactionSimulator
from src.features.build_features import NON_FEATURES, build_features, feature_columns
from tests.conftest import DATA_START


@pytest.fixture(scope="module")
def raw():
    sim = TransactionSimulator(n_users=100, seed=11)
    return pd.DataFrame([t.to_dict() for t in sim.generate_batch(6000, start=DATA_START, days=20)])


@pytest.fixture(scope="module")
def feats(raw):
    return build_features(raw)


def test_shape_and_no_label_leak(raw, feats):
    assert len(feats) == len(raw)
    cols = feature_columns(feats)
    assert not set(cols) & NON_FEATURES
    assert not any("fraud" in c for c in cols)
    assert not feats[cols].isna().any().any()
    assert (feats[cols].dtypes == "float64").all()


def test_point_in_time_correctness(raw, feats):
    """Recompute features by brute force using only strictly-earlier rows."""
    r = raw.assign(timestamp=pd.to_datetime(raw.timestamp, utc=True)).set_index("transaction_id")
    f = feats.set_index("transaction_id")
    for tid in np.random.default_rng(0).choice(f.index, 200, replace=False):
        row, me = f.loc[tid], r.loc[tid]
        hist = r[(r.user_id == me.user_id) & (r.timestamp < me.timestamp)]
        assert row.user_prior_txn_count == len(hist)
        assert row.txn_count_1h == (hist.timestamp >= me.timestamp - pd.Timedelta("1h")).sum()
        assert row.txn_count_24h == (hist.timestamp >= me.timestamp - pd.Timedelta("24h")).sum()
        if len(hist):
            assert row.user_avg_amount == pytest.approx(hist.transaction_amount.mean())
            assert row.is_new_device == int(me.device_type not in set(hist.device_type))
        else:
            assert row.is_first_txn == 1 and row.is_new_device == 0


def test_future_rows_do_not_change_past_features(raw):
    """Appending later transactions must not alter features of earlier ones."""
    raw = raw.sort_values("timestamp")
    early = raw.iloc[: len(raw) // 2]
    a = build_features(early).set_index("transaction_id")
    b = build_features(raw).set_index("transaction_id").loc[a.index]
    cols = feature_columns(a)
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_fraud_signal_present(feats):
    by = feats.groupby("is_fraud")
    assert by.amount_to_user_avg.mean()[1] > 2 * by.amount_to_user_avg.mean()[0]
    assert by.txn_count_10m.mean()[1] > by.txn_count_10m.mean()[0]


def test_online_features_match_offline(raw, feats):
    """Serving path (one txn + its history) must reproduce training features exactly."""
    from src.features.build_features import build_online_features

    r = raw.assign(ts=pd.to_datetime(raw.timestamp, utc=True))
    off = feats.set_index("transaction_id")
    cols = feature_columns(feats)
    for tid in np.random.default_rng(1).choice(raw.transaction_id, 150, replace=False):
        me = r[r.transaction_id == tid].iloc[0]
        hist = r[(r.user_id == me.user_id) & (r.ts < me.ts)]
        # point-in-time category mean, as training computes it
        prior_cat = r[(r.merchant_category == me.merchant_category)
                      & ((r.ts < me.ts) | ((r.ts == me.ts) & (r.transaction_id < tid)))]
        cm = {me.merchant_category: prior_cat.transaction_amount.mean()} if len(prior_cat) else {}
        on = build_online_features(hist.drop(columns="ts"), me.drop("ts").to_dict(), cm)
        np.testing.assert_allclose(on[cols].to_numpy()[0], off.loc[tid, cols].to_numpy(dtype=float),
                                   rtol=1e-9, err_msg=tid)


def test_hour_features_use_local_time():
    from src.features.build_features import build_features as bf
    base = dict(user_id="u", transaction_amount=10.0, merchant_category="grocery",
                payment_method="e_wallet", device_type="ios", is_fraud=0)
    df = pd.DataFrame([
        {**base, "transaction_id": "a", "country": "SG", "timestamp": "2026-10-01T19:00:00+00:00"},  # 03:00 SGT
        {**base, "transaction_id": "b", "country": "GB", "timestamp": "2026-10-01T19:00:00+00:00"},  # 20:00 BST
    ])
    f = bf(df).set_index("transaction_id")
    assert f.loc["a", "hour"] == 3 and f.loc["a", "is_night"] == 1
    assert f.loc["b", "hour"] == 20 and f.loc["b", "is_night"] == 0
