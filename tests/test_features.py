import numpy as np
import pandas as pd
import pytest

from src.data_simulator import TransactionSimulator
from src.features.build_features import NON_FEATURES, build_features, feature_columns


@pytest.fixture(scope="module")
def raw():
    sim = TransactionSimulator(n_users=100, seed=11)
    return pd.DataFrame([t.to_dict() for t in sim.generate_batch(6000, days=20)])


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
