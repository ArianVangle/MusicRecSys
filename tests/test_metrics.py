"""
Тесты метрик, сплитов и построения матрицы.
"""

import numpy as np
import pandas as pd
import pytest

from evaluate import (
    dcg, evaluate, leave_n_out_split, temporal_split,
)
from recommender import build_user_item_matrix


class StubModel:
    """Модель-заглушка, всегда возвращает заранее заданный список.
    """
    name = "Stub"

    def __init__(self, recs):
        self.recs = recs

    def recommend(self, user_id, n=10, exclude_seen=True):
        return [(item, 1.0 / (i + 1)) for i, item in enumerate(self.recs[:n])]



def test_dcg_single_hit_at_first_position():
    assert dcg([1]) == pytest.approx(1.0)


def test_dcg_position_weights():
    assert dcg([1, 0, 0]) == pytest.approx(1.0)
    assert dcg([0, 1, 0]) == pytest.approx(0.63093, abs=1e-5)
    assert dcg([0, 0, 1]) == pytest.approx(0.5)


def test_dcg_is_position_sensitive():
    """NDCG. попадание наверху должно давать больше, чем внизу."""
    assert dcg([1, 1, 0, 0]) > dcg([0, 0, 1, 1])


def test_dcg_empty_and_zero():
    assert dcg([]) == 0.0
    assert dcg([0, 0, 0]) == 0.0



def _make_eval_data():
    train = pd.DataFrame({
        "user_id": [0, 0, 0],
        "item_id": [200, 201, 202],
        "weight": [1.0, 1.0, 1.0],
    })
    test = pd.DataFrame({
        "user_id": [0, 0],
        "item_id": [101, 104],
        "weight": [1.0, 1.0],
    })
    return train, test


def test_precision_recall_hitrate_by_hand():
    train, test = _make_eval_data()
    model = StubModel([100, 101, 102, 103, 104])

    res = evaluate(model, train, test, n_users=1, n_items=300, k=5)

    assert res["Precision@5"] == pytest.approx(2 / 5)
    assert res["Recall@5"] == pytest.approx(1.0)
    assert res["HitRate@5"] == pytest.approx(1.0)


def test_ndcg_by_hand():
    train, test = _make_eval_data()
    model = StubModel([100, 101, 102, 103, 104])

    res = evaluate(model, train, test, n_users=1, n_items=300, k=5)
    assert res["NDCG@5"] == pytest.approx(0.62405, abs=1e-4)


def test_ndcg_is_one_when_all_hits_on_top():
    train = pd.DataFrame({"user_id": [0], "item_id": [200], "weight": [1.0]})
    test = pd.DataFrame({"user_id": [0, 0], "item_id": [10, 11], "weight": [1.0, 1.0]})
    model = StubModel([10, 11, 12, 13, 14])

    res = evaluate(model, train, test, n_users=1, n_items=300, k=5)
    assert res["NDCG@5"] == pytest.approx(1.0)


def test_metrics_are_zero_when_no_hits():
    train = pd.DataFrame({"user_id": [0], "item_id": [200], "weight": [1.0]})
    test = pd.DataFrame({"user_id": [0], "item_id": [999], "weight": [1.0]})
    model = StubModel([10, 11, 12])

    res = evaluate(model, train, test, n_users=1, n_items=1000, k=3)
    assert res["Precision@3"] == 0.0
    assert res["Recall@3"] == 0.0
    assert res["NDCG@3"] == 0.0
    assert res["HitRate@3"] == 0.0


def test_recall_when_relevant_exceeds_k():
    train = pd.DataFrame({"user_id": [0], "item_id": [200], "weight": [1.0]})
    test = pd.DataFrame({
        "user_id": [0, 0, 0, 0],
        "item_id": [10, 11, 12, 13],
        "weight": [1.0] * 4,
    })
    model = StubModel([10, 11, 99, 98])

    res = evaluate(model, train, test, n_users=1, n_items=300, k=2)
    assert res["Recall@2"] == pytest.approx(2 / 4)
    assert res["Precision@2"] == pytest.approx(1.0)
    assert res["NDCG@2"] == pytest.approx(1.0)


def test_per_user_arrays_are_aligned():
    train = pd.DataFrame({
        "user_id": [0, 1, 2], "item_id": [200, 201, 202], "weight": [1.0] * 3,
    })
    test = pd.DataFrame({
        "user_id": [2, 0, 1], "item_id": [10, 11, 12], "weight": [1.0] * 3,
    })
    model = StubModel([10, 11, 12])

    res = evaluate(model, train, test, n_users=3, n_items=300, k=3)
    pu = res["per_user"]

    assert list(pu["user_ids"]) == [0, 1, 2], "user_ids должны быть отсортированы"
    assert len(pu["ndcg"]) == len(pu["user_ids"])
    assert len(pu["precision"]) == len(pu["user_ids"])


def test_users_not_in_train_are_excluded():
    train = pd.DataFrame({"user_id": [0], "item_id": [200], "weight": [1.0]})
    test = pd.DataFrame({"user_id": [0, 5], "item_id": [10, 11], "weight": [1.0, 1.0]})
    model = StubModel([10, 11, 12])

    res = evaluate(model, train, test, n_users=10, n_items=300, k=3)
    assert res["n_users_eval"] == 1
    assert list(res["per_user"]["user_ids"]) == [0]


def test_coverage_counts_distinct_items():
    train = pd.DataFrame({
        "user_id": [0, 1], "item_id": [200, 201], "weight": [1.0, 1.0],
    })
    test = pd.DataFrame({
        "user_id": [0, 1], "item_id": [10, 10], "weight": [1.0, 1.0],
    })
    model = StubModel([10, 11])

    res = evaluate(model, train, test, n_users=2, n_items=100, k=2)
    assert res["Coverage"] == pytest.approx(2 / 100)



def test_matrix_applies_log1p():
    df = pd.DataFrame({"user_id": [0], "item_id": [0], "weight": [99.0]})
    m = build_user_item_matrix(df, n_users=1, n_items=1)
    assert m[0, 0] == pytest.approx(np.log1p(99.0), abs=1e-5)


def test_matrix_shape_and_sparsity():
    df = pd.DataFrame({
        "user_id": [0, 2], "item_id": [1, 3], "weight": [1.0, 5.0],
    })
    m = build_user_item_matrix(df, n_users=4, n_items=5)
    assert m.shape == (4, 5)
    assert m.nnz == 2, "хранятся только ненулевые элементы"
    assert m[1, 1] == 0.0


def test_log1p_compresses_heavy_tail():
    df = pd.DataFrame({
        "user_id": [0, 1], "item_id": [0, 0], "weight": [20.0, 5000.0],
    })
    m = build_user_item_matrix(df, n_users=2, n_items=1)
    ratio = m[1, 0] / m[0, 0]
    assert ratio < 3.0, f"после log1p соотношение {ratio:.2f}, ожидалось < 3"


def _split_df(n_users=10, n_items_per_user=10):
    rows = []
    for u in range(n_users):
        for i in range(n_items_per_user):
            rows.append({"user_id": u, "item_id": i, "weight": 1.0,
                         "timestamp": i})
    return pd.DataFrame(rows)


def test_split_has_no_overlap():
    df = _split_df()
    train, test = leave_n_out_split(df, test_frac=0.2, seed=0)

    train_pairs = set(zip(train.user_id, train.item_id))
    test_pairs = set(zip(test.user_id, test.item_id))
    assert train_pairs & test_pairs == set(), "train и test пересекаются"


def test_split_preserves_all_interactions():
    df = _split_df()
    train, test = leave_n_out_split(df, test_frac=0.2, seed=0)
    assert len(train) + len(test) == len(df), "часть данных потерялась"


def test_split_every_user_present_in_train():
    df = _split_df()
    train, test = leave_n_out_split(df, test_frac=0.2, seed=0)
    assert set(train.user_id) == set(df.user_id)


def test_split_is_reproducible():
    df = _split_df()
    a1, b1 = leave_n_out_split(df, seed=42)
    a2, b2 = leave_n_out_split(df, seed=42)
    assert set(zip(b1.user_id, b1.item_id)) == set(zip(b2.user_id, b2.item_id))


def test_split_differs_with_different_seed():
    df = _split_df(n_users=30, n_items_per_user=20)
    _, b1 = leave_n_out_split(df, seed=1)
    _, b2 = leave_n_out_split(df, seed=2)
    assert set(zip(b1.user_id, b1.item_id)) != set(zip(b2.user_id, b2.item_id))


def test_short_history_user_goes_entirely_to_train():
    df = pd.DataFrame({
        "user_id": [0, 0], "item_id": [1, 2], "weight": [1.0, 1.0],
    })
    train, test = leave_n_out_split(df, test_frac=0.2, min_train=3)
    assert len(train) == 2
    assert len(test) == 0


def test_temporal_split_puts_latest_in_test():
    df = _split_df(n_users=1, n_items_per_user=10)
    train, test = temporal_split(df, test_frac=0.2)

    assert train.timestamp.max() < test.timestamp.min(), \
        "в train попало событие позже, чем в test — утечка из будущего"
    assert len(test) == 2


def test_temporal_split_requires_timestamp():
    df = pd.DataFrame({"user_id": [0] * 5, "item_id": range(5), "weight": [1.0] * 5})
    with pytest.raises(ValueError, match="timestamp"):
        temporal_split(df)
