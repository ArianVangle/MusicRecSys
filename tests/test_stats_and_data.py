"""
Тесты статистических проверок и подготовки данных.
"""

import numpy as np
import pandas as pd
import pytest

from loaders import _apply_kcore, _encode_ids, has_timestamps
from stats_tests import bootstrap_ci, paired_test



def test_bootstrap_returns_correct_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, lo, hi = bootstrap_ci(values, n_boot=500, seed=0)
    assert mean == pytest.approx(3.0)


def test_bootstrap_interval_contains_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.5, scale=0.1, size=200)
    mean, lo, hi = bootstrap_ci(values, n_boot=1000, seed=0)
    assert lo < mean < hi


def test_bootstrap_interval_narrows_with_more_data():
    rng = np.random.default_rng(1)
    small = rng.normal(0.5, 0.1, size=30)
    large = rng.normal(0.5, 0.1, size=3000)

    _, lo_s, hi_s = bootstrap_ci(small, n_boot=1000, seed=0)
    _, lo_l, hi_l = bootstrap_ci(large, n_boot=1000, seed=0)

    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bootstrap_zero_variance():
    values = np.full(50, 0.7)
    mean, lo, hi = bootstrap_ci(values, n_boot=200, seed=0)
    assert mean == pytest.approx(0.7)
    assert lo == pytest.approx(0.7)
    assert hi == pytest.approx(0.7)


def test_bootstrap_empty_input():
    mean, lo, hi = bootstrap_ci(np.array([]))
    assert (mean, lo, hi) == (0.0, 0.0, 0.0)



def test_paired_detects_clear_difference():
    rng = np.random.default_rng(0)
    b = rng.uniform(0.1, 0.3, size=200)
    a = b + 0.1
    res = paired_test(a, b, "A", "B")

    assert res["significant"] is True
    assert res["mean_diff"] == pytest.approx(0.1, abs=1e-6)
    assert res["wins_a"] == 200
    assert res["wins_b"] == 0


def test_paired_finds_no_difference_in_noise():
    rng = np.random.default_rng(42)
    a = rng.normal(0.5, 0.1, size=300)
    b = rng.normal(0.5, 0.1, size=300)

    res = paired_test(a, b, "A", "B")
    assert res["significant"] is False


def test_paired_identical_inputs():
    values = np.array([0.1, 0.5, 0.3, 0.9])
    res = paired_test(values, values, "SVD", "Hybrid")

    assert res["mean_diff"] == 0.0
    assert res["p_value"] == 1.0
    assert res["significant"] is False
    assert res["ties"] == 4
    assert res["wins_a"] == 0 and res["wins_b"] == 0


def test_paired_counts_wins_correctly():
    a = np.array([1.0, 0.0, 0.5, 2.0])
    b = np.array([0.0, 1.0, 0.5, 1.0])
    res = paired_test(a, b, "A", "B")

    assert res["wins_a"] == 2
    assert res["wins_b"] == 1
    assert res["ties"] == 1


def test_paired_direction_is_a_minus_b():
    a = np.array([0.5, 0.6, 0.7])
    b = np.array([0.1, 0.2, 0.3])
    res = paired_test(a, b, "A", "B")
    assert res["mean_diff"] > 0


def test_paired_rejects_mismatched_lengths():
    with pytest.raises(AssertionError):
        paired_test(np.array([1.0, 2.0]), np.array([1.0]))



def test_rank_biserial_is_plus_one_when_a_always_wins():
    a = np.array([0.5, 0.6, 0.7])
    b = np.array([0.1, 0.2, 0.3])
    res = paired_test(a, b, "A", "B")
    assert res["rank_biserial"] == pytest.approx(1.0)


def test_rank_biserial_is_minus_one_when_b_always_wins():
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.5, 0.6, 0.7])
    res = paired_test(a, b, "A", "B")
    assert res["rank_biserial"] == pytest.approx(-1.0)


def test_rank_biserial_near_zero_for_symmetric_diffs():
    a = np.array([1.0, 0.0, 1.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 1.0])
    res = paired_test(a, b, "A", "B")
    assert res["rank_biserial"] == pytest.approx(0.0)


def test_rank_biserial_zero_when_identical():
    values = np.array([0.3, 0.4, 0.5])
    res = paired_test(values, values, "A", "B")
    assert res["rank_biserial"] == 0.0


def test_effect_sizes_share_sign_with_mean_diff():
    rng = np.random.default_rng(3)
    b = rng.uniform(0.1, 0.4, size=100)
    a = b + rng.uniform(0.0, 0.2, size=100)

    res = paired_test(a, b, "A", "B")
    assert res["mean_diff"] > 0
    assert res["cohens_dz"] > 0
    assert res["rank_biserial"] > 0

    flipped = paired_test(b, a, "B", "A")
    assert flipped["mean_diff"] < 0
    assert flipped["cohens_dz"] < 0
    assert flipped["rank_biserial"] < 0


def test_format_comparison_names_better_model():
    from stats_tests import format_comparison

    a = np.array([0.9] * 50)
    b = np.array([0.1] * 50)
    text = format_comparison(paired_test(a, b, "ItemCF", "SVD"), "NDCG@10")

    assert "лучше ItemCF" in text
    assert "rank-biserial" in text


def test_format_comparison_handles_tiny_p_value():
    from stats_tests import format_comparison

    rng = np.random.default_rng(0)
    b = rng.uniform(0.0, 0.1, size=500)
    a = b + 0.5
    text = format_comparison(paired_test(a, b, "A", "B"), "NDCG@10")

    assert "< 1e-16" in text or "e-" in text
    assert "p-value = 0.00e+00" not in text



def test_kcore_removes_rare_items():
    df = pd.DataFrame({
        "u": [0, 1, 2, 0],
        "i": ["popular", "popular", "popular", "rare"],
    })
    result = _apply_kcore(df, "u", "i", min_user=1, min_item=2)
    assert "rare" not in set(result.i)
    assert len(result) == 3


def test_kcore_removes_rare_users():
    df = pd.DataFrame({
        "u": [0, 0, 0, 1],
        "i": ["a", "b", "c", "a"],
    })
    result = _apply_kcore(df, "u", "i", min_user=2, min_item=1)
    assert 1 not in set(result.u)


def test_kcore_is_iterative():
    df = pd.DataFrame({
        "u": [0, 0, 1, 1, 2],
        "i": ["a", "b", "a", "b", "rare"],
    })
    result = _apply_kcore(df, "u", "i", min_user=2, min_item=2)

    assert "rare" not in set(result.i)
    assert 2 not in set(result.u), "пользователь должен исчезнуть следом за объектом"
    assert len(result) == 4


def test_kcore_keeps_everything_when_thresholds_low():
    df = pd.DataFrame({"u": [0, 1], "i": ["a", "b"]})
    result = _apply_kcore(df, "u", "i", min_user=1, min_item=1)
    assert len(result) == 2


def test_encode_ids_produces_contiguous_range():
    df = pd.DataFrame({
        "user": ["sha_aaa", "sha_bbb", "sha_aaa"],
        "item": ["Metallica", "Slayer", "Slayer"],
        "weight": [10.0, 20.0, 30.0],
    })
    out = _encode_ids(df, "user", "item")

    assert sorted(out.user_id.unique()) == [0, 1]
    assert sorted(out.item_id.unique()) == [0, 1]


def test_encode_ids_is_consistent():
    df = pd.DataFrame({
        "user": ["x", "y", "x"],
        "item": ["a", "b", "b"],
        "weight": [1.0, 1.0, 1.0],
    })
    out = _encode_ids(df, "user", "item")
    assert out.user_id.iloc[0] == out.user_id.iloc[2]


def test_encode_ids_preserves_weights():
    df = pd.DataFrame({
        "user": ["x", "y"], "item": ["a", "b"], "weight": [7.0, 42.0],
    })
    out = _encode_ids(df, "user", "item")
    assert list(out.weight) == [7.0, 42.0]



def test_has_timestamps_true_when_present():
    df = pd.DataFrame({"user_id": [0], "item_id": [0], "timestamp": [12345]})
    assert has_timestamps(df) is True


def test_has_timestamps_false_for_lastfm_shape():
    df = pd.DataFrame({"user_id": [0], "item_id": [0], "weight": [1.0]})
    assert has_timestamps(df) is False


def test_has_timestamps_false_when_all_null():
    df = pd.DataFrame({"user_id": [0, 1], "timestamp": [None, None]})
    assert has_timestamps(df) is False
