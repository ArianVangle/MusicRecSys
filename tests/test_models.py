"""
Тесты моделей рекомендаций.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from recommender import (
    ColdStartHybrid, ItemBasedCF, MatrixFactorizationSVD, PopularityBaseline,
    build_user_item_matrix, _keep_top_k_per_row, _top_n,
)


@pytest.fixture
def genre_matrix():
    df = pd.DataFrame({
        "user_id": [0, 0, 1, 1, 2, 2, 3, 3],
        "item_id": [0, 1, 0, 1, 2, 3, 2, 3],
        "weight":  [50., 30., 40., 25., 60., 45., 50., 40.],
    })
    return build_user_item_matrix(df, n_users=4, n_items=5)



def test_top_n_orders_by_score_descending():
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    result = _top_n(scores, n=3)
    assert [i for i, _ in result] == [1, 2, 3]
    assert [round(s, 2) for _, s in result] == [0.9, 0.5, 0.3]


def test_top_n_skips_non_positive_scores():
    scores = np.array([0.5, 0.0, -1.0, 0.2])
    result = _top_n(scores, n=10)
    assert [i for i, _ in result] == [0, 3]


def test_top_n_handles_all_excluded():
    scores = np.array([-np.inf, -np.inf])
    assert _top_n(scores, n=5) == []


def test_top_n_when_n_exceeds_catalog():
    scores = np.array([0.5, 0.3])
    result = _top_n(scores, n=100)
    assert len(result) == 2


def test_keep_top_k_leaves_only_largest():
    m = csr_matrix(np.array([
        [0.1, 0.9, 0.5],
        [0.7, 0.2, 0.4],
    ]))
    result = _keep_top_k_per_row(m, k=2).toarray()

    assert result[0, 0] == 0.0, "самое маленькое значение должно обнулиться"
    assert result[0, 1] == pytest.approx(0.9)
    assert result[0, 2] == pytest.approx(0.5)
    assert result[1, 1] == 0.0
    assert set(np.flatnonzero(result[1])) == {0, 2}


def test_keep_top_k_noop_when_k_large():
    m = csr_matrix(np.array([[0.1, 0.9, 0.5]]))
    result = _keep_top_k_per_row(m, k=99).toarray()
    np.testing.assert_allclose(result, [[0.1, 0.9, 0.5]])



def test_popularity_counts_users_not_plays():
    df = pd.DataFrame({
        "user_id": [0, 1, 2, 3],
        "item_id": [0, 1, 1, 1],
        "weight":  [1000., 1., 1., 1.],
    })
    matrix = build_user_item_matrix(df, n_users=4, n_items=2)
    model = PopularityBaseline().fit(matrix)

    assert model.item_popularity[1] > model.item_popularity[0], \
        "один фанат не должен создавать популярность"


def test_popularity_excludes_already_seen():
    df = pd.DataFrame({
        "user_id": [0, 1, 1], "item_id": [0, 0, 1], "weight": [1., 1., 1.],
    })
    matrix = build_user_item_matrix(df, n_users=2, n_items=2)
    model = PopularityBaseline().fit(matrix)

    recs = model.recommend(0, n=10)
    assert 0 not in [i for i, _ in recs], "прослушанное не рекомендуем"



def test_itemcf_similarity_reflects_shared_audience(genre_matrix):
    model = ItemBasedCF(top_k=10).fit(genre_matrix)
    sim = model.similarity.toarray()

    assert sim[0, 1] > 0.9, "исполнители одного жанра должны быть похожи"
    assert sim[0, 2] == pytest.approx(0.0, abs=1e-6), \
        "исполнители без общей аудитории не должны быть похожи"


def test_itemcf_diagonal_is_zero(genre_matrix):
    model = ItemBasedCF(top_k=10).fit(genre_matrix)
    diag = model.similarity.diagonal()
    np.testing.assert_allclose(diag, np.zeros_like(diag), atol=1e-9)


def test_itemcf_recommends_same_genre(genre_matrix):
    model = ItemBasedCF(top_k=10).fit(genre_matrix)
    recs = model.recommend(0, n=5)
    recommended = [i for i, _ in recs]

    assert 0 not in recommended and 1 not in recommended, "уже слушал"
    for item in recommended:
        assert item not in (2, 3), f"джаз ({item}) не должен попасть металисту"


def test_itemcf_empty_for_unknown_user(genre_matrix):
    model = ItemBasedCF(top_k=10).fit(genre_matrix)
    assert model.recommend(999, n=5) == []


def test_itemcf_empty_for_user_without_history():
    df = pd.DataFrame({"user_id": [0], "item_id": [0], "weight": [1.0]})
    matrix = build_user_item_matrix(df, n_users=3, n_items=2)
    model = ItemBasedCF(top_k=5).fit(matrix)
    assert model.recommend(2, n=5) == []



def test_svd_factor_shapes(genre_matrix):
    model = MatrixFactorizationSVD(n_factors=2).fit(genre_matrix)
    assert model.user_factors.shape == (4, 2)
    assert model.item_factors.shape == (5, 2)


def test_svd_clips_factors_to_matrix_size(genre_matrix):
    model = MatrixFactorizationSVD(n_factors=100).fit(genre_matrix)
    assert model.user_factors.shape[1] < 100


def test_svd_excludes_seen_items(genre_matrix):
    model = MatrixFactorizationSVD(n_factors=2).fit(genre_matrix)
    recs = model.recommend(0, n=5)
    recommended = [i for i, _ in recs]
    assert 0 not in recommended and 1 not in recommended


def test_svd_reconstructs_known_structure(genre_matrix):
    model = MatrixFactorizationSVD(n_factors=4).fit(genre_matrix)
    approx = model.user_factors @ model.item_factors.T
    original = genre_matrix.toarray()

    mask = original > 0
    np.testing.assert_allclose(approx[mask], original[mask], atol=0.15)


def test_svd_unknown_user(genre_matrix):
    model = MatrixFactorizationSVD(n_factors=2).fit(genre_matrix)
    assert model.recommend(999, n=5) == []



def _hybrid_matrix():
    df = pd.DataFrame({
        "user_id": [0, 0, 0, 1, 2, 2, 2],
        "item_id": [0, 1, 2, 0, 0, 1, 3],
        "weight":  [5., 5., 5., 5., 5., 5., 5.],
    })
    return build_user_item_matrix(df, n_users=3, n_items=4)


def test_hybrid_detects_cold_user():
    matrix = _hybrid_matrix()
    model = ColdStartHybrid(
        MatrixFactorizationSVD(n_factors=2), PopularityBaseline(), min_interactions=3
    ).fit(matrix)

    assert model.is_cold(1) is True, "1 взаимодействие -> холодный"
    assert model.is_cold(0) is False, "3 взаимодействия -> не холодный"


def test_hybrid_unknown_user_is_cold():
    matrix = _hybrid_matrix()
    model = ColdStartHybrid(
        MatrixFactorizationSVD(n_factors=2), PopularityBaseline(), min_interactions=3
    ).fit(matrix)
    assert model.is_cold(999) is True


def test_hybrid_uses_fallback_for_cold_user():
    matrix = _hybrid_matrix()
    personalized = MatrixFactorizationSVD(n_factors=2)
    fallback = PopularityBaseline()
    hybrid = ColdStartHybrid(personalized, fallback, min_interactions=3).fit(matrix)

    cold_recs = hybrid.recommend(1, n=3)
    assert cold_recs == fallback.recommend(1, n=3)

    warm_recs = hybrid.recommend(0, n=3)
    assert warm_recs == personalized.recommend(0, n=3)


def test_hybrid_threshold_boundary():
    df = pd.DataFrame({
        "user_id": [0, 0, 0, 1], "item_id": [0, 1, 2, 0], "weight": [1.] * 4,
    })
    matrix = build_user_item_matrix(df, n_users=2, n_items=3)
    model = ColdStartHybrid(
        MatrixFactorizationSVD(n_factors=1), PopularityBaseline(), min_interactions=3
    ).fit(matrix)
    assert model.is_cold(0) is False
    assert model.is_cold(1) is True


def test_svd_rejects_degenerate_matrix():
    df = pd.DataFrame({"user_id": [0, 0], "item_id": [0, 1], "weight": [1., 1.]})
    matrix = build_user_item_matrix(df, n_users=1, n_items=2)

    with pytest.raises(ValueError, match="слишком мала"):
        MatrixFactorizationSVD(n_factors=1).fit(matrix)
