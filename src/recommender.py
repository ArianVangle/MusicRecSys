"""
Модели рекомендаций.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


def build_user_item_matrix(df, n_users=None, n_items=None) -> csr_matrix:
    """
    Строит разреженную матрицу user-item из таблицы взаимодействий.
    """
    n_users = n_users or df.user_id.max() + 1
    n_items = n_items or df.item_id.max() + 1
    return csr_matrix(
        (np.log1p(df.weight.values), (df.user_id.values, df.item_id.values)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )


class PopularityBaseline:
    """
    рекомендуем самое популярное.
    """

    name = "Popularity"

    def fit(self, user_item: csr_matrix):
        self.item_popularity = np.asarray((user_item > 0).sum(axis=0)).ravel()
        self.user_item = user_item.tocsr()
        return self

    def recommend(self, user_id: int, n=10, exclude_seen=True):
        scores = self.item_popularity.astype(np.float32).copy()
        if exclude_seen and user_id < self.user_item.shape[0]:
            scores[self.user_item[user_id].indices] = -np.inf
        return _top_n(scores, n)


class ItemBasedCF:
    """
    Item-based collaborative filtering на косинусном сходстве.
    """

    name = "ItemCF"

    def __init__(self, top_k: int = 200, shrink: float = 0.0):
        self.top_k = top_k
        self.shrink = shrink

    def fit(self, user_item: csr_matrix):
        self.user_item = user_item.tocsr()

        item_user = normalize(user_item.T.tocsr(), norm="l2", axis=1)
        sim = (item_user @ item_user.T).tolil()
        sim.setdiag(0)
        sim = sim.tocsr()
        sim.eliminate_zeros()

        self.similarity = _keep_top_k_per_row(sim, self.top_k)
        return self

    def recommend(self, user_id: int, n=10, exclude_seen=True):
        if user_id >= self.user_item.shape[0]:
            return []
        user_vector = self.user_item[user_id]
        if user_vector.nnz == 0:
            return []

        scores = np.asarray((user_vector @ self.similarity).todense()).ravel()
        if exclude_seen:
            scores[user_vector.indices] = -np.inf
        return _top_n(scores, n)


class MatrixFactorizationSVD:
    """
    Matrix factorization через усеченное SVD.
    """

    name = "SVD-MF"

    def __init__(self, n_factors: int = 64, seed: int = 42):
        self.n_factors = n_factors
        self.seed = seed

    def fit(self, user_item: csr_matrix):
        self.user_item = user_item.tocsr()

        max_k = min(user_item.shape) - 1
        if max_k < 1:
            raise ValueError(
                f"Матрица {user_item.shape} слишком мала для SVD: "
                "нужно минимум 2 пользователя и 2 объекта."
            )
        k = min(self.n_factors, max_k)

        U, sigma, Vt = svds(user_item.astype(np.float32), k=k, random_state=self.seed)
        idx = np.argsort(-sigma)
        U, sigma, Vt = U[:, idx], sigma[idx], Vt[idx]

        sqrt_sigma = np.sqrt(sigma)
        self.user_factors = (U * sqrt_sigma).astype(np.float32)
        self.item_factors = (Vt.T * sqrt_sigma).astype(np.float32)
        return self

    def recommend(self, user_id: int, n=10, exclude_seen=True):
        if user_id >= self.user_factors.shape[0]:
            return []
        scores = self.item_factors @ self.user_factors[user_id]
        if exclude_seen:
            scores[self.user_item[user_id].indices] = -np.inf
        return _top_n(scores, n)


class ColdStartHybrid:
    """
    Обертка, решающая холодный старт.
    """

    name = "Hybrid"

    def __init__(self, personalized, fallback, min_interactions: int = 3):
        self.personalized = personalized
        self.fallback = fallback
        self.min_interactions = min_interactions

    def fit(self, user_item: csr_matrix):
        self.user_item = user_item.tocsr()
        self.personalized.fit(user_item)
        self.fallback.fit(user_item)
        return self

    def is_cold(self, user_id: int) -> bool:
        if user_id >= self.user_item.shape[0]:
            return True
        return self.user_item[user_id].nnz < self.min_interactions

    def recommend(self, user_id: int, n=10, exclude_seen=True):
        model = self.fallback if self.is_cold(user_id) else self.personalized
        return model.recommend(user_id, n=n, exclude_seen=exclude_seen)


def _top_n(scores: np.ndarray, n: int):
    n = min(n, len(scores))
    finite = np.isfinite(scores)
    if not finite.any():
        return []

    idx = np.argpartition(-scores, range(min(n, finite.sum())))[:n]
    idx = idx[np.isfinite(scores[idx]) & (scores[idx] > 0)]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


def _keep_top_k_per_row(sim: csr_matrix, k: int) -> csr_matrix:
    sim = sim.tocsr()
    data, indices, indptr = [], [], [0]

    for row in range(sim.shape[0]):
        start, end = sim.indptr[row], sim.indptr[row + 1]
        row_data = sim.data[start:end]
        row_idx = sim.indices[start:end]

        if len(row_data) > k:
            top = np.argpartition(-row_data, k)[:k]
            row_data, row_idx = row_data[top], row_idx[top]

        data.append(row_data)
        indices.append(row_idx)
        indptr.append(indptr[-1] + len(row_data))

    return csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.array(indptr)),
        shape=sim.shape,
    )
