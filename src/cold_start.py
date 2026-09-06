from __future__ import annotations

import numpy as np
import pandas as pd

from evaluate import evaluate, leave_n_out_split
from recommender import (
    ColdStartHybrid, MatrixFactorizationSVD, PopularityBaseline, build_user_item_matrix,
)


def simulate_cold_users(train_df, cold_frac=0.15, keep_interactions=1, seed=7):
    """Урезает историю части пользователей до keep_interactions записей."""
    rng = np.random.default_rng(seed)
    users = train_df.user_id.unique()
    n_cold = int(len(users) * cold_frac)
    cold_users = set(rng.choice(users, size=n_cold, replace=False))

    parts = []
    for user_id, group in train_df.groupby("user_id", sort=False):
        if user_id in cold_users and len(group) > keep_interactions:
            parts.append(group.sample(n=keep_interactions, random_state=seed))
        else:
            parts.append(group)

    return pd.concat(parts).reset_index(drop=True), cold_users


def run_cold_start_experiment(df, k=10):
    n_users = df.user_id.max() + 1
    n_items = df.item_id.max() + 1

    train_df, test_df = leave_n_out_split(df)
    train_cold, cold_users = simulate_cold_users(train_df)

    test_cold = test_df[test_df.user_id.isin(cold_users)].reset_index(drop=True)

    print(f"Пользователей-новичков: {len(cold_users)} "
          f"(история урезана до 1 взаимодействия)")
    print(f"Тестовых взаимодействий у них: {len(test_cold):,}\n")

    matrix = build_user_item_matrix(train_cold, n_users, n_items)

    models = {
        "Popularity": PopularityBaseline(),
        "SVD-MF (без fallback)": MatrixFactorizationSVD(n_factors=64),
        "Hybrid (SVD + fallback)": ColdStartHybrid(
            MatrixFactorizationSVD(n_factors=64), PopularityBaseline(), min_interactions=3
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(matrix)
        res = evaluate(model, train_cold, test_cold, n_users, n_items, k=k, name=name)
        res.pop("per_user", None)
        rows.append(res)

    return pd.DataFrame(rows).set_index("model")


if __name__ == "__main__":
    import sys
    from loaders import load_lastfm_360k, load_movielens_100k

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="lastfm", choices=["lastfm", "movielens"])
    a = ap.parse_args()
    df = load_lastfm_360k() if a.dataset == "lastfm" else load_movielens_100k()

    print("=" * 78)
    print("ЭКСПЕРИМЕНТ: ХОЛОДНЫЙ СТАРТ")
    print("=" * 78)
    results = run_cold_start_experiment(df)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(results.round(4))
