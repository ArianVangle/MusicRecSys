"""
Развертка по гиперпараметрам: как n_factors и top_k влияют на точность
и на покрытие каталога.

Запуск:
    python src/sweep.py --param n_factors --values 32 64 128 256 512
    python src/sweep.py --param top_k --values 50 200 500 1000
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from evaluate import evaluate, make_split
from recommender import (
    ItemBasedCF, MatrixFactorizationSVD, build_user_item_matrix,
)


def sweep(df, param="n_factors", values=None, k=10, mode="random", seed=0, verbose=True):
    if values is None:
        values = [8, 16, 32, 64, 128, 256] if param == "n_factors" else [25, 50, 100, 200, 500]

    n_users = int(df.user_id.max() + 1)
    n_items = int(df.item_id.max() + 1)

    train_df, test_df = make_split(df, mode=mode, seed=seed)
    matrix = build_user_item_matrix(train_df, n_users, n_items)

    if verbose:
        print(f"каталог: {n_items:,} объектов | пользователей: {n_users:,}")
        print(f"перебираем {param}: {values}\n")

    rows = []
    for value in values:
        if param == "n_factors":
            model = MatrixFactorizationSVD(n_factors=value)
            label = f"SVD n_factors={value}"
        elif param == "top_k":
            model = ItemBasedCF(top_k=value)
            label = f"ItemCF top_k={value}"
        else:
            raise ValueError("param должен быть n_factors или top_k")

        if verbose:
            print(f"  {label}...", flush=True)

        model.fit(matrix)
        res = evaluate(model, train_df, test_df, n_users, n_items, k=k, name=label)
        res.pop("per_user", None)
        res[param] = value
        rows.append(res)

    return pd.DataFrame(rows).set_index("model")


if __name__ == "__main__":
    from loaders import load_lastfm_360k, load_movielens_100k

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="lastfm",
                        choices=["lastfm", "movielens"])
    parser.add_argument("--param", default="n_factors", choices=["n_factors", "top_k"])
    parser.add_argument("--values", type=int, nargs="+", default=None)
    parser.add_argument("--mode", default="random", choices=["random", "temporal"])
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    df = load_lastfm_360k() if args.dataset == "lastfm" else load_movielens_100k()
    title = "Last.fm-360K" if args.dataset == "lastfm" else "MovieLens-100k"

    result = sweep(df, param=args.param, values=args.values, k=args.k, mode=args.mode)

    print(f"\n{'=' * 78}")
    print(f"РАЗВЕРТКА ПО {args.param.upper()} — {title}, режим {args.mode}")
    print("=" * 78)

    cols = [args.param, f"NDCG@{args.k}", f"Recall@{args.k}", "Coverage", "AvgPopularity"]
    with pd.option_context("display.width", 200):
        print(result[cols].round(4).to_string())

