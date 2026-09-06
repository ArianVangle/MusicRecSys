"""
Оценка рекомендаций.

Метрики точности:
- Precision@K, Recall@K — сколько угадали
- NDCG@K — учитывает порядок: попасть на 1-й позиции ценнее, чем на 10-й
- HitRate@K — доля пользователей хотя бы с одним попаданием

Метрики, определяющие пригодность системы для продукта:
- Catalog Coverage — какую долю каталога модель вообще рекомендует.
- Popularity Bias — насколько выдача смещена в сторону уже известного.

Два режима разбиения:
- random: случайные 20% истории каждого пользователя в тест
- temporal: последние по времени 20% истории в тест

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recommender import build_user_item_matrix


def leave_n_out_split(df: pd.DataFrame, test_frac=0.2, min_train=3, seed=42):
    """
    Случайный сплит. у каждого пользователя прячем долю взаимодействий.
    """
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []

    for _, group in df.groupby("user_id", sort=False):
        n = len(group)
        if n < min_train + 1:
            train_parts.append(group)
            continue

        n_test = max(1, int(round(n * test_frac)))
        n_test = min(n_test, n - min_train)

        perm = rng.permutation(n)
        test_parts.append(group.iloc[perm[:n_test]])
        train_parts.append(group.iloc[perm[n_test:]])

    return _concat_parts(train_parts, df), _concat_parts(test_parts, df)


def _concat_parts(parts, template: pd.DataFrame) -> pd.DataFrame:

    if not parts:
        return template.iloc[0:0].copy().reset_index(drop=True)
    return pd.concat(parts).reset_index(drop=True)


def temporal_split(df: pd.DataFrame, test_frac=0.2, min_train=3):
    """
    в тест уходят последние по времени взаимодействия
    каждого пользователя.
    """
    if "timestamp" not in df.columns:
        raise ValueError(
            "Временной сплит требует колонку timestamp. "
            "В Last.fm-360K времени нет — используйте режим random."
        )

    train_parts, test_parts = [], []

    for _, group in df.groupby("user_id", sort=False):
        n = len(group)
        if n < min_train + 1:
            train_parts.append(group)
            continue

        n_test = max(1, int(round(n * test_frac)))
        n_test = min(n_test, n - min_train)

        ordered = group.sort_values("timestamp")
        train_parts.append(ordered.iloc[:-n_test])
        test_parts.append(ordered.iloc[-n_test:])

    return _concat_parts(train_parts, df), _concat_parts(test_parts, df)


def make_split(df, mode="random", seed=42, test_frac=0.2):
    if mode == "temporal":
        return temporal_split(df, test_frac=test_frac)
    return leave_n_out_split(df, test_frac=test_frac, seed=seed)


def dcg(relevances) -> float:
    """Discounted Cumulative Gain. вклад позиции i делится на log2(i+2)."""
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def evaluate(model, train_df, test_df, n_users, n_items, k=10, name=None):
    """
    Возвращает средние метрики И массивы по пользователям.

    per_user нужен для статистических тестов. Чтобы сравнить две модели попарно, 
    нужны их метрики на одних и тех же пользователях в одном порядке.
    """
    test_by_user = test_df.groupby("user_id")["item_id"].apply(set).to_dict()
    train_users = set(train_df.user_id.unique())

    eval_users = sorted(u for u in test_by_user if u in train_users)

    item_popularity = train_df.groupby("item_id").user_id.nunique()
    total_users_train = train_df.user_id.nunique()

    precisions, recalls, ndcgs, hits = [], [], [], []
    recommended_items = set()
    recommended_popularity = []

    for user_id in eval_users:
        relevant = test_by_user[user_id]
        recs = model.recommend(user_id, n=k)

        if not recs:
            precisions.append(0.0)
            recalls.append(0.0)
            ndcgs.append(0.0)
            hits.append(0.0)
            continue

        rec_items = [item for item, _ in recs]
        recommended_items.update(rec_items)
        recommended_popularity.extend(
            item_popularity.get(i, 0) / total_users_train for i in rec_items
        )

        gains = [1 if item in relevant else 0 for item in rec_items]
        n_hits = sum(gains)

        precisions.append(n_hits / k)
        recalls.append(n_hits / len(relevant))
        hits.append(1.0 if n_hits > 0 else 0.0)

        ideal = [1] * min(len(relevant), k)
        ndcgs.append(dcg(gains) / dcg(ideal) if ideal else 0.0)

    return {
        "model": name or getattr(model, "name", type(model).__name__),
        f"Precision@{k}": float(np.mean(precisions)),
        f"Recall@{k}": float(np.mean(recalls)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"HitRate@{k}": float(np.mean(hits)),
        "Coverage": len(recommended_items) / n_items,
        "AvgPopularity": float(np.mean(recommended_popularity)) if recommended_popularity else 0.0,
        "n_users_eval": len(eval_users),
        "per_user": {
            "user_ids": np.array(eval_users),
            "precision": np.array(precisions),
            "recall": np.array(recalls),
            "ndcg": np.array(ndcgs),
            "hit": np.array(hits),
        },
    }


def build_models():
    """Единый список моделей"""
    from recommender import (
        PopularityBaseline, ItemBasedCF, MatrixFactorizationSVD, ColdStartHybrid,
    )
    return [
        PopularityBaseline(),
        ItemBasedCF(top_k=200),
        MatrixFactorizationSVD(n_factors=64),
        ColdStartHybrid(MatrixFactorizationSVD(n_factors=64), PopularityBaseline()),
    ]


def run_benchmark(df, k=10, mode="random", seed=42, verbose=True):
    n_users = int(df.user_id.max() + 1)
    n_items = int(df.item_id.max() + 1)

    train_df, test_df = make_split(df, mode=mode, seed=seed)
    train_matrix = build_user_item_matrix(train_df, n_users, n_items)

    if verbose:
        print(f"[{mode}] train: {len(train_df):,} | test: {len(test_df):,}")

    results = []
    for model in build_models():
        if verbose:
            print(f"  {model.name}...", flush=True)
        model.fit(train_matrix)
        results.append(evaluate(model, train_df, test_df, n_users, n_items, k=k))

    return results


def run_repeated(df, k=10, mode="random", seeds=(0, 1, 2, 3, 4), verbose=True):
    """
    Повторяет весь бенчмарк на нескольких разбиениях.
    """
    if mode == "temporal":
        seeds = (0,)

    all_runs = []
    for seed in seeds:
        if verbose:
            print(f"\n--- прогон seed={seed} ---")
        all_runs.append(run_benchmark(df, k=k, mode=mode, seed=seed, verbose=verbose))

    metric_keys = [f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}",
                   f"HitRate@{k}", "Coverage", "AvgPopularity"]

    rows = []
    for model_idx in range(len(all_runs[0])):
        row = {"model": all_runs[0][model_idx]["model"]}
        for key in metric_keys:
            values = [run[model_idx][key] for run in all_runs]
            row[key] = float(np.mean(values))
            row[key + "_std"] = float(np.std(values))
        rows.append(row)

    return pd.DataFrame(rows).set_index("model"), all_runs


def compare_models_statistically(run_results, k=10, metric="ndcg"):
    """Парные сравнения всех моделей между собой на одном прогоне."""
    from stats_tests import paired_test, format_comparison

    print(f"\n{'=' * 78}")
    print(f"ПРОВЕРКА ЗНАЧИМОСТИ РАЗЛИЧИЙ (метрика: {metric.upper()}@{k})")
    print("=" * 78)
    print("Сравниваем модели попарно на одних и тех же пользователях.\n")

    outputs = []
    for i in range(len(run_results)):
        for j in range(i + 1, len(run_results)):
            a, b = run_results[i], run_results[j]
            assert np.array_equal(a["per_user"]["user_ids"], b["per_user"]["user_ids"])

            res = paired_test(
                a["per_user"][metric], b["per_user"][metric],
                name_a=a["model"], name_b=b["model"],
            )
            outputs.append(res)
            print(format_comparison(res, f"{metric.upper()}@{k}"))
            print()

    return outputs


if __name__ == "__main__":
    import argparse
    from loaders import load_lastfm_360k, load_movielens_100k, describe, has_timestamps

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="lastfm",
                        choices=["lastfm", "movielens"])
    parser.add_argument("--mode", default="random", choices=["random", "temporal", "both"])
    parser.add_argument("--seeds", type=int, default=5,
                        help="сколько разных разбиений прогнать (для random)")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    if args.dataset == "lastfm":
        df = load_lastfm_360k()
        title = "Last.fm-360K"
    else:
        df = load_movielens_100k()
        title = "MovieLens-100k"

    describe(df, title)

    modes = ["random", "temporal"] if args.mode == "both" else [args.mode]

    for mode in modes:
        if mode == "temporal" and not has_timestamps(df):
            print(f"\n[!] {title} не содержит timestamp — временной сплит пропущен.")
            continue

        seeds = tuple(range(args.seeds))
        summary, all_runs = run_repeated(df, k=args.k, mode=mode, seeds=seeds)

        print(f"\n{'=' * 78}")
        label = (f"{len(seeds)} разбиений, среднее ± ст.откл."
                 if mode == "random" else "временной сплит, одно разбиение")
        print(f"РЕЗУЛЬТАТЫ — {title}, режим {mode.upper()} ({label})")
        print("=" * 78)

        cols = [f"NDCG@{args.k}", f"Precision@{args.k}", f"Recall@{args.k}",
                f"HitRate@{args.k}", "Coverage", "AvgPopularity"]
        display = pd.DataFrame(index=summary.index)
        for c in cols:
            if mode == "random":
                display[c] = [f"{m:.4f}±{s:.4f}"
                              for m, s in zip(summary[c], summary[c + "_std"])]
            else:
                display[c] = [f"{m:.4f}" for m in summary[c]]

        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(display)

        compare_models_statistically(all_runs[0], k=args.k, metric="ndcg")
