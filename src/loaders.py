"""
Загрузка Last.fm-360K в единый формат.

Схема после загрузки:
    user_id (int), item_id (int), weight (float), item_name (str)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_lastfm_360k(
    path: str | Path | None = None,
    min_plays: int = 1,
    min_user_interactions: int = 5,
    min_item_interactions: int = 20,
    max_users: int | None = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    if path is None:
        path = DATA_DIR / "lastfm-dataset-360K" / "usersha1-artmbid-artname-plays.tsv"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл Last.fm-360K: {path}\n"
            "Скачайте датасет со страницы Zenodo:\n"
            "  https://zenodo.org/records/6090214\n"
            "Нужен файл usersha1-artmbid-artname-plays.tsv "
        )

    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_sha1", "artist_mbid", "artist_name", "plays"],
        usecols=["user_sha1", "artist_name", "plays"],
        na_values=["", " "],
        on_bad_lines="skip",
        dtype={"user_sha1": str, "artist_name": str, "plays": "float32"},
    )

    before = len(df)
    df = df.dropna(subset=["user_sha1", "artist_name", "plays"])
    df = df[df.plays >= min_plays]
    print(f"[lastfm] отброшено битых/пустых строк: {before - len(df):,}")

    if max_users is not None:
        rng = np.random.default_rng(seed)
        users = df.user_sha1.unique()
        if len(users) > max_users:
            keep = set(rng.choice(users, size=max_users, replace=False))
            df = df[df.user_sha1.isin(keep)]
            print(f"[lastfm] выборка пользователей: {max_users:,} из {len(users):,}")

    df = df.rename(columns={"artist_name": "item_name", "plays": "weight"})
    df = _apply_kcore(df, "user_sha1", "item_name", min_user_interactions, min_item_interactions)
    return _encode_ids(df, user_col="user_sha1", item_col="item_name")


def load_movielens_100k(
    path: str | Path | None = None,
    item_path: str | Path | None = None,
    positive_threshold: float = 4.0,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> pd.DataFrame:
    if path is None:
        path = DATA_DIR / "u.data"
    if item_path is None:
        item_path = DATA_DIR / "u.item"

    path, item_path = Path(path), Path(item_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл MovieLens: {path}\n"
            "Скачайте https://files.grouplens.org/datasets/movielens/ml-100k.zip\n"
            "и положите u.data и u.item в data/"
        )

    df = pd.read_csv(path, sep="\t", names=["user_raw", "item_raw", "rating", "timestamp"])

    items = pd.read_csv(
        item_path, sep="|", encoding="latin-1", header=None,
        usecols=[0, 1], names=["item_raw", "item_name"],
    )
    df = df.merge(items, on="item_raw", how="left")

    df = df[df.rating >= positive_threshold].copy()
    df["weight"] = df.rating.astype("float32")

    df = _apply_kcore(df, "user_raw", "item_raw", min_user_interactions, min_item_interactions)
    return _encode_ids(df, user_col="user_raw", item_col="item_raw")


def has_timestamps(df: pd.DataFrame) -> bool:
    return bool("timestamp" in df.columns and df["timestamp"].notna().any())


def _apply_kcore(df, user_col, item_col, min_user, min_item, max_iter=10):
    """
    Итеративная k-core фильтрация.
    """
    for _ in range(max_iter):
        n_before = len(df)

        item_counts = df[item_col].value_counts()
        df = df[df[item_col].isin(item_counts[item_counts >= min_item].index)]

        user_counts = df[user_col].value_counts()
        df = df[df[user_col].isin(user_counts[user_counts >= min_user].index)]

        if len(df) == n_before:
            break
    return df


def _encode_ids(df, user_col, item_col):
    """Кодируем произвольные ID (sha1-хеши, строки) в компактные целые 0..N-1,
      чтобы использовать их как индексы разреженной матрицы.
    """
    user_codes = {u: i for i, u in enumerate(df[user_col].unique())}
    item_codes = {t: i for i, t in enumerate(df[item_col].unique())}

    out = pd.DataFrame({
        "user_id": df[user_col].map(user_codes).astype("int32"),
        "item_id": df[item_col].map(item_codes).astype("int32"),
        "weight": df["weight"].astype("float32"),
    })

    if "item_name" in df.columns:
        out["item_name"] = df["item_name"].values
    else:
        out["item_name"] = df[item_col].astype(str).values

    if "timestamp" in df.columns:
        out["timestamp"] = df["timestamp"].values

    return out.reset_index(drop=True)


def describe(df: pd.DataFrame, name: str = "dataset"):
    n_users = df.user_id.nunique()
    n_items = df.item_id.nunique()
    density = len(df) / (n_users * n_items)
    per_user = df.groupby("user_id").size()

    print(f"\n=== {name} ===")
    print(f"Взаимодействий: {len(df):,}")
    print(f"Пользователей:  {n_users:,}")
    print(f"Объектов:       {n_items:,}")
    print(f"Плотность:      {density:.4%}")
    print(f"На пользователя: медиана {per_user.median():.0f}, "
          f"среднее {per_user.mean():.1f}, макс {per_user.max()}")
