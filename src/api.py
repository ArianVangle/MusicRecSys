from __future__ import annotations

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from loaders import load_lastfm_360k
from recommender import (
    ColdStartHybrid, MatrixFactorizationSVD, PopularityBaseline, build_user_item_matrix,
)

state: dict = {}

DATASET = "lastfm"


@asynccontextmanager
async def lifespan(app: FastAPI):
    df = load_lastfm_360k()

    n_users = df.user_id.max() + 1
    n_items = df.item_id.max() + 1
    matrix = build_user_item_matrix(df, n_users, n_items)

    model = ColdStartHybrid(
        MatrixFactorizationSVD(n_factors=64),
        PopularityBaseline(),
        min_interactions=3,
    ).fit(matrix)

    item_names = df.drop_duplicates("item_id").set_index("item_id")["item_name"].to_dict()

    state.update(
        model=model, item_names=item_names, n_users=n_users,
        n_items=n_items, dataset=DATASET, matrix=matrix,
    )
    yield
    state.clear()


app = FastAPI(
    title="Music Recommender API",
    description="Рекомендации на основе matrix factorization с fallback на популярность "
                "для новых пользователей",
    version="2.0",
    lifespan=lifespan,
)


class Recommendation(BaseModel):
    item_id: int
    name: str
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    is_cold_start: bool
    strategy: str
    history_size: int
    recommendations: list[Recommendation]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset": state["dataset"],
        "users": int(state["n_users"]),
        "items": int(state["n_items"]),
    }


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, n: int = Query(10, ge=1, le=100)):
    model = state["model"]
    if user_id < 0:
        raise HTTPException(status_code=400, detail="user_id должен быть неотрицательным")

    is_cold = model.is_cold(user_id)
    history_size = (
        int(state["matrix"][user_id].nnz) if user_id < state["n_users"] else 0
    )

    recs = model.recommend(user_id, n=n)
    names = state["item_names"]

    return RecommendResponse(
        user_id=user_id,
        is_cold_start=is_cold,
        strategy="popularity_fallback" if is_cold else "personalized_svd",
        history_size=history_size,
        recommendations=[
            Recommendation(item_id=i, name=str(names.get(i, f"item_{i}")), score=round(s, 4))
            for i, s in recs
        ],
    )


@app.get("/user/{user_id}/history")
def user_history(user_id: int, n: int = Query(10, ge=1, le=100)):
    """История пользователя — чтобы можно было глазами оценить осмысленность рекомендаций."""
    if user_id >= state["n_users"]:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    row = state["matrix"][user_id]
    order = row.data.argsort()[::-1][:n]
    names = state["item_names"]

    return {
        "user_id": user_id,
        "history_size": int(row.nnz),
        "top_items": [
            {"item_id": int(row.indices[i]), "name": str(names.get(int(row.indices[i]), "?")),
             "weight": float(row.data[i])}
            for i in order
        ],
    }
