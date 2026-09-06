"""
Статистическая проверка, отличима ли разница между моделями от шума.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """
    Доверительный интервал среднего методом бутстрапа.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)

    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def paired_test(values_a: np.ndarray, values_b: np.ndarray,
                name_a: str = "A", name_b: str = "B") -> dict:
    """
    Парное сравнение двух моделей по пользователям.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    assert len(a) == len(b), "выборки должны быть одинаковой длины и в одном порядке"

    diff = a - b
    nonzero = diff[diff != 0]

    if len(nonzero) == 0:
        p_value = 1.0
        statistic = 0.0
    else:
        statistic, p_value = stats.wilcoxon(a, b, zero_method="wilcox")

    cohens_dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0

    rank_biserial = _rank_biserial(nonzero)

    mean_diff, lo, hi = bootstrap_ci(diff)

    return {
        "model_a": name_a,
        "model_b": name_b,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": mean_diff,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": float(p_value),
        "cohens_dz": float(cohens_dz),
        "rank_biserial": float(rank_biserial),
        "wins_a": int((diff > 0).sum()),
        "wins_b": int((diff < 0).sum()),
        "ties": int((diff == 0).sum()),
        "significant": bool(p_value < 0.05 and (lo > 0 or hi < 0)),
    }


def _rank_biserial(nonzero_diff: np.ndarray) -> float:
    """
    Rank-biserial correlation для критерия Уилкоксона.
    """
    if len(nonzero_diff) == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(nonzero_diff))
    w_plus = ranks[nonzero_diff > 0].sum()
    w_minus = ranks[nonzero_diff < 0].sum()
    total = w_plus + w_minus
    return (w_plus - w_minus) / total if total > 0 else 0.0


def format_comparison(result: dict, metric_name: str = "метрика") -> str:

    verdict = "ЗНАЧИМА" if result["significant"] else "НЕ значима (может быть шумом)"
    better = result["model_a"] if result["mean_diff"] > 0 else result["model_b"]

    p = result["p_value"]
    p_str = "< 1e-16" if p < 1e-16 else f"= {p:.2e}"

    return (
        f"{result['model_a']} vs {result['model_b']} по {metric_name}:\n"
        f"  средние: {result['mean_a']:.4f} против {result['mean_b']:.4f}\n"
        f"  разница: {result['mean_diff']:+.4f} "
        f"[95% ДИ: {result['ci_low']:+.4f}, {result['ci_high']:+.4f}]\n"
        f"  p-value {p_str}\n"
        f"  размер эффекта: rank-biserial {result['rank_biserial']:+.3f}, "
        f"Cohen's dz {result['cohens_dz']:+.3f}\n"
        f"  по пользователям: {result['wins_a']} за {result['model_a']}, "
        f"{result['wins_b']} за {result['model_b']}, {result['ties']} ничья\n"
        f"  ВЫВОД: разница {verdict}"
        + (f", лучше {better}" if result["significant"] else "")
    )
