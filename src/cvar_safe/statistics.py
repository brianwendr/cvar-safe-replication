from __future__ import annotations

import math
from typing import Iterable
import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    mean = float(x.mean())
    if x.size < 2:
        return mean, 0.0
    sem = stats.sem(x)
    half = float(stats.t.ppf((1.0 + confidence) / 2.0, x.size - 1) * sem)
    return mean, half


def paired_cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.size < 2 or np.std(diff, ddof=1) == 0:
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def paired_difference_ci(a: np.ndarray, b: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean, half = mean_ci(diff, confidence)
    return mean, mean - half, mean + half


def matched_rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    denom = w_pos + w_neg
    return 0.0 if denom == 0 else (w_pos - w_neg) / denom


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def paired_tests(seed_df: pd.DataFrame, reference_policy: str = "cvar_safe") -> pd.DataFrame:
    metrics = ["slo_violation_pct", "cvar95_ms", "p99_ms", "mean_latency_ms", "replica_seconds", "scaling_actions"]
    rows = []
    comparisons = [p for p in seed_df["policy"].unique() if p != reference_policy]
    raw_ps: list[float] = []
    pending: list[dict] = []
    for policy in comparisons:
        joined = seed_df[seed_df.policy == reference_policy].merge(
            seed_df[seed_df.policy == policy], on="seed", suffixes=("_ref", "_cmp")
        )
        for metric in metrics:
            a = joined[f"{metric}_ref"].to_numpy(float)
            b = joined[f"{metric}_cmp"].to_numpy(float)
            t_res = stats.ttest_rel(a, b, nan_policy="omit")
            try:
                w_res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                w_p = float(w_res.pvalue)
            except ValueError:
                w_p = 1.0
            mean_diff, ci_low, ci_high = paired_difference_ci(a, b)
            row = {
                "reference_policy": reference_policy,
                "comparison_policy": policy,
                "metric": metric,
                "n_pairs": len(a),
                "mean_paired_difference_ref_minus_comparison": mean_diff,
                "paired_difference_ci95_low": ci_low,
                "paired_difference_ci95_high": ci_high,
                "paired_t_statistic": float(t_res.statistic),
                "paired_t_p_raw": float(t_res.pvalue),
                "cohens_dz": paired_cohens_dz(a, b),
                "wilcoxon_p": w_p,
                "matched_rank_biserial": matched_rank_biserial(a, b),
            }
            raw_ps.append(row["paired_t_p_raw"])
            pending.append(row)
    adjusted = holm_adjust(raw_ps)
    for row, p_adj in zip(pending, adjusted):
        row["paired_t_p_holm"] = p_adj
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_seed_results(seed_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["slo_violation_pct", "cvar95_ms", "p99_ms", "mean_latency_ms", "replica_seconds", "scaling_actions"]
    rows = []
    for policy, group in seed_df.groupby("policy", sort=False):
        row = {"policy": policy, "seeds": int(group["seed"].nunique())}
        for metric in metrics:
            mean, half = mean_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_halfwidth"] = half
        rows.append(row)
    return pd.DataFrame(rows)


def forecast_metrics(forecast_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = forecast_df.dropna(subset=["predicted_p99_ms", "observed_next_p99_ms"]).copy()
    per_seed = []
    for (policy, seed), g in valid.groupby(["policy", "seed"]):
        err = g["predicted_p99_ms"].to_numpy() - g["observed_next_p99_ms"].to_numpy()
        if len(g) > 2 and np.std(g["predicted_p99_ms"]) > 0 and np.std(g["observed_next_p99_ms"]) > 0:
            r = float(np.corrcoef(g["predicted_p99_ms"], g["observed_next_p99_ms"])[0, 1])
            rho = float(stats.spearmanr(g["predicted_p99_ms"], g["observed_next_p99_ms"]).statistic)
        else:
            r = float("nan")
            rho = float("nan")
        per_seed.append({
            "policy": policy,
            "seed": seed,
            "mae_p99_ms": float(np.mean(np.abs(err))),
            "rmse_p99_ms": float(np.sqrt(np.mean(err ** 2))),
            "pearson_r": r,
            "spearman_rho": rho,
            "windows": len(g),
        })
    per_seed_df = pd.DataFrame(per_seed)
    aggregate = aggregate_forecast(per_seed_df)
    return per_seed_df, aggregate


def aggregate_forecast(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in per_seed_df.groupby("policy", sort=False):
        row = {"policy": policy, "seeds": int(group.seed.nunique()), "windows": int(group.windows.sum())}
        for metric in ["mae_p99_ms", "rmse_p99_ms", "pearson_r", "spearman_rho"]:
            mean, half = mean_ci(group[metric].dropna())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_halfwidth"] = half
        rows.append(row)
    return pd.DataFrame(rows)


def moving_block_bootstrap_correlation(x: np.ndarray, y: np.ndarray, block_length: int = 8, repetitions: int = 1000, seed: int = 2026) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < block_length * 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n - block_length + 1)
    vals = []
    for _ in range(repetitions):
        idx = []
        while len(idx) < n:
            s = int(rng.choice(starts))
            idx.extend(range(s, s + block_length))
        idx = np.asarray(idx[:n])
        if np.std(x[idx]) > 0 and np.std(y[idx]) > 0:
            vals.append(np.corrcoef(x[idx], y[idx])[0, 1])
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
