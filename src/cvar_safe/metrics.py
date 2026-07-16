from __future__ import annotations

import numpy as np


def percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def empirical_cvar(values: np.ndarray, alpha: float = 0.95) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    var = np.quantile(values, alpha)
    tail = values[values >= var]
    if tail.size == 0:
        return float(var)
    return float(tail.mean())


def summarize_latencies(values: np.ndarray, slo_ms: float, alpha: float = 0.95) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "slo_violation_pct": float("nan"),
            "cvar95_ms": float("nan"),
            "p95_ms": float("nan"),
            "p99_ms": float("nan"),
            "mean_latency_ms": float("nan"),
        }
    return {
        "slo_violation_pct": float(100.0 * np.mean(values > slo_ms)),
        "cvar95_ms": empirical_cvar(values, alpha),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "mean_latency_ms": float(values.mean()),
    }


def coefficient_of_variation_squared(values: np.ndarray, default: float = 1.0) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2 or np.mean(values) <= 0:
        return float(default)
    return float(np.var(values, ddof=1) / (np.mean(values) ** 2 + 1e-12))
