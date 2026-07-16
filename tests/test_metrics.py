import numpy as np
from cvar_safe.metrics import empirical_cvar, summarize_latencies


def test_empirical_cvar_monotone_tail():
    x = np.arange(1, 101, dtype=float)
    assert empirical_cvar(x, 0.95) >= np.quantile(x, 0.95)


def test_summary_violation_rate():
    s = summarize_latencies(np.array([100.0, 600.0]), 500.0)
    assert s["slo_violation_pct"] == 50.0
