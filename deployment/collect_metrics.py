from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def cvar(values: np.ndarray, alpha: float = 0.95) -> float:
    threshold = np.quantile(values, alpha)
    return float(values[values >= threshold].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-csv", required=True)
    parser.add_argument("--slo-ms", type=float, default=500.0)
    parser.add_argument("--output", default="results/local_summary.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.latency_csv)
    values = frame.loc[frame.status == 200, "latency_ms"].to_numpy(float)
    summary = pd.DataFrame([{
        "requests": len(values),
        "mean_latency_ms": values.mean(),
        "p95_ms": np.percentile(values, 95),
        "p99_ms": np.percentile(values, 99),
        "cvar95_ms": cvar(values),
        "slo_violation_pct": 100.0 * np.mean(values > args.slo_ms),
    }])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
