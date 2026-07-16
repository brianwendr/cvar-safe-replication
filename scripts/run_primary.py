from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from _common import ROOT
from cvar_safe.config import load_config
from cvar_safe.workload import generate_workload
from cvar_safe.simulator import run_policy
from cvar_safe.statistics import aggregate_seed_results, paired_tests, forecast_metrics


def run(config_path: str) -> Path:
    cfg = load_config(ROOT / config_path if not Path(config_path).is_absolute() else config_path)
    sim_cfg = cfg["simulation"]
    policies = list(cfg["experiment"]["policies"])
    seeds = [int(cfg["experiment"].get("seed_start", 101)) + i for i in range(int(cfg["experiment"]["seeds"]))]
    out = ROOT / cfg.get("output", {}).get("directory", "results/generated")
    out.mkdir(parents=True, exist_ok=True)
    seed_rows, traces, forecasts, latency_samples = [], [], [], []
    for seed in seeds:
        workload = generate_workload(sim_cfg, seed)
        for policy in policies:
            result = run_policy(workload, sim_cfg, policy, seed)
            seed_rows.append(result.summary)
            traces.append(result.trace)
            forecasts.append(result.forecast)
            # Deterministic evenly spaced latency sample for the manuscript ECDF.
            values = result.latencies
            sample_count = min(2000, len(values))
            if sample_count > 0:
                import numpy as np
                idx = np.linspace(0, len(values) - 1, sample_count, dtype=int)
                latency_samples.append(pd.DataFrame({
                    "seed": seed, "policy": policy, "latency_ms": values[idx]
                }))
    seed_df = pd.DataFrame(seed_rows)
    trace_df = pd.concat(traces, ignore_index=True)
    forecast_df = pd.concat(forecasts, ignore_index=True)
    latency_sample_df = pd.concat(latency_samples, ignore_index=True)
    seed_df.to_csv(out / "primary_seed_results.csv", index=False)
    trace_df.to_csv(out / "primary_interval_traces.csv", index=False)
    forecast_df.to_csv(out / "forecast_windows.csv", index=False)
    latency_sample_df.to_csv(out / "primary_latency_samples.csv", index=False)
    aggregate_seed_results(seed_df).to_csv(out / "primary_aggregate.csv", index=False)
    paired_tests(seed_df).to_csv(out / "paired_tests_holm.csv", index=False)
    per_seed_forecast, aggregate_forecast = forecast_metrics(forecast_df)
    per_seed_forecast.to_csv(out / "forecast_per_seed.csv", index=False)
    aggregate_forecast.to_csv(out / "forecast_aggregate.csv", index=False)
    print(f"Primary results written to {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/primary.yaml")
    args = parser.parse_args()
    run(args.config)
