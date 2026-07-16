from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from _common import ROOT
from cvar_safe.config import load_config
from cvar_safe.trace_replay import load_and_map_trace
from cvar_safe.workload import workload_from_trace
from cvar_safe.simulator import run_policy
from cvar_safe.statistics import aggregate_seed_results, paired_tests


def run(config_path: str) -> Path:
    cfg = load_config(ROOT / config_path if not Path(config_path).is_absolute() else config_path)
    trace_cfg = dict(cfg["trace"])
    trace_cfg["path"] = str(ROOT / trace_cfg["path"])
    mapped, arrival_rates = load_and_map_trace(trace_cfg)
    sim_cfg = dict(cfg["simulation"])
    sim_cfg["duration_intervals"] = len(arrival_rates)
    seeds = [int(cfg["experiment"].get("seed_start", 501)) + i for i in range(int(cfg["experiment"]["seeds"]))]
    policies = list(cfg["experiment"]["policies"])
    out = ROOT / cfg.get("output", {}).get("directory", "results/generated")
    out.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(out / "trace_mapping.csv", index=False)
    rows, traces = [], []
    for seed in seeds:
        workload = workload_from_trace(arrival_rates, sim_cfg, seed)
        for policy in policies:
            result = run_policy(workload, sim_cfg, policy, seed)
            rows.append(result.summary)
            traces.append(result.trace)
    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(out / "trace_replay_seed_results.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(out / "trace_replay_interval_traces.csv", index=False)
    aggregate_seed_results(seed_df).to_csv(out / "trace_replay_aggregate.csv", index=False)
    paired_tests(seed_df).to_csv(out / "trace_replay_paired_tests.csv", index=False)
    print(f"Trace replay results written to {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trace_replay.yaml")
    args = parser.parse_args()
    run(args.config)
