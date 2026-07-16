from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import pandas as pd
from scipy import stats

from _common import ROOT
from cvar_safe.config import load_config
from cvar_safe.workload import generate_workload
from cvar_safe.simulator import run_policy
from cvar_safe.statistics import aggregate_seed_results, mean_ci, paired_difference_ci, paired_cohens_dz, matched_rank_biserial, holm_adjust


def _run_scenario(base_cfg: dict, policies: list[str], seeds: list[int], label: str) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        workload = generate_workload(base_cfg, seed)
        for policy in policies:
            result = run_policy(workload, base_cfg, policy, seed)
            row = dict(result.summary)
            row["scenario"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def run(config_path: str) -> Path:
    cfg = load_config(ROOT / config_path if not Path(config_path).is_absolute() else config_path)
    base = cfg["simulation"]
    seed_start = int(cfg["experiment"].get("seed_start", 101))
    seeds = [seed_start + i for i in range(int(cfg["experiment"]["seeds"]))]
    out = ROOT / cfg.get("output", {}).get("directory", "results/generated")
    out.mkdir(parents=True, exist_ok=True)

    # Risk-budget sweep
    sweep_rows = []
    for factor in [0.50, 0.55, 0.60, 0.70, 0.80]:
        scfg = deepcopy(base)
        scfg["cvar_budget_factor"] = factor
        df = _run_scenario(scfg, ["cvar_safe"], seeds, f"risk_budget_{factor:.2f}")
        df["risk_budget_factor"] = factor
        sweep_rows.append(df)
    sweep = pd.concat(sweep_rows, ignore_index=True)
    sweep.to_csv(out / "risk_budget_seed_results.csv", index=False)
    grouped = sweep.groupby("risk_budget_factor", as_index=False).agg(
        slo_violation_pct=("slo_violation_pct", "mean"),
        cvar95_ms=("cvar95_ms", "mean"),
        replica_seconds=("replica_seconds", "mean"),
        scaling_actions=("scaling_actions", "mean"),
    )
    grouped.to_csv(out / "risk_budget_sweep.csv", index=False)

    # Ablations and parameter variants
    variants = [
        ("full", {}, "cvar_safe"),
        ("no_cvar", {}, "no_cvar"),
        ("no_queueing", {}, "no_queueing"),
        ("cooldown_2", {"cooldown_intervals": 2}, "cvar_safe"),
        ("cooldown_4", {"cooldown_intervals": 4}, "cvar_safe"),
        ("window_5", {"window_intervals": 5}, "cvar_safe"),
        ("window_10", {"window_intervals": 10}, "cvar_safe"),
    ]
    ablation_rows = []
    for label, updates, policy in variants:
        scfg = deepcopy(base)
        scfg.update(updates)
        ablation_rows.append(_run_scenario(scfg, [policy], seeds, label))
    ablation = pd.concat(ablation_rows, ignore_index=True)
    ablation.to_csv(out / "ablation_seed_results.csv", index=False)
    ablation.groupby("scenario", as_index=False).agg(
        slo_violation_pct=("slo_violation_pct", "mean"),
        cvar95_ms=("cvar95_ms", "mean"),
        replica_seconds=("replica_seconds", "mean"),
        scaling_actions=("scaling_actions", "mean"),
    ).to_csv(out / "ablation_summary.csv", index=False)

    # Seed-paired component inference against the full policy.
    order = ["full", "no_cvar", "no_queueing", "cooldown_2", "cooldown_4", "window_5", "window_10"]
    names = {
        "full": "Full CVaR-Safe", "no_cvar": "No CVaR trigger (pressure + p99)",
        "no_queueing": "No queueing signal (CVaR + p99)", "cooldown_2": "Cooldown kappa=2",
        "cooldown_4": "Cooldown kappa=4", "window_5": "Window W=5",
        "window_10": "Window W=10",
    }
    full = ablation[ablation.scenario == "full"].set_index("seed")
    inference_rows, raw_t, raw_w = [], [], []
    for label in order:
        group = ablation[ablation.scenario == label].set_index("seed").loc[full.index]
        mean, half = mean_ci(group.slo_violation_pct)
        row = {
            "scenario": label, "configuration": names[label],
            "slo_violations_pct_mean": mean,
            "slo_violations_pct_ci95_halfwidth": half,
        }
        if label == "full":
            row.update({
                "paired_difference_variant_minus_full_pp": float("nan"),
                "paired_difference_ci95_low": float("nan"),
                "paired_difference_ci95_high": float("nan"),
                "paired_t_p_raw": float("nan"), "cohens_dz": float("nan"),
                "wilcoxon_p_raw": float("nan"), "matched_rank_biserial": float("nan"),
            })
        else:
            a = group.slo_violation_pct.to_numpy(float)
            b = full.slo_violation_pct.to_numpy(float)
            t_res = stats.ttest_rel(a, b)
            try:
                w_res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                w_p = float(w_res.pvalue)
            except ValueError:
                w_p = 1.0
            diff, low, high = paired_difference_ci(a, b)
            row.update({
                "paired_difference_variant_minus_full_pp": diff,
                "paired_difference_ci95_low": low,
                "paired_difference_ci95_high": high,
                "paired_t_p_raw": float(t_res.pvalue),
                "cohens_dz": paired_cohens_dz(a, b),
                "wilcoxon_p_raw": w_p,
                "matched_rank_biserial": matched_rank_biserial(a, b),
            })
            raw_t.append(row["paired_t_p_raw"])
            raw_w.append(row["wilcoxon_p_raw"])
        inference_rows.append(row)
    t_adj, w_adj = holm_adjust(raw_t), holm_adjust(raw_w)
    index = 0
    for row in inference_rows:
        if row["scenario"] == "full":
            row["paired_t_p_holm"] = float("nan")
            row["wilcoxon_p_holm"] = float("nan")
        else:
            row["paired_t_p_holm"] = t_adj[index]
            row["wilcoxon_p_holm"] = w_adj[index]
            index += 1
    pd.DataFrame(inference_rows).to_csv(out / "ablation_inference.csv", index=False)

    secondary_rows = []
    for label in order:
        group = ablation[ablation.scenario == label]
        row = {"scenario": label, "configuration": names[label]}
        for metric in ["cvar95_ms", "replica_seconds", "scaling_actions"]:
            mean, half = mean_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_halfwidth"] = half
        secondary_rows.append(row)
    pd.DataFrame(secondary_rows).to_csv(out / "ablation_secondary_summary.csv", index=False)

    # Cross-family mechanism contrasts. These compare each reduced variant with
    # the closest common baseline using the same seeds. Holm correction is
    # applied across the two paired contrasts.
    primary = pd.read_csv(out / "primary_seed_results.csv")
    contrast_specs = [
        ("no_cvar_vs_keda_style", "no_cvar", "keda_style",
         "No CVaR trigger (pressure + p99) vs KEDA-style queue controller"),
        ("no_queueing_vs_hpa_like", "no_queueing", "hpa_like",
         "No queueing signal (CVaR + p99) vs HPA-like controller"),
    ]
    contrast_rows, raw_t, raw_w = [], [], []
    for label, scenario, baseline_policy, description in contrast_specs:
        variant = ablation[ablation.scenario == scenario].set_index("seed").loc[seeds]
        baseline = primary[primary.policy == baseline_policy].set_index("seed").loc[seeds]
        a = variant.slo_violation_pct.to_numpy(float)
        b = baseline.slo_violation_pct.to_numpy(float)
        t_res = stats.ttest_rel(a, b)
        try:
            w_res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            w_p = float(w_res.pvalue)
        except ValueError:
            w_p = 1.0
        diff, low, high = paired_difference_ci(a, b)
        row = {
            "contrast": label,
            "description": description,
            "variant_mean_slo_violations_pct": float(a.mean()),
            "baseline_mean_slo_violations_pct": float(b.mean()),
            "baseline_to_variant_ratio": float(b.mean() / a.mean()),
            "paired_difference_variant_minus_baseline_pp": diff,
            "paired_difference_ci95_low": low,
            "paired_difference_ci95_high": high,
            "paired_t_p_raw": float(t_res.pvalue),
            "cohens_dz": paired_cohens_dz(a, b),
            "wilcoxon_p_raw": w_p,
            "matched_rank_biserial": matched_rank_biserial(a, b),
        }
        raw_t.append(row["paired_t_p_raw"])
        raw_w.append(row["wilcoxon_p_raw"])
        contrast_rows.append(row)
    t_adj, w_adj = holm_adjust(raw_t), holm_adjust(raw_w)
    for row, t_p, w_p in zip(contrast_rows, t_adj, w_adj):
        row["paired_t_p_holm"] = t_p
        row["wilcoxon_p_holm"] = w_p
    pd.DataFrame(contrast_rows).to_csv(out / "mechanism_contrasts.csv", index=False)

    # Fault and tight SLO
    robust_frames = []
    fault_cfg = deepcopy(base)
    fault_cfg["fault"] = {"enabled": True, "start_interval": int(base["duration_intervals"] * 0.50), "duration_intervals": 15, "capacity_factor": 0.65}
    robust_frames.append(_run_scenario(fault_cfg, ["cvar_safe", "hpa_like", "keda_style"], seeds, "fault_injection"))
    tight_cfg = deepcopy(base)
    tight_cfg["slo_ms"] = 300.0
    # Preserve the primary absolute CVaR budget (250 ms) when tightening the p99 SLO.
    tight_cfg["cvar_budget_factor"] = float(base.get("cvar_budget_factor", 0.50)) * float(base.get("slo_ms", 500.0)) / 300.0
    robust_frames.append(_run_scenario(tight_cfg, ["cvar_safe", "hpa_like", "keda_style"], seeds, "tight_slo_300ms"))
    robust = pd.concat(robust_frames, ignore_index=True)
    robust.to_csv(out / "robustness_seed_results.csv", index=False)
    robust.groupby(["scenario", "policy"], as_index=False).agg(
        slo_violation_pct=("slo_violation_pct", "mean"),
        cvar95_ms=("cvar95_ms", "mean"),
        replica_seconds=("replica_seconds", "mean"),
    ).to_csv(out / "robustness_summary.csv", index=False)

    # Bottleneck sensitivity
    bottleneck_frames = []
    for mode in ["app", "mixed", "db"]:
        scfg = deepcopy(base)
        scfg["bottleneck_mode"] = mode
        bottleneck_frames.append(_run_scenario(scfg, ["cvar_safe", "hpa_like", "keda_style"], seeds, mode))
    bottleneck = pd.concat(bottleneck_frames, ignore_index=True)
    bottleneck.to_csv(out / "bottleneck_seed_results.csv", index=False)
    bottleneck.groupby(["scenario", "policy"], as_index=False).agg(
        slo_violation_pct=("slo_violation_pct", "mean"),
        cvar95_ms=("cvar95_ms", "mean"),
        replica_seconds=("replica_seconds", "mean"),
    ).to_csv(out / "bottleneck_summary.csv", index=False)

    # CPU-demand signal sensitivity (partially I/O-bound to CPU-bound service)
    cpu_frames = []
    for factor in [0.40, 0.50, 0.70, 1.00]:
        scfg = deepcopy(base)
        scfg["cpu_utilization_factor"] = factor
        frame = _run_scenario(scfg, ["cvar_safe", "hpa_like", "keda_style"], seeds, f"cpu_factor_{factor:.2f}")
        frame["cpu_utilization_factor"] = factor
        cpu_frames.append(frame)
    cpu_sensitivity = pd.concat(cpu_frames, ignore_index=True)
    cpu_sensitivity.to_csv(out / "cpu_signal_sensitivity_seed_results.csv", index=False)
    cpu_sensitivity.groupby(["cpu_utilization_factor", "policy"], as_index=False).agg(
        slo_violation_pct=("slo_violation_pct", "mean"),
        cvar95_ms=("cvar95_ms", "mean"),
        replica_seconds=("replica_seconds", "mean"),
        scaling_actions=("scaling_actions", "mean"),
    ).to_csv(out / "cpu_signal_sensitivity.csv", index=False)

    print(f"Sensitivity results written to {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/primary.yaml")
    args = parser.parse_args()
    run(args.config)
