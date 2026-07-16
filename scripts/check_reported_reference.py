from __future__ import annotations

from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "results" / "generated"
REF = ROOT / "reported_reference"

LABEL = {
    "cvar_safe": "CVaR-Safe",
    "hpa_like": "HPA-like controller",
    "keda_style": "KEDA-style queue controller",
    "rule_based": "Rule-based ladder",
    "pid": "PID controller",
    "predictive": "Predictive threshold",
}


def expected_tables() -> dict[str, pd.DataFrame]:
    p = pd.read_csv(GEN / "primary_aggregate.csv")
    p["method"] = p["policy"].map(LABEL)

    t4 = p[[
        "method", "slo_violation_pct_mean", "slo_violation_pct_ci95_halfwidth",
        "cvar95_ms_mean", "cvar95_ms_ci95_halfwidth", "p99_ms_mean",
        "p99_ms_ci95_halfwidth", "mean_latency_ms_mean",
        "mean_latency_ms_ci95_halfwidth",
    ]].rename(columns={
        "slo_violation_pct_mean": "slo_violations_pct_mean",
        "slo_violation_pct_ci95_halfwidth": "slo_violations_pct_ci95_halfwidth",
    })
    for col in t4.columns[1:]:
        t4[col] = t4[col].round(2)

    t5 = p[[
        "method", "replica_seconds_mean", "replica_seconds_ci95_halfwidth",
        "scaling_actions_mean", "scaling_actions_ci95_halfwidth",
    ]].copy()
    for col in t5.columns[1:]:
        t5[col] = t5[col].round(2)

    t6 = pd.read_csv(GEN / "risk_budget_sweep.csv").copy()
    t6.insert(0, "method", "CVaR-Safe")
    t6 = t6.rename(columns={"slo_violation_pct": "slo_violations_pct"})
    t6["risk_budget_factor"] = t6["risk_budget_factor"].round(2)
    t6["slo_violations_pct"] = t6["slo_violations_pct"].round(3)
    t6["cvar95_ms"] = t6["cvar95_ms"].round(2)
    t6["replica_seconds"] = t6["replica_seconds"].round(1)
    t6["scaling_actions"] = t6["scaling_actions"].round(2)

    f = pd.read_csv(GEN / "forecast_aggregate.csv").query("policy == 'cvar_safe'").iloc[0]
    t7 = pd.DataFrame([{
        "mae_p99_ms_mean": round(f.mae_p99_ms_mean, 2),
        "mae_p99_ms_ci95_halfwidth": round(f.mae_p99_ms_ci95_halfwidth, 2),
        "rmse_p99_ms_mean": round(f.rmse_p99_ms_mean, 2),
        "rmse_p99_ms_ci95_halfwidth": round(f.rmse_p99_ms_ci95_halfwidth, 2),
        "pearson_r_mean": round(f.pearson_r_mean, 3),
        "pearson_r_ci95_halfwidth": round(f.pearson_r_ci95_halfwidth, 3),
        "spearman_rho_mean": round(f.spearman_rho_mean, 3),
        "spearman_rho_ci95_halfwidth": round(f.spearman_rho_ci95_halfwidth, 3),
        "windows": int(f.windows),
        "seed_units": int(f.seeds),
    }])

    rob = pd.read_csv(GEN / "robustness_summary.csv")
    scen = {"fault_injection": "Fault injection", "tight_slo_300ms": "Tight SLO 300 ms"}
    t8 = pd.DataFrame({
        "scenario": rob.scenario.map(scen),
        "method": rob.policy.map(LABEL),
        "slo_violations_pct": rob.slo_violation_pct.round(2),
        "cvar95_ms": rob.cvar95_ms.round(2),
        "replica_seconds": rob.replica_seconds.round(1),
    })

    b = pd.read_csv(GEN / "bottleneck_summary.csv")
    order_mode = {"app": 0, "mixed": 1, "db": 2}
    order_policy = {"cvar_safe": 0, "hpa_like": 1, "keda_style": 2}
    b["_mode"] = b.scenario.map(order_mode)
    b["_policy"] = b.policy.map(order_policy)
    b = b.sort_values(["_mode", "_policy"])
    t9 = pd.DataFrame({
        "mode": b.scenario,
        "method": b.policy.map(LABEL),
        "slo_violations_pct": b.slo_violation_pct.round(2),
        "cvar95_ms": b.cvar95_ms.round(2),
        "replica_seconds": b.replica_seconds.round(1),
    }).reset_index(drop=True)

    inf = pd.read_csv(GEN / "ablation_inference.csv")
    t10 = inf[[
        "configuration", "slo_violations_pct_mean", "slo_violations_pct_ci95_halfwidth",
        "paired_difference_variant_minus_full_pp", "paired_difference_ci95_low",
        "paired_difference_ci95_high", "paired_t_p_holm", "cohens_dz",
        "wilcoxon_p_holm",
    ]].copy()
    for col in [
        "slo_violations_pct_mean", "slo_violations_pct_ci95_halfwidth",
        "paired_difference_variant_minus_full_pp", "paired_difference_ci95_low",
        "paired_difference_ci95_high", "cohens_dz",
    ]:
        t10[col] = t10[col].round(2)
    for col in ["paired_t_p_holm", "wilcoxon_p_holm"]:
        t10[col] = t10[col].round(6)

    sec = pd.read_csv(GEN / "ablation_secondary_summary.csv")
    t11 = sec[[
        "configuration", "cvar95_ms_mean", "cvar95_ms_ci95_halfwidth",
        "replica_seconds_mean", "replica_seconds_ci95_halfwidth",
        "scaling_actions_mean", "scaling_actions_ci95_halfwidth",
    ]].copy()
    for col in [
        "cvar95_ms_mean", "cvar95_ms_ci95_halfwidth",
        "scaling_actions_mean", "scaling_actions_ci95_halfwidth",
    ]:
        t11[col] = t11[col].round(2)
    for col in ["replica_seconds_mean", "replica_seconds_ci95_halfwidth"]:
        t11[col] = t11[col].round(1)

    mech = pd.read_csv(GEN / "mechanism_contrasts.csv")
    mechanism = mech[[
        "description", "variant_mean_slo_violations_pct",
        "baseline_mean_slo_violations_pct", "baseline_to_variant_ratio",
        "paired_difference_variant_minus_baseline_pp",
        "paired_difference_ci95_low", "paired_difference_ci95_high",
        "paired_t_p_holm", "cohens_dz", "wilcoxon_p_holm",
    ]].copy()
    for col in [
        "variant_mean_slo_violations_pct",
        "baseline_mean_slo_violations_pct",
        "baseline_to_variant_ratio",
        "paired_difference_variant_minus_baseline_pp",
        "paired_difference_ci95_low", "paired_difference_ci95_high",
        "cohens_dz",
    ]:
        mechanism[col] = mechanism[col].round(2)
    for col in ["paired_t_p_holm", "wilcoxon_p_holm"]:
        mechanism[col] = mechanism[col].round(8)

    tr = pd.read_csv(GEN / "trace_replay_aggregate.csv")
    tr["method"] = tr.policy.map(LABEL)
    t12 = tr[[
        "method", "slo_violation_pct_mean", "slo_violation_pct_ci95_halfwidth",
        "cvar95_ms_mean", "cvar95_ms_ci95_halfwidth", "p99_ms_mean",
        "p99_ms_ci95_halfwidth", "replica_seconds_mean",
        "replica_seconds_ci95_halfwidth", "scaling_actions_mean",
        "scaling_actions_ci95_halfwidth",
    ]].rename(columns={
        "slo_violation_pct_mean": "slo_violations_pct_mean",
        "slo_violation_pct_ci95_halfwidth": "slo_violations_pct_ci95_halfwidth",
    })
    for col in ["slo_violations_pct_mean", "slo_violations_pct_ci95_halfwidth"]:
        t12[col] = t12[col].round(3)
    for col in [
        "cvar95_ms_mean", "cvar95_ms_ci95_halfwidth", "p99_ms_mean",
        "p99_ms_ci95_halfwidth", "scaling_actions_mean",
        "scaling_actions_ci95_halfwidth",
    ]:
        t12[col] = t12[col].round(2)
    for col in ["replica_seconds_mean", "replica_seconds_ci95_halfwidth"]:
        t12[col] = t12[col].round(0).astype(int)

    return {
        "table_4_primary_latency.csv": t4,
        "table_5_resource_control.csv": t5,
        "table_6_risk_sweep.csv": t6,
        "table_7_forecast_calibration.csv": t7,
        "table_8_robustness.csv": t8,
        "table_9_bottleneck.csv": t9,
        "table_10_ablation_inference.csv": t10,
        "table_11_ablation_secondary.csv": t11,
        "section_5_5_mechanism_contrasts.csv": mechanism,
        "table_12_trace_replay.csv": t12,
    }


def main() -> None:
    problems: list[str] = []
    if not GEN.exists():
        raise SystemExit("Canonical output directory is missing: results/generated")

    for name, expected in expected_tables().items():
        path = REF / name
        if not path.exists():
            problems.append(f"missing {name}")
            continue
        actual = pd.read_csv(path)
        try:
            assert_frame_equal(
                actual.reset_index(drop=True), expected.reset_index(drop=True),
                check_dtype=False, check_exact=False, rtol=0, atol=1e-9,
            )
        except AssertionError as exc:
            problems.append(f"{name}: {str(exc).splitlines()[0]}")

    stale = [name for name in ("table_10_ablation.csv", "table_11_trace_replay.csv", "table_11_local_pilot.csv") if (REF / name).exists()]
    if stale:
        problems.append(f"stale reference files remain: {stale}")

    if problems:
        raise SystemExit("Manuscript-table alignment check failed:\n- " + "\n- ".join(problems))
    print("Tables 4-12 and the Section 5.5 mechanism contrasts match the canonical generated outputs at manuscript display precision.")


if __name__ == "__main__":
    main()
