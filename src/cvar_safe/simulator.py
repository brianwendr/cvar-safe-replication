from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd

from .controllers import Observation, make_controller
from .metrics import coefficient_of_variation_squared, empirical_cvar, summarize_latencies
from .workload import Workload


@dataclass
class RunResult:
    summary: dict
    trace: pd.DataFrame
    latencies: np.ndarray
    forecast: pd.DataFrame


def pressure_score(rho: float, ca2: float, cs2: float, replicas: int) -> float:
    """Queueing-inspired heuristic score; not a waiting-time approximation."""
    rho_capped = min(max(rho, 0.0), 0.995)
    exponent = math.sqrt(2.0 * (replicas + 1.0))
    variability = max((ca2 + cs2) / 2.0, 0.01)
    return float((variability / max(replicas, 1)) * (rho_capped ** exponent) / max(1.0 - rho_capped, 1e-3))


def _service_samples(cfg: dict, normals: np.ndarray) -> np.ndarray:
    mean_ms = float(cfg.get("service_mean_ms", 55.0))
    sigma = float(cfg.get("service_sigma", 0.6))
    mu = math.log(mean_ms) - 0.5 * sigma * sigma
    return np.exp(mu + sigma * normals)


def run_policy(workload: Workload, cfg: dict, policy_name: str, seed: int) -> RunResult:
    controller = make_controller(policy_name, cfg)
    duration = len(workload.arrivals)
    dt = float(cfg.get("decision_interval_seconds", 1.0))
    sample_n = int(cfg.get("sample_requests_per_interval", workload.service_normals.shape[1]))
    replicas = int(cfg.get("initial_replicas", 4))
    active_replicas = replicas
    activation_delay = int(cfg.get("activation_delay_intervals", 2))
    pending: deque[tuple[int, int]] = deque()
    queue = 0.0
    latency_windows: deque[np.ndarray] = deque(maxlen=int(cfg.get("window_intervals", 8)))
    arrival_history: deque[float] = deque(maxlen=int(cfg.get("window_intervals", 8)))
    smoothed_cvar = float(cfg.get("slo_ms", 500.0)) * 0.50
    utilization_ema = 0.0
    all_latency_chunks: list[np.ndarray] = []
    trace_rows: list[dict] = []
    forecast_rows: list[dict] = []
    actions = 0
    replica_seconds = 0.0
    previous_p99 = float(cfg.get("service_mean_ms", 55.0))

    fault_cfg = cfg.get("fault", {}) or {}
    fault_enabled = bool(fault_cfg.get("enabled", False))
    fault_start = int(fault_cfg.get("start_interval", duration // 2))
    fault_duration = int(fault_cfg.get("duration_intervals", 15))
    fault_capacity_factor = float(fault_cfg.get("capacity_factor", 0.70))
    bottleneck = str(cfg.get("bottleneck_mode", "app"))

    for t in range(duration):
        while pending and pending[0][0] <= t:
            _, active_replicas = pending.popleft()

        capacity_factor = 1.0
        if fault_enabled and fault_start <= t < fault_start + fault_duration:
            capacity_factor = fault_capacity_factor
        per_replica_capacity = float(cfg.get("per_replica_capacity_rps", 52.0)) * dt
        capacity = max(active_replicas * per_replica_capacity * capacity_factor, 1.0)
        arrivals = float(workload.arrivals[t])
        queue_before = queue
        offered = queue_before + arrivals
        processed = min(offered, capacity)
        queue = max(0.0, offered - capacity)
        utilization = min(offered / capacity, 1.50)

        service = _service_samples(cfg, workload.service_normals[t, :sample_n])
        congestion = max(utilization - 0.72, 0.0)
        queue_wait_ms = 300.0 * queue_before / capacity
        nonlinear_wait_ms = 8.0 * (congestion ** 2) / max(1.02 - min(utilization, 1.01), 0.05)
        if bottleneck == "mixed":
            downstream = 35.0 + 70.0 * max(utilization - 0.80, 0.0)
        elif bottleneck == "db":
            downstream = 120.0 + 60.0 * max(utilization - 0.70, 0.0)
        else:
            downstream = 8.0
        # Deterministic service samples shared across policies; controller affects queueing terms only.
        latencies = service + queue_wait_ms + nonlinear_wait_ms + downstream
        # Modest tail amplification under sustained backlog.
        if queue_before > float(cfg.get("queue_threshold", 90.0)):
            tail_mask = workload.service_uniforms[t, :sample_n] > 0.94
            latencies = latencies.copy()
            latencies[tail_mask] += 0.40 * queue_wait_ms + 120.0 * congestion
        latency_windows.append(latencies)
        arrival_history.append(arrivals)
        window_values = np.concatenate(list(latency_windows))
        stats = summarize_latencies(window_values, float(cfg.get("slo_ms", 500.0)), float(cfg.get("cvar_alpha", 0.95)))
        current_cvar = stats["cvar95_ms"]
        beta = float(cfg.get("ema_beta", 0.30))
        smoothed_cvar = beta * current_cvar + (1.0 - beta) * smoothed_cvar

        arrivals_arr = np.asarray(arrival_history, dtype=float)
        interarrival_proxy = 1.0 / np.maximum(arrivals_arr, 1.0)
        ca2 = coefficient_of_variation_squared(interarrival_proxy, default=1.0)
        cs2 = math.exp(float(cfg.get("service_sigma", 0.6)) ** 2) - 1.0
        rho = workload.arrival_rates[t] * dt / max(capacity, 1.0)
        pressure = pressure_score(rho, ca2, cs2, active_replicas)
        predicted_p99 = max(stats["p99_ms"], previous_p99) + 38.0 * math.log1p(pressure * 20.0) + 0.55 * queue
        cpu_utilization = min(utilization * float(cfg.get("cpu_utilization_factor", 0.50)), 1.50)
        utilization_ema = 0.82 * utilization_ema + 0.18 * cpu_utilization
        observed_utilization = cpu_utilization
        obs = Observation(
            p95_ms=stats["p95_ms"], p99_ms=stats["p99_ms"], cvar95_ms=current_cvar,
            smoothed_cvar_ms=smoothed_cvar, queue=queue, utilization=observed_utilization,
            replicas=replicas, pressure=pressure, predicted_p99_ms=predicted_p99,
            slo_ms=float(cfg.get("slo_ms", 500.0)),
        )
        target = controller.decide(obs, t)
        if target != replicas:
            actions += 1
            replicas = target
            pending.append((t + activation_delay, replicas))

        replica_seconds += active_replicas * dt
        all_latency_chunks.append(latencies)
        trace_rows.append({
            "seed": seed,
            "policy": policy_name,
            "interval": t,
            "arrival_rate": workload.arrival_rates[t],
            "arrivals": arrivals,
            "queue": queue,
            "utilization": cpu_utilization,
            "offered_load_ratio": utilization,
            "replicas_target": replicas,
            "replicas_active": active_replicas,
            "p95_ms": stats["p95_ms"],
            "p99_ms": stats["p99_ms"],
            "cvar95_ms": current_cvar,
            "smoothed_cvar_ms": smoothed_cvar,
            "pressure": pressure,
            "predicted_next_p99_ms": predicted_p99,
            "burst": int(workload.burst_mask[t]),
        })
        forecast_rows.append({
            "seed": seed,
            "policy": policy_name,
            "interval": t,
            "predicted_p99_ms": predicted_p99,
            "observed_next_p99_ms": np.nan,
            "pressure": pressure,
        })
        previous_p99 = stats["p99_ms"]

    all_latencies = np.concatenate(all_latency_chunks)
    warmup = int(cfg.get("warmup_intervals", 0))
    if warmup > 0:
        keep = trace_rows[warmup:]
        # Each interval contributes equal-size sampled arrays.
        all_latencies = np.concatenate(all_latency_chunks[warmup:])
        trace_rows = keep
        forecast_rows = forecast_rows[warmup:]
        replica_seconds = sum(row["replicas_active"] * dt for row in trace_rows)
        actions = sum(
            int(trace_rows[i]["replicas_target"] != trace_rows[i - 1]["replicas_target"])
            for i in range(1, len(trace_rows))
        )

    trace_df = pd.DataFrame(trace_rows)
    forecast_df = pd.DataFrame(forecast_rows)
    if not forecast_df.empty:
        next_values = trace_df["p99_ms"].shift(-1).to_numpy()
        forecast_df = forecast_df.iloc[: len(next_values)].copy()
        forecast_df["observed_next_p99_ms"] = next_values
        forecast_df = forecast_df.iloc[:-1].reset_index(drop=True)

    summary = summarize_latencies(all_latencies, float(cfg.get("slo_ms", 500.0)), float(cfg.get("cvar_alpha", 0.95)))
    summary.update({
        "seed": seed,
        "policy": policy_name,
        "replica_seconds": float(replica_seconds),
        "scaling_actions": int(actions),
    })
    return RunResult(summary, trace_df, all_latencies, forecast_df)
