from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Workload:
    arrival_rates: np.ndarray
    arrivals: np.ndarray
    burst_mask: np.ndarray
    service_uniforms: np.ndarray
    service_normals: np.ndarray


def generate_workload(cfg: dict, seed: int) -> Workload:
    rng = np.random.default_rng(seed)
    duration = int(cfg["duration_intervals"])
    dt = float(cfg.get("decision_interval_seconds", 1.0))
    t = np.arange(duration, dtype=float)
    base = float(cfg["baseline_arrival_rate"])
    amplitude = float(cfg.get("sinusoidal_amplitude", 0.3))
    period = max(float(cfg.get("sinusoidal_period_intervals", duration)), 1.0)
    rates = base * (1.0 + amplitude * np.sin(2.0 * np.pi * t / period))
    burst_mask = np.zeros(duration, dtype=bool)
    for _ in range(int(cfg.get("burst_count", 0))):
        start = int(rng.integers(0, max(duration - 1, 1)))
        length = int(rng.integers(int(cfg.get("burst_duration_min", 5)), int(cfg.get("burst_duration_max", 20)) + 1))
        stop = min(duration, start + length)
        multiplier = float(rng.uniform(float(cfg.get("burst_multiplier_min", 1.5)), float(cfg.get("burst_multiplier_max", 3.0))))
        rates[start:stop] += base * (multiplier - 1.0)
        burst_mask[start:stop] = True
    rates = np.maximum(rates, 0.1)
    arrivals = rng.poisson(rates * dt).astype(int)
    samples = int(cfg.get("sample_requests_per_interval", 180))
    service_uniforms = rng.random((duration, samples))
    service_normals = rng.standard_normal((duration, samples))
    return Workload(rates, arrivals, burst_mask, service_uniforms, service_normals)


def workload_from_trace(arrival_rates: np.ndarray, cfg: dict, seed: int) -> Workload:
    rng = np.random.default_rng(seed)
    rates = np.asarray(arrival_rates, dtype=float)
    dt = float(cfg.get("decision_interval_seconds", 1.0))
    arrivals = rng.poisson(rates * dt).astype(int)
    samples = int(cfg.get("sample_requests_per_interval", 180))
    service_uniforms = rng.random((len(rates), samples))
    service_normals = rng.standard_normal((len(rates), samples))
    return Workload(rates, arrivals, np.zeros(len(rates), dtype=bool), service_uniforms, service_normals)
