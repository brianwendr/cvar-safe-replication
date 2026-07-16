from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def load_and_map_trace(trace_cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    path = Path(trace_cfg["path"])
    frame = pd.read_csv(path)
    column = trace_cfg.get("utilization_column", "cpu_util_percent")
    if column not in frame:
        raise KeyError(f"Missing trace column: {column}")
    raw = frame[column].to_numpy(dtype=float)
    if np.nanmax(raw) == np.nanmin(raw):
        normalized = np.zeros_like(raw)
    else:
        normalized = (raw - np.nanmin(raw)) / (np.nanmax(raw) - np.nanmin(raw))
    lam_min = float(trace_cfg.get("arrival_rate_min", 90.0))
    lam_max = float(trace_cfg.get("arrival_rate_max", 430.0))
    arrivals = lam_min + normalized * (lam_max - lam_min)
    mapped = frame.copy()
    mapped["normalized_0_1"] = normalized
    mapped["arrival_rate_rps"] = arrivals
    mapped["mapping_equation"] = f"lambda={lam_min}+u_norm*({lam_max}-{lam_min})"
    return mapped, arrivals
