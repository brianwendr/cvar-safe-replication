from pathlib import Path
from cvar_safe.trace_replay import load_and_map_trace


def test_trace_mapping_bounds():
    root = Path(__file__).resolve().parents[1]
    frame, arrivals = load_and_map_trace({
        "path": root / "data/trace/alibaba_2018_machine_cpu_profile_300bins.csv",
        "utilization_column": "cpu_util_percent",
        "arrival_rate_min": 90.0,
        "arrival_rate_max": 430.0,
    })
    assert len(frame) == 300
    assert arrivals.min() >= 90.0 - 1e-9
    assert arrivals.max() <= 430.0 + 1e-9
