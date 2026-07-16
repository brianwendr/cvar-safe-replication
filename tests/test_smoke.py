from pathlib import Path
from cvar_safe.config import load_config
from cvar_safe.workload import generate_workload
from cvar_safe.simulator import run_policy


def test_small_shared_seed_smoke():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/primary.yaml")["simulation"]
    cfg = dict(cfg)
    cfg["duration_intervals"] = 30
    cfg["warmup_intervals"] = 3
    cfg["sample_requests_per_interval"] = 30
    workload = generate_workload(cfg, 123)
    a = run_policy(workload, cfg, "cvar_safe", 123)
    b = run_policy(workload, cfg, "hpa_like", 123)
    assert a.summary["seed"] == b.summary["seed"] == 123
    assert len(a.trace) == 27
    assert a.summary["replica_seconds"] > 0
