from cvar_safe.controllers import Observation, make_controller


def observation(**updates):
    base = dict(p95_ms=200.0, p99_ms=250.0, cvar95_ms=260.0, smoothed_cvar_ms=260.0,
                queue=0.0, utilization=0.2, replicas=4, pressure=0.0,
                predicted_p99_ms=250.0, slo_ms=500.0)
    base.update(updates)
    return Observation(**base)


def test_cvar_safe_scales_up_on_risk():
    cfg = {"min_replicas": 2, "max_replicas": 20, "cooldown_intervals": 3,
           "cvar_budget_factor": 0.9, "queue_threshold": 90, "window_intervals": 8,
           "pressure_threshold": 0.045, "p99_safety_factor": 1.1}
    controller = make_controller("cvar_safe", cfg)
    assert controller.decide(observation(smoothed_cvar_ms=480.0), 10) == 5


def test_hpa_like_name_is_explicit():
    cfg = {"min_replicas": 2, "max_replicas": 20, "cooldown_intervals": 3, "queue_threshold": 90}
    controller = make_controller("hpa_like", cfg)
    assert controller.name == "hpa_like"
