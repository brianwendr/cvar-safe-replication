from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class Observation:
    p95_ms: float
    p99_ms: float
    cvar95_ms: float
    smoothed_cvar_ms: float
    queue: float
    utilization: float
    replicas: int
    pressure: float
    predicted_p99_ms: float
    slo_ms: float


class BaseController:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.last_action_interval = -10**9
        self.low_streak = 0
        self.high_queue_streak = 0
        self.integral = 0.0
        self.previous_error = 0.0

    def _cooldown(self, interval: int) -> bool:
        return interval - self.last_action_interval < int(self.cfg.get("cooldown_intervals", 3))

    def _bounded(self, target: int) -> int:
        return int(max(int(self.cfg.get("min_replicas", 2)), min(int(self.cfg.get("max_replicas", 20)), target)))

    def _record(self, interval: int, old: int, new: int) -> int:
        new = self._bounded(new)
        if new != old:
            self.last_action_interval = interval
        return new

    def decide(self, obs: Observation, interval: int) -> int:
        return obs.replicas


class CVaRSafeController(BaseController):
    name = "cvar_safe"

    def __init__(self, cfg: dict, *, use_cvar: bool = True, use_queueing: bool = True):
        super().__init__(cfg)
        self.use_cvar = use_cvar
        self.use_queueing = use_queueing

    def decide(self, obs: Observation, interval: int) -> int:
        q_threshold = float(self.cfg.get("queue_threshold", 90.0))
        self.high_queue_streak = self.high_queue_streak + 1 if obs.queue > q_threshold else 0
        safe = obs.p95_ms < 0.80 * obs.slo_ms and obs.utilization < float(self.cfg.get("cvar_safe_down_utilization", 0.20))
        self.low_streak = self.low_streak + 1 if safe else 0
        if self._cooldown(interval):
            return obs.replicas
        risk_breach = self.use_cvar and obs.smoothed_cvar_ms > float(self.cfg.get("cvar_budget_factor", 0.50)) * obs.slo_ms
        p99_breach = obs.p99_ms > float(self.cfg.get("p99_safety_factor", 1.10)) * obs.slo_ms
        pressure_breach = self.use_queueing and obs.pressure > float(self.cfg.get("pressure_threshold", 0.1))
        queue_breach = self.use_queueing and self.high_queue_streak >= 2
        if risk_breach or p99_breach or pressure_breach or queue_breach:
            return self._record(interval, obs.replicas, obs.replicas + 1)
        if self.low_streak >= int(self.cfg.get("window_intervals", 8)):
            return self._record(interval, obs.replicas, obs.replicas - 1)
        return obs.replicas


class HPALikeController(BaseController):
    name = "hpa_like"

    def decide(self, obs: Observation, interval: int) -> int:
        q_threshold = float(self.cfg.get("queue_threshold", 90.0))
        low = obs.utilization < 0.30 and obs.queue < 0.15 * q_threshold
        self.low_streak = self.low_streak + 1 if low else 0
        if self._cooldown(interval):
            return obs.replicas
        if obs.utilization > 0.60 or obs.queue > 1.25 * q_threshold:
            return self._record(interval, obs.replicas, obs.replicas + 1)
        if self.low_streak >= int(self.cfg.get("baseline_low_streak_intervals", 2)):
            return self._record(interval, obs.replicas, obs.replicas - 1)
        return obs.replicas


class KEDAStyleController(BaseController):
    name = "keda_style"

    def decide(self, obs: Observation, interval: int) -> int:
        per_replica = obs.queue / max(obs.replicas, 1)
        low = per_replica < 8.0 and obs.utilization < 0.45
        self.low_streak = self.low_streak + 1 if low else 0
        if self._cooldown(interval):
            return obs.replicas
        if per_replica > 36.0:
            return self._record(interval, obs.replicas, obs.replicas + 1)
        if self.low_streak >= int(self.cfg.get("baseline_low_streak_intervals", 2)):
            return self._record(interval, obs.replicas, obs.replicas - 1)
        return obs.replicas


class RuleBasedController(BaseController):
    name = "rule_based"

    def decide(self, obs: Observation, interval: int) -> int:
        q_threshold = float(self.cfg.get("queue_threshold", 90.0))
        low = obs.p95_ms < 0.70 * obs.slo_ms and obs.queue < 0.25 * q_threshold
        self.low_streak = self.low_streak + 1 if low else 0
        if self._cooldown(interval):
            return obs.replicas
        if obs.p95_ms > obs.slo_ms or obs.queue > q_threshold:
            return self._record(interval, obs.replicas, obs.replicas + 1)
        if self.low_streak >= int(self.cfg.get("baseline_low_streak_intervals", 2)):
            return self._record(interval, obs.replicas, obs.replicas - 1)
        return obs.replicas


class PIDController(BaseController):
    name = "pid"

    def decide(self, obs: Observation, interval: int) -> int:
        error = (obs.p95_ms - float(self.cfg.get("pid_target_factor", 0.75)) * obs.slo_ms) / max(obs.slo_ms, 1.0)
        clip = float(self.cfg.get("pid_integral_clip", 5.0))
        self.integral = max(-clip, min(clip, self.integral + error))
        derivative = error - self.previous_error
        self.previous_error = error
        output = (float(self.cfg.get("pid_kp", 2.4)) * error
                  + float(self.cfg.get("pid_ki", 0.18)) * self.integral
                  + float(self.cfg.get("pid_kd", 0.45)) * derivative)
        if self._cooldown(interval):
            return obs.replicas
        if output > float(self.cfg.get("pid_up_threshold", 1.0)):
            return self._record(interval, obs.replicas, obs.replicas + 1)
        if output < float(self.cfg.get("pid_down_threshold", -1.0)) and obs.utilization < 0.45:
            return self._record(interval, obs.replicas, obs.replicas - 1)
        return obs.replicas


class PredictiveController(BaseController):
    name = "predictive"

    def decide(self, obs: Observation, interval: int) -> int:
        low = obs.predicted_p99_ms < 0.60 * obs.slo_ms and obs.utilization < 0.40
        self.low_streak = self.low_streak + 1 if low else 0
        if self._cooldown(interval):
            return obs.replicas
        if obs.predicted_p99_ms > 0.95 * obs.slo_ms:
            return self._record(interval, obs.replicas, obs.replicas + 1)
        if self.low_streak >= int(self.cfg.get("baseline_low_streak_intervals", 2)):
            return self._record(interval, obs.replicas, obs.replicas - 1)
        return obs.replicas


def make_controller(name: str, cfg: dict, variant: str | None = None) -> BaseController:
    if name == "cvar_safe":
        return CVaRSafeController(cfg)
    if name == "hpa_like":
        return HPALikeController(cfg)
    if name == "keda_style":
        return KEDAStyleController(cfg)
    if name == "rule_based":
        return RuleBasedController(cfg)
    if name == "pid":
        return PIDController(cfg)
    if name == "predictive":
        return PredictiveController(cfg)
    if name == "no_cvar":
        return CVaRSafeController(cfg, use_cvar=False, use_queueing=True)
    if name == "no_queueing":
        return CVaRSafeController(cfg, use_cvar=True, use_queueing=False)
    raise ValueError(f"Unknown controller: {name}")
