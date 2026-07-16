from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "generated"
OUT = ROOT / "figures" / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Arimo", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.framealpha": 1.0,
})

LABELS = {
    "cvar_safe": "CVaR-Safe",
    "hpa_like": "HPA-like controller",
    "keda_style": "KEDA-style queue controller",
    "rule_based": "Rule-based ladder",
    "pid": "PID controller",
    "predictive": "Predictive threshold",
}


def save(fig: plt.Figure, stem: str) -> None:
    """Export the exact same artwork as embedded PNG and editable vector EPS/PDF."""
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.eps", format="eps", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, wh, text, fontsize=9):
    x, y = xy
    w, h = wh
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                       linewidth=1.1, edgecolor="black", facecolor="white")
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)
    return p


def arrow(ax, p1, p2, **kwargs):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                        linewidth=1.0, color="black", shrinkA=0, shrinkB=0, **kwargs)
    ax.add_patch(a)
    return a


# Fig. 1: closed-loop service-management view.
fig, ax = plt.subplots(figsize=(8.7, 1.75))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
boxes = [
    (0.02, 0.47, 0.18, 0.33, "Observe\nlatency, queue, utilization"),
    (0.27, 0.47, 0.18, 0.33, "Estimate\nCVaR and tail pressure"),
    (0.59, 0.47, 0.18, 0.33, "Decide\nscale up, hold, or scale down"),
    (0.82, 0.47, 0.16, 0.33, "Act\nset bounded replica count"),
]
for x,y,w,h,t in boxes: box(ax,(x,y),(w,h),t,fontsize=9)
for a,b in [((0.20,0.635),(0.27,0.635)),((0.45,0.635),(0.59,0.635)),((0.77,0.635),(0.82,0.635))]: arrow(ax,a,b)
ax.plot([0.90,0.90,0.11,0.11],[0.47,0.28,0.28,0.47],color="black",linewidth=1.0)
arrow(ax,(0.11,0.28),(0.11,0.47))
ax.text(0.50,0.10,"next decision window",ha="center",va="center",fontsize=8)
save(fig,"Fig1")

# Fig. 2: system architecture and telemetry/control paths.
fig, ax = plt.subplots(figsize=(10.2, 2.05))
ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")
box(ax,(0.01,0.30),(0.10,0.32),"Workload\ngenerator or trace",8)
box(ax,(0.17,0.30),(0.11,0.32),"Gateway and\nrequest queue",8)
box(ax,(0.33,0.65),(0.10,0.25),"m(t) service\nreplicas",8)
box(ax,(0.52,0.22),(0.10,0.30),"Telemetry\ncollector",8)
box(ax,(0.67,0.22),(0.11,0.30),"Risk and pressure\nestimator",8)
box(ax,(0.82,0.22),(0.09,0.30),"CVaR-Safe\npolicy",8)
box(ax,(0.93,0.30),(0.06,0.32),"Scaling\nactuator",8)
arrow(ax,(0.11,0.46),(0.17,0.46)); ax.text(0.14,0.50,"requests",ha="center",fontsize=7)
# Request path to service replicas.
ax.plot([0.28,0.30,0.30],[0.46,0.46,0.775],color="black",linewidth=1.0)
arrow(ax,(0.30,0.775),(0.33,0.775))
# Backlog telemetry from gateway.
arrow(ax,(0.28,0.39),(0.52,0.37)); ax.text(0.39,0.42,"backlog",ha="center",fontsize=7)
# Latency/completion telemetry from replicas.
ax.plot([0.43,0.57],[0.775,0.775],color="black",linewidth=1.0)
arrow(ax,(0.57,0.775),(0.57,0.52)); ax.text(0.50,0.81,"latency, completions",ha="center",fontsize=7)
arrow(ax,(0.62,0.37),(0.67,0.37)); arrow(ax,(0.78,0.37),(0.82,0.37)); arrow(ax,(0.91,0.37),(0.93,0.46))
# Replica-target control path from actuator to service tier.
ax.plot([0.96,0.96,0.38],[0.62,0.93,0.93],color="black",linewidth=1.0)
arrow(ax,(0.38,0.93),(0.38,0.90)); ax.text(0.70,0.96,"replica target",ha="center",fontsize=7)
save(fig,"Fig2")

# Fig. 3: decision flow.
fig, ax = plt.subplots(figsize=(7.2, 7.0))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax,(0.27,0.90),(0.46,0.075),"Read p95, p99, CVaR@95, queue, utilization, replicas",8.5)

def diamond(cx, cy, w, h, text, fs=8.5):
    pts=np.array([[cx,cy+h/2],[cx+w/2,cy],[cx,cy-h/2],[cx-w/2,cy]])
    p=Polygon(pts,closed=True,edgecolor="black",facecolor="white",linewidth=1.1)
    ax.add_patch(p); ax.text(cx,cy,text,ha="center",va="center",fontsize=fs)
    return p

def elbow(points, label=None, label_xy=None):
    for a,b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color="black",linewidth=1.0)
    arrow(ax, points[-2], points[-1])
    if label and label_xy:
        ax.text(label_xy[0],label_xy[1],label,fontsize=8,ha="center",va="center")

diamond(0.50,0.81,0.26,0.09,"Cooldown active?")
diamond(0.39,0.62,0.62,0.12,"Risk budget, p99 margin, or queue pressure breached?")
diamond(0.42,0.43,0.60,0.12,"Latency and utilization safely low for one window?")
box(ax,(0.02,0.23),(0.32,0.08),"Increase replica target within upper bound",8)
box(ax,(0.31,0.12),(0.36,0.08),"Decrease replica target within lower bound",8)
box(ax,(0.70,0.12),(0.27,0.08),"Hold current replica count",8)
box(ax,(0.32,0.025),(0.36,0.065),"Update streaks and record action time",8)
arrow(ax,(0.50,0.90),(0.50,0.855))
elbow([(0.50,0.765),(0.50,0.70),(0.50,0.68)],"no",(0.475,0.72))
elbow([(0.63,0.81),(0.80,0.81),(0.80,0.22),(0.80,0.20)],"yes",(0.82,0.48))
elbow([(0.08,0.62),(0.08,0.33),(0.08,0.31)],"yes",(0.055,0.48))
elbow([(0.39,0.56),(0.42,0.51),(0.42,0.49)],"no",(0.39,0.52))
elbow([(0.42,0.37),(0.42,0.22),(0.42,0.20)],"yes",(0.39,0.28))
elbow([(0.72,0.43),(0.86,0.43),(0.86,0.22),(0.86,0.20)],"no",(0.76,0.39))
elbow([(0.18,0.23),(0.18,0.057),(0.32,0.057)],None,None)
arrow(ax,(0.49,0.12),(0.49,0.09))
elbow([(0.835,0.12),(0.835,0.057),(0.68,0.057)],None,None)
save(fig,"Fig3")

# Fig. 4: evaluation pipeline.
fig, ax = plt.subplots(figsize=(9.2, 1.65))
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
items=[(0.01,"Inputs\nsynthetic bursts, faults,\nAlibaba profile"),(0.22,"Policies\nCVaR-Safe and\nfive baselines"),(0.43,"Execution\nshared seeds and\ncommon bounds"),(0.64,"Measurements\nlatency, violations,\nreplicas, actions"),(0.84,"Analysis\nconfidence intervals,\nfrontier, and ablation")]
for x,t in items: box(ax,(x,0.30),(0.15,0.42),t,8)
for x1,x2 in [(0.16,0.22),(0.37,0.43),(0.58,0.64),(0.79,0.84)]: arrow(ax,(x1,0.51),(x2,0.51))
save(fig,"Fig4")

# Fig. 5: empirical CDF from deterministic sampled request latencies.
lat = pd.read_csv(RESULTS / "primary_latency_samples.csv")
fig, ax = plt.subplots(figsize=(8.6, 5.0))
for policy in ["cvar_safe", "hpa_like", "keda_style", "rule_based", "pid", "predictive"]:
    vals = np.sort(lat.loc[lat.policy == policy, "latency_ms"].to_numpy())
    y = np.arange(1, len(vals) + 1) / len(vals)
    ax.plot(vals, y, linewidth=1.5, label=LABELS[policy])
ax.axvline(500.0, linestyle="--", linewidth=1.2)
ax.text(507, 0.18, "SLO = 500 ms", rotation=90, va="bottom")
ax.set_xlim(left=0); ax.set_ylim(0, 1.01)
ax.set_xlabel("End-to-end latency (ms)"); ax.set_ylabel("Empirical cumulative probability")
ax.legend(loc="lower right"); ax.grid(color="0.88", linewidth=0.6)
save(fig, "Fig5")

# Fig. 6: representative burst response.
tr = pd.read_csv(RESULTS / "primary_interval_traces.csv")
seed = int(tr.seed.min())
sub = tr[(tr.seed == seed) & (tr.policy.isin(["cvar_safe", "hpa_like"]))].copy()
start = int(sub.interval.min()); stop = min(start + 600, int(sub.interval.max()))
sub = sub[(sub.interval >= start) & (sub.interval <= stop)]
fig, ax = plt.subplots(figsize=(8.8, 4.8))
for policy in ["cvar_safe", "hpa_like"]:
    g = sub[sub.policy == policy]
    ax.step(g.interval, g.replicas_active, where="post", linewidth=1.5, label=f"{LABELS[policy]} replicas")
ax.set_xlabel("Simulation time (s)"); ax.set_ylabel("Active replicas")
ax2 = ax.twinx(); arr = sub[sub.policy == "cvar_safe"]
ax2.plot(arr.interval, arr.arrival_rate, linewidth=1.0, linestyle=":", label="Arrival rate")
ax2.set_ylabel("Arrival rate (req/s)")
lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc="upper right"); ax.grid(color="0.88", linewidth=0.6)
save(fig, "Fig6")

# Fig. 7: local resource-risk operating region and baseline points.
sweep = pd.read_csv(RESULTS / "risk_budget_sweep.csv")
agg = pd.read_csv(RESULTS / "primary_aggregate.csv")
fig, ax = plt.subplots(figsize=(8.2, 4.8))
ax.plot(sweep.replica_seconds, sweep.slo_violation_pct, marker="o", linewidth=1.5, label="CVaR-Safe budget sweep")
for _, row in sweep.iterrows():
    ax.annotate(f"{row.risk_budget_factor:.2f}", (row.replica_seconds, row.slo_violation_pct), xytext=(4, 4), textcoords="offset points", fontsize=7)
markers={"hpa_like":"s","keda_style":"^","rule_based":"D","pid":"v","predictive":"P"}
for policy in ["hpa_like", "keda_style", "rule_based", "pid", "predictive"]:
    row = agg[agg.policy == policy].iloc[0]
    ax.scatter(row.replica_seconds_mean, row.slo_violation_pct_mean, s=42, marker=markers[policy], label=LABELS[policy])
ax.set_xlabel("Capacity proxy (replica-seconds)"); ax.set_ylabel("SLO violation rate (%)")
ax.legend(loc="best"); ax.grid(color="0.88", linewidth=0.6)
save(fig, "Fig7")

# Fig. 8: pressure-score calibration; deterministic sample for legibility.
fc = pd.read_csv(RESULTS / "forecast_windows.csv")
fc = fc[(fc.policy == "cvar_safe") & fc.observed_next_p99_ms.notna()].copy()
if len(fc) > 2500:
    fc = fc.sample(2500, random_state=17)
lim = float(max(fc.predicted_p99_ms.max(), fc.observed_next_p99_ms.max()))
fig, ax = plt.subplots(figsize=(6.1, 5.6))
ax.scatter(fc.predicted_p99_ms, fc.observed_next_p99_ms, s=9, marker="o", facecolors="none", edgecolors="tab:blue", linewidths=0.45)
ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.2)
ax.set_xlabel("Pressure-based next-window p99 estimate (ms)"); ax.set_ylabel("Observed next-window p99 (ms)")
ax.grid(color="0.88", linewidth=0.6)
save(fig, "Fig8")

# Fig. 9: bottleneck sensitivity.
bot = pd.read_csv(RESULTS / "bottleneck_summary.csv")
order = ["app", "mixed", "db"]; policies = ["cvar_safe", "hpa_like", "keda_style"]
x = np.arange(len(order)); width = 0.24
fig, ax = plt.subplots(figsize=(7.8, 4.8))
for j, policy in enumerate(policies):
    vals = [float(bot[(bot.scenario == mode) & (bot.policy == policy)].slo_violation_pct.iloc[0]) for mode in order]
    ax.bar(x + (j - 1) * width, vals, width=width, label=LABELS[policy], hatch=["", "//", "xx"][j])
ax.set_xticks(x, ["Application tier", "Mixed", "Database tier"])
ax.set_xlabel("Dominant bottleneck condition"); ax.set_ylabel("SLO violation rate (%)")
ax.legend(); ax.grid(axis="y", color="0.88", linewidth=0.6)
save(fig, "Fig9")

# Fig. 10: attributed public-trace replay and representative replica trajectories.
map_df = pd.read_csv(RESULTS / "trace_mapping.csv")
rt = pd.read_csv(RESULTS / "trace_replay_interval_traces.csv")
seed = int(rt.seed.min())
fig, ax = plt.subplots(figsize=(8.6, 4.8))
ax.plot(map_df.bin_index, map_df.cpu_util_percent, linewidth=1.2, label="Alibaba 2018 CPU utilization")
ax.set_xlabel("300-second trace bin"); ax.set_ylabel("CPU utilization (%)")
ax2 = ax.twinx()
styles={"cvar_safe":"--","hpa_like":"-.","keda_style":":"}
for policy in ["cvar_safe", "hpa_like", "keda_style"]:
    g = rt[(rt.seed == seed) & (rt.policy == policy)]
    ax2.step(g.interval, g.replicas_active, where="post", linewidth=1.2, linestyle=styles[policy], label=f"{LABELS[policy]} replicas")
ax2.set_ylabel("Active replica count")
lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc="upper left"); ax.grid(color="0.88", linewidth=0.6)
save(fig, "Fig10")

print(f"Manuscript figures written to {OUT}")
