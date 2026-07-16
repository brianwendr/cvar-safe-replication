# CVaR-Safe essential replication package

This repository contains the files needed to reproduce and verify the manuscript's emulator, sensitivity, trace-replay, statistical, and tabulated results. It also retains the small deployment examples described in the manuscript. These deployment files document an implementation path; they are not measured Kubernetes-cluster evidence.

Artwork and figure-generation scripts are intentionally excluded. Manuscript figures are supplied separately by the author and are not required to reproduce the numerical results reported in Tables 4-12 or the Section 5.5 mechanism contrasts.

## Requirements

- Python 3.10 or later
- Approximately 40 MB of disk space for the repository
- Additional temporary space when rerunning all experiments

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the manuscript results

Run the complete numerical analysis:

```bash
python scripts/run_all.py --config configs/primary.yaml
```

This executes the primary shared-seed experiment, sensitivity and ablation analyses, public-trace replay, and table-alignment check.

Verify the included canonical outputs without rerunning the simulations:

```bash
python scripts/check_reported_reference.py
PYTHONPATH=src pytest -q
```

## Manuscript-output map

- Tables 4-5: `results/generated/primary_aggregate.csv`
- Table 6: `results/generated/risk_budget_sweep.csv`
- Table 7: `results/generated/forecast_aggregate.csv`
- Table 8: `results/generated/robustness_summary.csv`
- Table 9: `results/generated/bottleneck_summary.csv`
- Table 10: `results/generated/ablation_inference.csv`
- Table 11: `results/generated/ablation_secondary_summary.csv`
- Section 5.5 contrasts: `results/generated/mechanism_contrasts.csv`
- Table 12: `results/generated/trace_replay_aggregate.csv`
- Display-precision table mirrors: `reported_reference/`

The seed-level, paired-test, interval-trace, latency-sample, forecast-window, CPU-sensitivity, and trace-mapping CSV files are retained as the underlying numerical evidence for the reported analyses.

## Main folders

- `src/cvar_safe/`: emulator, controllers, metrics, statistics, workload, and trace adapter
- `configs/`: fixed primary and trace-replay configurations
- `scripts/`: experiment and alignment scripts
- `results/generated/`: canonical numerical outputs used by the manuscript
- `reported_reference/`: manuscript tables at display precision
- `data/trace/`: attributed public-trace segment and provenance
- `tests/`: minimal validation tests
- `deployment/`: FastAPI, Docker, Kubernetes, KEDA, and kind examples described in the manuscript

## Local implementation check

```bash
python scripts/run_local_pilot.py --duration 30 --rate 25
```

## GitHub upload

```bash
git init
git add .
git commit -m "Add essential replication package"
git branch -M main
git remote add origin <repository-url>
git push -u origin main
```
