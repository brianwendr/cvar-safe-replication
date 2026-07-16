from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from _common import ROOT
from run_primary import run as run_primary
from run_sensitivity import run as run_sensitivity
from run_trace_replay import run as run_trace_replay


def run(config_path: str) -> None:
    run_primary(config_path)
    run_sensitivity(config_path)
    run_trace_replay("configs/trace_replay.yaml")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_reported_reference.py")],
        check=True,
    )
    print("All manuscript analyses and table-alignment checks completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/primary.yaml")
    args = parser.parse_args()
    run(args.config)
