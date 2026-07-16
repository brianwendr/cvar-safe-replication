from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path

from _common import ROOT


async def wait_until_ready(url: str, timeout: float = 20.0) -> None:
    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("Install dependencies: pip install -r requirements.txt") from exc
    started = time.perf_counter()
    async with httpx.AsyncClient() as client:
        while time.perf_counter() - started < timeout:
            try:
                response = await client.get(url, timeout=1.0)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError("Local API did not become ready")


async def run(args: argparse.Namespace) -> None:
    command = [
        sys.executable, "-m", "uvicorn", "cvar_safe.api:app",
        "--host", "127.0.0.1", "--port", str(args.port),
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    try:
        await wait_until_ready(f"http://127.0.0.1:{args.port}/health")
        load_cmd = [
            sys.executable, str(ROOT / "deployment/loadgen.py"),
            "--url", f"http://127.0.0.1:{args.port}/work",
            "--duration", str(args.duration), "--rate", str(args.rate),
            "--workers", str(args.workers), "--output", args.output,
        ]
        subprocess.run(load_cmd, cwd=ROOT, check=True, env=env)
        collect_cmd = [
            sys.executable, str(ROOT / "deployment/collect_metrics.py"),
            "--latency-csv", args.output, "--slo-ms", str(args.slo_ms),
            "--output", args.summary,
        ]
        subprocess.run(collect_cmd, cwd=ROOT, check=True, env=env)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local process-level consistency check")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=25.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--slo-ms", type=float, default=500.0)
    parser.add_argument("--output", default="results/local_pilot_latency.csv")
    parser.add_argument("--summary", default="results/local_pilot_summary.csv")
    asyncio.run(run(parser.parse_args()))
