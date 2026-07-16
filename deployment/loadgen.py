from __future__ import annotations

import argparse
import asyncio
import csv
import random
import time
from pathlib import Path

try:
    import httpx
except ImportError as exc:
    raise SystemExit("Install deployment extras: pip install .[deployment]") from exc


async def worker(client: httpx.AsyncClient, url: str, rows: list[dict], stop_at: float, rate: float) -> None:
    while time.perf_counter() < stop_at:
        started = time.perf_counter()
        status = 0
        try:
            response = await client.get(url, timeout=10.0)
            status = response.status_code
        except Exception:
            status = -1
        elapsed = 1000.0 * (time.perf_counter() - started)
        rows.append({"timestamp": time.time(), "latency_ms": elapsed, "status": status})
        await asyncio.sleep(random.expovariate(max(rate, 0.01)))


async def main_async(args: argparse.Namespace) -> None:
    rows: list[dict] = []
    stop_at = time.perf_counter() + args.duration
    async with httpx.AsyncClient() as client:
        tasks = [worker(client, args.url, rows, stop_at, args.rate / args.workers) for _ in range(args.workers)]
        await asyncio.gather(*tasks)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "latency_ms", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} requests to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/work")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", default="results/local_latency.csv")
    asyncio.run(main_async(parser.parse_args()))
