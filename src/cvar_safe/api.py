from __future__ import annotations

import asyncio
import math
import os
import random
import time

try:
    from fastapi import FastAPI
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install deployment extras: pip install .[deployment]") from exc

app = FastAPI(title="Latency-Critical Demo Service", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/work")
async def work(complexity: float = 1.0, slow_probability: float = 0.03) -> dict:
    complexity = max(0.1, min(complexity, 10.0))
    base_ms = float(os.getenv("BASE_SERVICE_MS", "35")) * complexity
    sigma = float(os.getenv("SERVICE_SIGMA", "0.6"))
    delay_ms = random.lognormvariate(math.log(base_ms) - 0.5 * sigma * sigma, sigma)
    if random.random() < slow_probability:
        delay_ms += random.uniform(120.0, 450.0)
    started = time.perf_counter()
    await asyncio.sleep(delay_ms / 1000.0)
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    return {"elapsed_ms": elapsed_ms, "requested_complexity": complexity}
