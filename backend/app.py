"""
Infra Watchdog - instrumented FastAPI backend.

Exposes:
  - GET /api/simulate?intensity=N   dummy business endpoint for Locust to hammer
  - GET /metrics                    current {cpu_usage, active_connections, lag_ms},
                                     polled by the Lambda watchdog every ~30s
  - GET /health                     liveness check

A request-logging middleware appends (features, response_time_ms) to a local
CSV on every request. That CSV is the training data for the SageMaker model
-- run scripts/upload_metrics_to_s3.py after a load test to ship it to S3.
"""

import asyncio
import csv
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Process-wide state (single uvicorn worker -- see README "Known simplifications")
# ---------------------------------------------------------------------------


class ServerState:
    def __init__(self):
        self.active_connections = 0
        self.lag_ms = 0.0
        self.lock = asyncio.Lock()


state = ServerState()

LOG_PATH = Path(os.getenv("METRICS_LOG_PATH", "metrics_log.csv"))
LOG_FIELDS = ["timestamp", "cpu_usage", "active_connections", "lag_ms", "response_time_ms"]


def _ensure_log_file() -> None:
    if not LOG_PATH.exists():
        with LOG_PATH.open("w", newline="") as f:
            csv.writer(f).writerow(LOG_FIELDS)


async def _event_loop_lag_monitor(interval_s: float = 0.25) -> None:
    """
    Measures event-loop lag: schedule a wakeup `interval_s` seconds from now,
    then compare the actual delay to the requested one. Any extra delay is
    time the loop spent busy on something else (CPU-bound handlers, GC,
    blocking I/O) instead of scheduling this coroutine on time -- that's
    exactly the "backend compute strain" signal we want as a feature.
    """
    loop = asyncio.get_event_loop()
    while True:
        start = loop.time()
        await asyncio.sleep(interval_s)
        actual = loop.time() - start
        drift_ms = max(0.0, (actual - interval_s) * 1000)
        # EMA smoothing so a single spike doesn't dominate the reading
        state.lag_ms = 0.7 * state.lag_ms + 0.3 * drift_ms


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_log_file()
    psutil.cpu_percent(interval=None)  # prime psutil's internal sample window
    task = asyncio.create_task(_event_loop_lag_monitor())
    yield
    task.cancel()


app = FastAPI(title="Infra Watchdog Demo Backend", lifespan=lifespan)


@app.middleware("http")
async def instrument_requests(request: Request, call_next):
    async with state.lock:
        state.active_connections += 1

    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        async with state.lock:
            state.active_connections -= 1

        if request.url.path not in ("/metrics", "/health"):
            cpu = psutil.cpu_percent(interval=None)
            with LOG_PATH.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        time.time(),
                        cpu,
                        state.active_connections,
                        round(state.lag_ms, 3),
                        round(elapsed_ms, 3),
                    ]
                )
    return response


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    """Polled by the Lambda watchdog. Field names match the model's features exactly."""
    return {
        "cpu_usage": psutil.cpu_percent(interval=None),
        "active_connections": state.active_connections,
        "lag_ms": round(state.lag_ms, 3),
    }


@app.get("/api/simulate")
async def simulate(intensity: int = 1):
    """
    Dummy business endpoint for Locust to hit. `intensity` (1-10) controls how
    much CPU-bound work happens per request. This is deliberately synchronous,
    blocking work inside an async handler -- under concurrent load it starves
    the event loop, which is exactly the degradation pattern the whole project
    is trying to predict ahead of time.
    """
    intensity = max(1, min(intensity, 10))
    total = 0
    for i in range(intensity * 200_000):
        total += i * i
    return JSONResponse({"result": total % 100_000})
