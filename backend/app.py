"""
Infra Watchdog - instrumented FastAPI backend.

Exposes:
  - GET /                           browser dashboard (dashboard.html next to this file)
  - GET /api/simulate?intensity=N   dummy business endpoint for Locust to hammer
  - GET /metrics                    current {cpu_usage, active_connections, lag_ms,
                                     recent_latency_ms}, polled by the Lambda
                                     watchdog every ~30s
  - GET /prediction                 the model's latency forecast for the current
                                     readings, for the dashboard (see _Predictor)
  - GET /health                     liveness check

A request-logging middleware appends (features, response_time_ms) to a local
CSV on every request. That CSV is the training data for the SageMaker model
-- run scripts/upload_metrics_to_s3.py after a load test to ship it to S3.
"""

import asyncio
import collections
import csv
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import psutil
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

# ---------------------------------------------------------------------------
# Process-wide state (single uvicorn worker -- see README "Known simplifications")
# ---------------------------------------------------------------------------


class ServerState:
    def __init__(self):
        self.active_connections = 0  # in-flight /api/ requests only
        self.lag_ms = 0.0
        self.cpu_usage = 0.0
        # (finish time, latency ms) of /api/ requests in the last RECENT_WINDOW_S,
        # with a running sum; only touched from the event loop.
        self.recent = collections.deque()
        self.recent_sum = 0.0
        self.lock = asyncio.Lock()

    def add_latency(self, ms: float) -> None:
        self.recent.append((time.time(), ms))
        self.recent_sum += ms

    def recent_latency(self):
        """(mean latency of requests that finished in the last RECENT_WINDOW_S, count)."""
        cutoff = time.time() - RECENT_WINDOW_S
        while self.recent and self.recent[0][0] < cutoff:
            self.recent_sum -= self.recent.popleft()[1]
        n = len(self.recent)
        return (self.recent_sum / n if n else None), n


state = ServerState()

RECENT_WINDOW_S = 10.0  # must match RECENT_WINDOW_S in ml/train.py
# Threads that run /api/simulate's CPU work at once. Python's GIL lets only one
# compute at a time anyway; more threads would just starve the event loop of
# the GIL so it stops accepting requests (see simulate()).
CPU_WORKERS = int(os.getenv("CPU_WORKERS", "1"))
LOG_PATH = Path(os.getenv("METRICS_LOG_PATH", "metrics_log.csv"))
LOG_FIELDS = ["timestamp", "cpu_usage", "active_connections", "lag_ms", "response_time_ms"]


def _ensure_log_file() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
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


async def _cpu_sampler(interval_s: float = 1.0) -> None:
    """
    The only caller of psutil.cpu_percent(). It reports usage since its previous
    call, so letting both the request logger and /metrics call it would give
    each a different (and for per-request calls, millisecond-wide) window.
    One fixed ~1s sample shared by both keeps training and serving features
    on the same definition. If the loop is blocked, the next sample simply
    covers the whole blocked stretch.
    """
    psutil.cpu_percent(interval=None)  # prime psutil's internal sample window
    while True:
        await asyncio.sleep(interval_s)
        state.cpu_usage = psutil.cpu_percent(interval=None)


def _current_features() -> dict:
    """
    The model's input, read identically by the request logger and /metrics.
    recent_latency_ms isn't written to the CSV: ml/train.py rebuilds it from
    each row's arrival time + latency, with the same 10 s definition.
    """
    recent_ms, _ = state.recent_latency()
    return {
        "cpu_usage": state.cpu_usage,
        "active_connections": state.active_connections,
        "lag_ms": round(state.lag_ms, 3),
        "recent_latency_ms": round(recent_ms or 0.0, 3),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_log_file()
    app.state.cpu_limiter = anyio.CapacityLimiter(CPU_WORKERS)
    tasks = [asyncio.create_task(_event_loop_lag_monitor()), asyncio.create_task(_cpu_sampler())]
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="Infra Watchdog Demo Backend", lifespan=lifespan)


@app.middleware("http")
async def instrument_requests(request: Request, call_next):
    # Only business traffic is load and training data -- not /metrics polls,
    # the dashboard page, /docs, or stray scanner hits on the public port.
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    # Snapshot the features as the request arrives, before counting it: that is
    # the state the watchdog sees when it polls /metrics, and it can't be
    # influenced by this request's own work (which would leak the target).
    features = _current_features()
    arrived = time.time()
    async with state.lock:
        state.active_connections += 1

    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        async with state.lock:
            state.active_connections -= 1
        state.add_latency(elapsed_ms)

        # Stamped with the arrival time, the same moment the features describe;
        # train.py builds its "latency H seconds later" target from these times.
        with LOG_PATH.open("a", newline="") as f:
            csv.writer(f).writerow(
                [arrived] + [features[k] for k in LOG_FIELDS[1:4]] + [round(elapsed_ms, 3)]
            )
    return response


DASHBOARD_PATH = Path(__file__).with_name("dashboard.html")


@app.get("/", include_in_schema=False)
async def dashboard():
    """Browser console: live readings from /metrics plus a small load generator."""
    if not DASHBOARD_PATH.exists():
        return RedirectResponse("/docs")
    return FileResponse(DASHBOARD_PATH)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    """Polled by the Lambda watchdog. Field names match the model's features exactly."""
    return _current_features()


# ---------------------------------------------------------------------------
# Latency forecast for the dashboard
# ---------------------------------------------------------------------------

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).with_name("model") / "model.joblib")))
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "")
LATENCY_THRESHOLD_MS = float(os.getenv("LATENCY_THRESHOLD_MS", "500"))
LEGACY_FEATURES = ["cpu_usage", "active_connections", "lag_ms"]


class _Predictor:
    """
    Gives the dashboard the same forecast the Lambda guardrail acts on.

    Uses a local model.joblib if one exists (scripts/train_on_server.sh leaves
    one at /opt/infra-watchdog/model/), reloading it when the file changes;
    otherwise calls the SageMaker endpoint named by SAGEMAKER_ENDPOINT_NAME.
    Both are optional: with neither, /prediction reports available=false and
    the dashboard says so.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._mtime = None
        self._model = None
        self._meta = {}
        self._sagemaker = None

    def _local_model(self):
        meta_path = MODEL_PATH.with_name("model_meta.json")
        try:
            mtime = MODEL_PATH.stat().st_mtime
        except FileNotFoundError:
            return None
        # Keyed on both files: train.py writes model_meta.json just after
        # model.joblib, and a load in between must not pin stale metadata.
        version = (mtime, meta_path.stat().st_mtime if meta_path.exists() else None)
        with self._lock:
            if version != self._mtime:
                import joblib  # only needed when a local model is present

                model = joblib.load(MODEL_PATH)
                model.n_jobs = 1  # one row per call; worker processes only add overhead
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                self._model, self._meta, self._mtime = model, meta, version
            return self._model

    def predict(self, features: dict):
        """Returns (predicted_ms, horizon_s, source). Same arithmetic as ml/inference.py."""
        model = self._local_model()
        if model is not None:
            names = self._meta.get("features", LEGACY_FEATURES)
            row = [float(features[f]) for f in names]
            predicted = float(model.predict([row])[0])
            if self._meta.get("predicts") == "change_from_recent_latency":
                predicted = max(0.0, features["recent_latency_ms"] + predicted)
            return predicted, self._meta.get("horizon_s"), "local model"
        if SAGEMAKER_ENDPOINT_NAME:
            if self._sagemaker is None:
                import boto3

                self._sagemaker = boto3.client("sagemaker-runtime")
            response = self._sagemaker.invoke_endpoint(
                EndpointName=SAGEMAKER_ENDPOINT_NAME,
                ContentType="application/json",
                Body=json.dumps(features),
            )
            result = json.loads(response["Body"].read().decode())
            return result["predicted_response_time_ms"][0], result.get("horizon_s"), "SageMaker endpoint"
        raise LookupError("no model: train one or set SAGEMAKER_ENDPOINT_NAME")


predictor = _Predictor()


@app.get("/prediction")
async def prediction():
    """
    The model's forecast for the current readings, next to the latency actually
    measured over the last 10 s. Readings are taken on the event loop; model
    inference or the SageMaker call runs in a worker thread so it never blocks it.
    """
    features = _current_features()
    recent_ms, recent_n = state.recent_latency()
    body = {
        "available": False,
        "source": None,
        "predicted_ms": None,
        "horizon_s": None,
        "threshold_ms": LATENCY_THRESHOLD_MS,
        "would_scale_out": None,
        "scaling_suppressed_reason": None,
        "features": features,
        "recent_latency_ms": recent_ms,
        "recent_requests": recent_n,
        "error": None,
    }
    try:
        predicted, horizon_s, source = await run_in_threadpool(predictor.predict, features)
    except LookupError as exc:
        body["error"] = str(exc)
    except Exception as exc:  # model file unreadable, endpoint deleted, no permission...
        body["error"] = f"{type(exc).__name__}: {exc}"[:300]
    else:
        has_traffic = features["active_connections"] > 0 or features["recent_latency_ms"] > 0
        body.update(available=True, source=source, predicted_ms=predicted, horizon_s=horizon_s,
                    would_scale_out=has_traffic and predicted > LATENCY_THRESHOLD_MS,
                    scaling_suppressed_reason=None if has_traffic else "no recent business traffic")
    return body


def _burn_cpu(intensity: int) -> int:
    total = 0
    for i in range(intensity * 200_000):
        total += i * i
    return total % 100_000


@app.get("/api/simulate")
async def simulate(request: Request, intensity: int = 1):
    """
    Dummy business endpoint for Locust to hit. `intensity` (1-10) controls how
    much CPU-bound work happens per request.

    The work runs on CPU_WORKERS worker thread(s); requests beyond that wait
    their turn inside the app, like a server with a fixed worker pool. The
    event loop stays free to accept every request as it arrives, which is what
    makes overload measurable: queued requests are counted in
    active_connections, and their waiting time is part of response_time_ms --
    the latency users actually get.

    Running the work directly on the event loop (the original design) hid all
    of that: waiting requests sat in the OS socket queue, invisible to the app.
    At 16 clients users waited ~3.5 s while the app recorded ~0.4 s and never
    more than one active connection. One thread per request (a plain `def`
    handler) only halved the gap, because 16 threads fighting for the GIL
    starve the loop that accepts requests.
    """
    intensity = max(1, min(intensity, 10))
    result = await anyio.to_thread.run_sync(_burn_cpu, intensity, limiter=request.app.state.cpu_limiter)
    return JSONResponse({"result": result})
