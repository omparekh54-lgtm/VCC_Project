"""Infra Watchdog - instrumented FastAPI backend.

Enhancements for the Ideal Design version:
- fleet/model features include trends, request arrival rate and recent p95;
- request logging is moved off the request path into an async queue;
- local CSV logs rotate instead of growing forever;
- optional per-instance CloudWatch metric publishing every 10 s;
- /prediction supports ideal model bundles with p50/p90 and breach probability;
- legacy models and the existing dashboard/API remain compatible.
"""

import asyncio
import collections
import csv
import json
import os
import socket
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import psutil
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse


class ServerState:
    def __init__(self):
        self.active_connections = 0
        self.lag_ms = 0.0
        self.cpu_usage = 0.0
        self.recent = collections.deque()      # (finish_ts, latency_ms)
        self.recent_sum = 0.0
        self.arrivals = collections.deque()    # request arrival timestamps
        self.samples = collections.deque()     # (ts, cpu, conns, lag)
        self.lock = asyncio.Lock()

    def add_latency(self, ms: float) -> None:
        self.recent.append((time.time(), ms))
        self.recent_sum += ms

    def add_arrival(self, ts: float) -> None:
        self.arrivals.append(ts)

    def recent_latency(self):
        cutoff = time.time() - RECENT_WINDOW_S
        while self.recent and self.recent[0][0] < cutoff:
            self.recent_sum -= self.recent.popleft()[1]
        n = len(self.recent)
        return (self.recent_sum / n if n else None), n

    def recent_p95(self):
        self.recent_latency()  # prune first
        if not self.recent:
            return 0.0
        vals = sorted(v for _, v in self.recent)
        idx = min(len(vals) - 1, int(0.95 * len(vals)))
        return float(vals[idx])

    def request_rate(self):
        now = time.time()
        cutoff = now - RECENT_WINDOW_S
        while self.arrivals and self.arrivals[0] < cutoff:
            self.arrivals.popleft()
        return len(self.arrivals) / RECENT_WINDOW_S

    def add_sample(self):
        now = time.time()
        self.samples.append((now, self.cpu_usage, float(self.active_connections), self.lag_ms))
        cutoff = now - 90
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def _past(self, seconds):
        if not self.samples:
            return (self.cpu_usage, float(self.active_connections), self.lag_ms)
        target = time.time() - seconds
        candidate = self.samples[0]
        for sample in self.samples:
            if sample[0] <= target:
                candidate = sample
            else:
                break
        return candidate[1], candidate[2], candidate[3]


state = ServerState()
RECENT_WINDOW_S = 10.0
CPU_WORKERS = int(os.getenv("CPU_WORKERS", "1"))
LOG_PATH = Path(os.getenv("METRICS_LOG_PATH", "metrics_log.csv"))
LOG_MAX_BYTES = int(os.getenv("METRICS_LOG_MAX_BYTES", str(25 * 1024 * 1024)))
LOG_BACKUPS = int(os.getenv("METRICS_LOG_BACKUPS", "5"))
LOG_FIELDS = ["timestamp", "cpu_usage", "active_connections", "lag_ms", "response_time_ms", "instance_count"]
INSTANCE_COUNT = float(os.getenv("INSTANCE_COUNT", "1"))
INSTANCE_ID = os.getenv("INSTANCE_ID", socket.gethostname())
PUSH_CLOUDWATCH = os.getenv("PUSH_CLOUDWATCH", "false").lower() in {"1", "true", "yes"}
CLOUDWATCH_INTERVAL_S = float(os.getenv("CLOUDWATCH_INTERVAL_S", "10"))


def _ensure_log_file():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with LOG_PATH.open("w", newline="") as f:
            csv.writer(f).writerow(LOG_FIELDS)


def _rotate_log_if_needed():
    try:
        if LOG_PATH.stat().st_size < LOG_MAX_BYTES:
            return
    except FileNotFoundError:
        return
    for i in range(LOG_BACKUPS - 1, 0, -1):
        src = LOG_PATH.with_suffix(LOG_PATH.suffix + f".{i}")
        dst = LOG_PATH.with_suffix(LOG_PATH.suffix + f".{i + 1}")
        if src.exists():
            src.replace(dst)
    LOG_PATH.replace(LOG_PATH.with_suffix(LOG_PATH.suffix + ".1"))
    _ensure_log_file()


def _write_rows(rows):
    _rotate_log_if_needed()
    with LOG_PATH.open("a", newline="") as f:
        csv.writer(f).writerows(rows)


async def _log_writer(queue: asyncio.Queue):
    pending = []
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=0.5)
            pending.append(item)
            queue.task_done()
        except asyncio.TimeoutError:
            pass
        if pending:
            rows, pending = pending, []
            await asyncio.to_thread(_write_rows, rows)


async def _event_loop_lag_monitor(interval_s=0.25):
    loop = asyncio.get_event_loop()
    while True:
        start = loop.time()
        await asyncio.sleep(interval_s)
        drift_ms = max(0.0, (loop.time() - start - interval_s) * 1000)
        state.lag_ms = 0.7 * state.lag_ms + 0.3 * drift_ms


async def _cpu_sampler(interval_s=1.0):
    psutil.cpu_percent(interval=None)
    while True:
        await asyncio.sleep(interval_s)
        state.cpu_usage = psutil.cpu_percent(interval=None)
        state.add_sample()


def _current_features():
    recent_ms, _ = state.recent_latency()
    cpu15, conn15, lag15 = state._past(15)
    cpu30, conn30, lag30 = state._past(30)
    return {
        "cpu_usage": round(state.cpu_usage, 3),
        "active_connections": state.active_connections,
        "lag_ms": round(state.lag_ms, 3),
        "recent_latency_ms": round(recent_ms or 0.0, 3),
        "cpu_delta_15s": round(state.cpu_usage - cpu15, 3),
        "cpu_delta_30s": round(state.cpu_usage - cpu30, 3),
        "connections_delta_15s": round(state.active_connections - conn15, 3),
        "connections_delta_30s": round(state.active_connections - conn30, 3),
        "lag_delta_15s": round(state.lag_ms - lag15, 3),
        "lag_delta_30s": round(state.lag_ms - lag30, 3),
        "request_rate_rps": round(state.request_rate(), 3),
        "recent_p95_latency_ms": round(state.recent_p95(), 3),
        "instance_count": INSTANCE_COUNT,
    }


async def _cloudwatch_publisher():
    if not PUSH_CLOUDWATCH:
        return
    import boto3
    cw = boto3.client("cloudwatch")
    while True:
        await asyncio.sleep(CLOUDWATCH_INTERVAL_S)
        f = _current_features()
        data = [
            {"MetricName": "CPUUsage", "Dimensions": [{"Name": "InstanceId", "Value": INSTANCE_ID}], "Value": f["cpu_usage"], "Unit": "Percent"},
            {"MetricName": "ActiveConnections", "Dimensions": [{"Name": "InstanceId", "Value": INSTANCE_ID}], "Value": f["active_connections"], "Unit": "Count"},
            {"MetricName": "EventLoopLagMs", "Dimensions": [{"Name": "InstanceId", "Value": INSTANCE_ID}], "Value": f["lag_ms"], "Unit": "Milliseconds"},
            {"MetricName": "RecentLatencyMs", "Dimensions": [{"Name": "InstanceId", "Value": INSTANCE_ID}], "Value": f["recent_latency_ms"], "Unit": "Milliseconds"},
            {"MetricName": "RecentP95LatencyMs", "Dimensions": [{"Name": "InstanceId", "Value": INSTANCE_ID}], "Value": f["recent_p95_latency_ms"], "Unit": "Milliseconds"},
            {"MetricName": "RequestRateRps", "Dimensions": [{"Name": "InstanceId", "Value": INSTANCE_ID}], "Value": f["request_rate_rps"], "Unit": "Count/Second"},
        ]
        try:
            await asyncio.to_thread(cw.put_metric_data, Namespace="InfraWatchdog/App", MetricData=data)
        except Exception as exc:
            print(f"cloudwatch publish failed: {type(exc).__name__}: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_log_file()
    app.state.cpu_limiter = anyio.CapacityLimiter(CPU_WORKERS)
    app.state.log_queue = asyncio.Queue(maxsize=10000)
    tasks = [
        asyncio.create_task(_event_loop_lag_monitor()),
        asyncio.create_task(_cpu_sampler()),
        asyncio.create_task(_log_writer(app.state.log_queue)),
    ]
    if PUSH_CLOUDWATCH:
        tasks.append(asyncio.create_task(_cloudwatch_publisher()))
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="Infra Watchdog Demo Backend", lifespan=lifespan)


@app.middleware("http")
async def instrument_requests(request: Request, call_next):
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    features = _current_features()
    arrived = time.time()
    state.add_arrival(arrived)
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
        row = [arrived, features["cpu_usage"], features["active_connections"], features["lag_ms"], round(elapsed_ms, 3), INSTANCE_COUNT]
        try:
            request.app.state.log_queue.put_nowait(row)
        except asyncio.QueueFull:
            print("metrics log queue full; dropping one training row")
    return response


DASHBOARD_PATH = Path(__file__).with_name("dashboard.html")


@app.get("/", include_in_schema=False)
async def dashboard():
    if not DASHBOARD_PATH.exists():
        return RedirectResponse("/docs")
    return FileResponse(DASHBOARD_PATH)


@app.get("/health")
async def health():
    return {"status": "ok", "instance_id": INSTANCE_ID}


@app.get("/metrics")
async def metrics():
    return _current_features()


MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).with_name("model") / "model.joblib")))
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "")
LATENCY_THRESHOLD_MS = float(os.getenv("LATENCY_THRESHOLD_MS", "500"))
LEGACY_FEATURES = ["cpu_usage", "active_connections", "lag_ms"]


class _Predictor:
    def __init__(self):
        self._lock = threading.Lock()
        self._mtime = None
        self._loaded = None
        self._meta = {}
        self._sagemaker = None

    def _local_model(self):
        meta_path = MODEL_PATH.with_name("model_meta.json")
        try:
            mtime = MODEL_PATH.stat().st_mtime
        except FileNotFoundError:
            return None
        version = (mtime, meta_path.stat().st_mtime if meta_path.exists() else None)
        with self._lock:
            if version != self._mtime:
                import joblib
                loaded = joblib.load(MODEL_PATH)
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                # Legacy RandomForest supports n_jobs; ideal bundles contain several models.
                if not isinstance(loaded, dict) and hasattr(loaded, "n_jobs"):
                    loaded.n_jobs = 1
                self._loaded, self._meta, self._mtime = loaded, meta, version
            return self._loaded

    def predict(self, features):
        loaded = self._local_model()
        if loaded is not None:
            names = self._meta.get("features", LEGACY_FEATURES)
            row = [float(features.get(f, 0.0) or 0.0) for f in names]
            if isinstance(loaded, dict) and "regressor" in loaded:
                import numpy as np
                X = np.asarray([row], dtype=float)
                recent = float(features.get("recent_latency_ms", 0.0))
                point = max(0.0, recent + float(loaded["regressor"].predict(X)[0]))
                p50 = max(0.0, recent + float(loaded["q50"].predict(X)[0])) if loaded.get("q50") is not None else point
                p90 = max(0.0, recent + float(loaded["q90"].predict(X)[0])) if loaded.get("q90") is not None else point
                classifier = loaded.get("classifier")
                probability = float(classifier.predict_proba(X)[0, 1]) if classifier is not None else float(point > LATENCY_THRESHOLD_MS)
                return point, self._meta.get("horizon_s"), "local ideal model", p50, p90, probability
            predicted = float(loaded.predict([row])[0])
            if self._meta.get("predicts") == "change_from_recent_latency":
                predicted = max(0.0, features["recent_latency_ms"] + predicted)
            return predicted, self._meta.get("horizon_s"), "local model", predicted, predicted, None

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
            predicted = float(result["predicted_response_time_ms"][0])
            p50 = float(result.get("p50_response_time_ms", [predicted])[0])
            p90 = float(result.get("p90_response_time_ms", [predicted])[0])
            probs = result.get("breach_probability")
            probability = float(probs[0]) if probs else None
            return predicted, result.get("horizon_s"), "SageMaker endpoint", p50, p90, probability
        raise LookupError("no model: train one or set SAGEMAKER_ENDPOINT_NAME")


predictor = _Predictor()


@app.get("/prediction")
async def prediction():
    features = _current_features()
    recent_ms, recent_n = state.recent_latency()
    body = {
        "available": False,
        "source": None,
        "predicted_ms": None,
        "p50_ms": None,
        "p90_ms": None,
        "breach_probability": None,
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
        predicted, horizon_s, source, p50, p90, probability = await run_in_threadpool(predictor.predict, features)
    except LookupError as exc:
        body["error"] = str(exc)
    except Exception as exc:
        body["error"] = f"{type(exc).__name__}: {exc}"[:300]
    else:
        has_traffic = features["active_connections"] > 0 or features["recent_latency_ms"] > 0 or features["request_rate_rps"] > 0
        decision = probability >= 0.70 if probability is not None else predicted > LATENCY_THRESHOLD_MS
        body.update(
            available=True,
            source=source,
            predicted_ms=predicted,
            p50_ms=p50,
            p90_ms=p90,
            breach_probability=probability,
            horizon_s=horizon_s,
            would_scale_out=has_traffic and decision,
            scaling_suppressed_reason=None if has_traffic else "no recent business traffic",
        )
    return body


def _burn_cpu(intensity: int) -> int:
    total = 0
    for i in range(intensity * 200_000):
        total += i * i
    return total % 100_000


@app.get("/api/simulate")
async def simulate(request: Request, intensity: int = 1):
    intensity = max(1, min(intensity, 10))
    result = await anyio.to_thread.run_sync(_burn_cpu, intensity, limiter=request.app.state.cpu_limiter)
    return JSONResponse({"result": result})
