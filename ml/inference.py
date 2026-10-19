"""
SageMaker inference handlers, loaded by the scikit-learn inference container
when the endpoint (see deploy_endpoint.py) serves predictions.

The model forecasts latency `horizon_s` seconds ahead (see train.py). The
feature list, horizon and target form are read from model_meta.json; models
trained before it existed use the original three features, predict latency
directly, and report horizon_s = null. Extra keys in the request (the Lambda
forwards /metrics as-is) are ignored.
"""

import json
import os

import joblib
import numpy as np

LEGACY_FEATURES = ["cpu_usage", "active_connections", "lag_ms"]


def model_fn(model_dir):
    meta_path = os.path.join(model_dir, "model_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return {
        "model": joblib.load(os.path.join(model_dir, "model.joblib")),
        "features": meta.get("features", LEGACY_FEATURES),
        "horizon_s": meta.get("horizon_s"),
        "adds_recent": meta.get("predicts") == "change_from_recent_latency",
    }


def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        payload = json.loads(request_body)
        # Accept either a single reading ({"cpu_usage": .., ...}) or a batch (list of readings)
        return payload if isinstance(payload, list) else [payload]
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(rows, model):
    features = model["features"]
    X = np.array([[row[f] for f in features] for row in rows], dtype=float)
    pred = model["model"].predict(X)
    if model["adds_recent"]:  # forecast = recent latency + predicted change
        pred = np.maximum(0.0, X[:, features.index("recent_latency_ms")] + pred)
    return pred, model["horizon_s"]


def output_fn(prediction, accept="application/json"):
    values, horizon_s = prediction
    body = json.dumps({"predicted_response_time_ms": [float(p) for p in values],
                       "horizon_s": horizon_s})
    return body, accept
