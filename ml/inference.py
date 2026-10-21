"""SageMaker inference handlers for both legacy and ideal Infra Watchdog models.

Backward compatible with the existing RandomForest model.joblib, while also
supporting the ideal bundle produced by train_ideal.py:
  - point/p50/p90 latency forecasts 30 s ahead
  - probability that latency breaches the configured threshold
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
    loaded = joblib.load(os.path.join(model_dir, "model.joblib"))
    return {
        "loaded": loaded,
        "features": meta.get("features", LEGACY_FEATURES),
        "horizon_s": meta.get("horizon_s"),
        "adds_recent": meta.get("predicts") == "change_from_recent_latency",
        "threshold_ms": float(meta.get("threshold_ms", 500)),
        "ideal_bundle": isinstance(loaded, dict) and "regressor" in loaded,
    }


def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        payload = json.loads(request_body)
        return payload if isinstance(payload, list) else [payload]
    raise ValueError(f"Unsupported content type: {content_type}")


def _matrix(rows, features):
    return np.asarray([[float(row.get(f, 0.0) or 0.0) for f in features] for row in rows], dtype=float)


def predict_fn(rows, model):
    features = model["features"]
    X = _matrix(rows, features)
    loaded = model["loaded"]

    if model["ideal_bundle"]:
        recent_idx = features.index("recent_latency_ms")
        recent = X[:, recent_idx]
        delta = loaded["regressor"].predict(X)
        point = np.maximum(0.0, recent + delta)
        p50 = np.maximum(0.0, recent + loaded["q50"].predict(X)) if loaded.get("q50") is not None else point
        p90 = np.maximum(0.0, recent + loaded["q90"].predict(X)) if loaded.get("q90") is not None else point
        classifier = loaded.get("classifier")
        if classifier is not None and hasattr(classifier, "predict_proba"):
            probability = classifier.predict_proba(X)[:, 1]
        else:
            probability = (point > model["threshold_ms"]).astype(float)
        return {
            "predicted": point,
            "p50": p50,
            "p90": p90,
            "breach_probability": probability,
            "horizon_s": model["horizon_s"],
        }

    pred = loaded.predict(X)
    if model["adds_recent"]:
        pred = np.maximum(0.0, X[:, features.index("recent_latency_ms")] + pred)
    return {
        "predicted": pred,
        "p50": pred,
        "p90": pred,
        "breach_probability": None,
        "horizon_s": model["horizon_s"],
    }


def output_fn(prediction, accept="application/json"):
    body = {
        "predicted_response_time_ms": [float(p) for p in prediction["predicted"]],
        "p50_response_time_ms": [float(p) for p in prediction["p50"]],
        "p90_response_time_ms": [float(p) for p in prediction["p90"]],
        "horizon_s": prediction["horizon_s"],
    }
    if prediction["breach_probability"] is not None:
        body["breach_probability"] = [float(p) for p in prediction["breach_probability"]]
    return json.dumps(body), accept
