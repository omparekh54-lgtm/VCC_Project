"""
SageMaker inference handlers, loaded by the scikit-learn inference container
when the endpoint (see deploy_endpoint.py) serves predictions.
"""

import json
import os

import joblib
import numpy as np

FEATURES = ["cpu_usage", "active_connections", "lag_ms"]


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        payload = json.loads(request_body)
        # Accept either a single reading ({"cpu_usage": .., ...}) or a batch (list of readings)
        rows = payload if isinstance(payload, list) else [payload]
        return np.array([[row[f] for f in FEATURES] for row in rows])
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, accept="application/json"):
    body = json.dumps({"predicted_response_time_ms": [float(p) for p in prediction]})
    return body, accept
