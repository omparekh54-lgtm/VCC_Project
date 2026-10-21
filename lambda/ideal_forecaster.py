"""Ideal-design forecaster Lambda.

Unlike the demo guardrail, this function does not poll a public EC2 IP and it
does not directly change ASG desired capacity. Instances publish per-instance
metrics to CloudWatch; this function aggregates those metrics, invokes the ML
endpoint, and publishes PredictedLatencyMs / BreachProbability. Native EC2
Auto Scaling policies can then act on those CloudWatch metrics.

If inference fails, it publishes a conservative rule-based fallback decision
from current latency / CPU so scaling is never blind.
"""

from datetime import datetime, timedelta, timezone
import json
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError

ENDPOINT = os.environ["SAGEMAKER_ENDPOINT_NAME"]
LATENCY_THRESHOLD_MS = float(os.environ.get("LATENCY_THRESHOLD_MS", "500"))
CPU_THRESHOLD = float(os.environ.get("FALLBACK_CPU_THRESHOLD", "70"))
LOOKBACK_S = int(os.environ.get("LOOKBACK_S", "90"))

cw = boto3.client("cloudwatch")
sm = boto3.client("sagemaker-runtime")

METRICS = {
    "CPUUsage": ("cpu_usage", "max"),
    "ActiveConnections": ("active_connections", "sum"),
    "EventLoopLagMs": ("lag_ms", "max"),
    "RecentLatencyMs": ("recent_latency_ms", "max"),
    "RecentP95LatencyMs": ("recent_p95_latency_ms", "max"),
    "RequestRateRps": ("request_rate_rps", "sum"),
}


def latest_per_instance(metric_name):
    listed = cw.list_metrics(Namespace="InfraWatchdog/App", MetricName=metric_name).get("Metrics", [])
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=LOOKBACK_S)
    out = []
    for m in listed:
        stats = cw.get_metric_statistics(
            Namespace="InfraWatchdog/App",
            MetricName=metric_name,
            Dimensions=m.get("Dimensions", []),
            StartTime=start,
            EndTime=end,
            Period=60,
            Statistics=["Average", "Maximum"],
        ).get("Datapoints", [])
        if stats:
            latest = max(stats, key=lambda x: x["Timestamp"])
            out.append(float(latest.get("Maximum", latest.get("Average", 0.0))))
    return out


def aggregate_features():
    features = {}
    responding = 0
    for metric_name, (feature_name, mode) in METRICS.items():
        vals = latest_per_instance(metric_name)
        responding = max(responding, len(vals))
        if not vals:
            features[feature_name] = 0.0
        elif mode == "sum":
            features[feature_name] = sum(vals)
        else:
            features[feature_name] = max(vals)

    # Trend features are emitted by the app in /metrics for the demo guardrail.
    # In the CloudWatch-driven production path they can be zero until dedicated
    # app-side trend metrics are enabled; ideal models accept them safely.
    features.update({
        "cpu_delta_15s": 0.0,
        "cpu_delta_30s": 0.0,
        "connections_delta_15s": 0.0,
        "connections_delta_30s": 0.0,
        "lag_delta_15s": 0.0,
        "lag_delta_30s": 0.0,
        "instance_count": float(responding),
    })
    return features


def invoke_model(features):
    r = sm.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(features),
    )
    body = json.loads(r["Body"].read().decode())
    predicted = float(body["predicted_response_time_ms"][0])
    p90 = float(body.get("p90_response_time_ms", [predicted])[0])
    probs = body.get("breach_probability")
    probability = float(probs[0]) if probs else float(predicted > LATENCY_THRESHOLD_MS)
    return predicted, p90, probability


def publish(name, value, unit):
    cw.put_metric_data(
        Namespace="InfraWatchdog",
        MetricData=[{"MetricName": name, "Value": float(value), "Unit": unit}],
    )


def lambda_handler(event, context):
    features = aggregate_features()
    fallback = False
    try:
        predicted, p90, probability = invoke_model(features)
    except (ClientError, BotoCoreError, KeyError, ValueError, TypeError) as exc:
        fallback = True
        predicted = max(features["recent_latency_ms"], features["recent_p95_latency_ms"])
        p90 = predicted
        probability = 1.0 if (
            predicted > LATENCY_THRESHOLD_MS or features["cpu_usage"] > CPU_THRESHOLD
        ) else 0.0
        print(f"model fallback: {type(exc).__name__}: {exc}")

    publish("PredictedLatencyMs", predicted, "Milliseconds")
    publish("PredictedP90LatencyMs", p90, "Milliseconds")
    publish("BreachProbability", probability, "None")
    publish("ActualLatencyMs", features["recent_latency_ms"], "Milliseconds")
    publish("FleetSize", features["instance_count"], "Count")
    publish("FallbackUsed", 1 if fallback else 0, "Count")

    result = {
        "features": features,
        "predicted_ms": predicted,
        "p90_ms": p90,
        "breach_probability": probability,
        "fallback": fallback,
    }
    print(json.dumps(result))
    return result
