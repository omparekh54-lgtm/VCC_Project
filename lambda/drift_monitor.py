"""Infra Watchdog feature-drift and missed-breach monitor.

Reads a training baseline JSON from S3, compares it with recent CloudWatch app
metrics, publishes FeatureDriftScore and MissedBreach, and optionally starts
the SageMaker retraining pipeline when drift exceeds the threshold.
"""

from datetime import datetime, timedelta, timezone
import json
import os

import boto3

BASELINE_BUCKET = os.environ["BASELINE_BUCKET"]
BASELINE_KEY = os.environ.get("BASELINE_KEY", "ml/drift_baseline.json")
PIPELINE_NAME = os.environ.get("PIPELINE_NAME", "InfraWatchdogWeeklyRetraining")
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "3.0"))
LATENCY_THRESHOLD_MS = float(os.environ.get("LATENCY_THRESHOLD_MS", "500"))
PROBABILITY_THRESHOLD = float(os.environ.get("PROBABILITY_THRESHOLD", "0.70"))
LOOKBACK_MINUTES = int(os.environ.get("LOOKBACK_MINUTES", "15"))

s3 = boto3.client("s3")
cw = boto3.client("cloudwatch")
sm = boto3.client("sagemaker")


def baseline():
    obj = s3.get_object(Bucket=BASELINE_BUCKET, Key=BASELINE_KEY)
    return json.loads(obj["Body"].read().decode())["features"]


def recent_app_value(metric_name):
    metrics = cw.list_metrics(Namespace="InfraWatchdog/App", MetricName=metric_name).get("Metrics", [])
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=LOOKBACK_MINUTES)
    values = []
    for m in metrics:
        d = cw.get_metric_statistics(
            Namespace="InfraWatchdog/App",
            MetricName=metric_name,
            Dimensions=m.get("Dimensions", []),
            StartTime=start,
            EndTime=end,
            Period=60,
            Statistics=["Average"],
        ).get("Datapoints", [])
        values.extend(float(x["Average"]) for x in d)
    return sum(values) / len(values) if values else None


def latest_guardrail(metric_name):
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=5)
    d = cw.get_metric_statistics(
        Namespace="InfraWatchdog",
        MetricName=metric_name,
        StartTime=start,
        EndTime=end,
        Period=60,
        Statistics=["Maximum"],
    ).get("Datapoints", [])
    if not d:
        return None
    return float(max(d, key=lambda x: x["Timestamp"])["Maximum"])


def publish(name, value):
    cw.put_metric_data(
        Namespace="InfraWatchdog",
        MetricData=[{"MetricName": name, "Value": float(value), "Unit": "None"}],
    )


def lambda_handler(event, context):
    base = baseline()
    zscores = {}
    for metric_name, stats in base.items():
        value = recent_app_value(metric_name)
        if value is None:
            continue
        zscores[metric_name] = abs(value - float(stats["mean"])) / max(float(stats["std"]), 1e-6)

    drift_score = max(zscores.values()) if zscores else 0.0
    publish("FeatureDriftScore", drift_score)

    actual = latest_guardrail("ActualLatencyMs")
    probability = latest_guardrail("BreachProbability")
    missed = int(
        actual is not None and probability is not None
        and actual > LATENCY_THRESHOLD_MS
        and probability < PROBABILITY_THRESHOLD
    )
    publish("MissedBreach", missed)

    retraining_started = False
    if drift_score >= DRIFT_THRESHOLD or missed:
        try:
            sm.start_pipeline_execution(
                PipelineName=PIPELINE_NAME,
                PipelineExecutionDisplayName=f"auto-retrain-{int(datetime.now().timestamp())}",
            )
            retraining_started = True
        except sm.exceptions.ConflictException:
            retraining_started = False

    result = {
        "drift_score": drift_score,
        "feature_zscores": zscores,
        "actual_latency_ms": actual,
        "breach_probability": probability,
        "missed_breach": missed,
        "retraining_started": retraining_started,
    }
    print(json.dumps(result))
    return result
