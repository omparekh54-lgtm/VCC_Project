"""
Infra Watchdog - resilient predictive scaling guardrail.

This version implements the gaps called out in the Ideal Design review:
- polls every InService instance instead of only the first one;
- treats timeout/refused/reset/HTTP failures as overload signals rather than
  crashing blind;
- understands optional breach_probability from the model;
- falls back to current latency/CPU when SageMaker is unavailable;
- publishes fleet-wide observability metrics;
- supports conservative scale-in after two consecutive healthy checks.

The demo stack may still cap MaxSize at 1 for cost safety. In that case the
logic records the scale decision but EC2 Auto Scaling correctly refuses to
increase the fleet beyond the configured ceiling.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
import urllib.error
import urllib.request

import boto3
from botocore.exceptions import BotoCoreError, ClientError

SAGEMAKER_ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]
ASG_NAME = os.environ["ASG_NAME"]
APP_PORT = os.environ.get("APP_PORT", "8000")
LATENCY_THRESHOLD_MS = float(os.environ.get("LATENCY_THRESHOLD_MS", "500"))
BREACH_PROBABILITY_THRESHOLD = float(os.environ.get("BREACH_PROBABILITY_THRESHOLD", "0.70"))
FALLBACK_CPU_THRESHOLD = float(os.environ.get("FALLBACK_CPU_THRESHOLD", "70"))
SCALE_IN_LATENCY_MS = float(os.environ.get("SCALE_IN_LATENCY_MS", "250"))
SCALE_IN_CPU_THRESHOLD = float(os.environ.get("SCALE_IN_CPU_THRESHOLD", "35"))
MIN_CAPACITY = int(os.environ.get("MIN_CAPACITY", "1"))
MAX_SCALE_STEP = int(os.environ.get("MAX_SCALE_STEP", "1"))
METRICS_TIMEOUT_S = float(os.environ.get("METRICS_TIMEOUT_S", "5"))

sagemaker_runtime = boto3.client("sagemaker-runtime")
autoscaling = boto3.client("autoscaling")
ec2 = boto3.client("ec2")
cloudwatch = boto3.client("cloudwatch")


def get_group():
    groups = autoscaling.describe_auto_scaling_groups(AutoScalingGroupNames=[ASG_NAME])["AutoScalingGroups"]
    if not groups:
        raise RuntimeError(f"Auto Scaling group not found: {ASG_NAME}")
    return groups[0]


def get_app_instances():
    group = get_group()
    ids = [
        i["InstanceId"] for i in group["Instances"]
        if i["LifecycleState"] == "InService" and i.get("HealthStatus") == "Healthy"
    ]
    if not ids:
        return [], group
    reservations = ec2.describe_instances(InstanceIds=ids)["Reservations"]
    instances = [i for r in reservations for i in r.get("Instances", [])]
    return instances, group


def fetch_one(instance):
    ip = instance.get("PublicIpAddress") or instance.get("PrivateIpAddress")
    if not ip:
        return None, f"{instance['InstanceId']}: no reachable IP"
    url = f"http://{ip}:{APP_PORT}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=METRICS_TIMEOUT_S) as resp:
            if resp.status >= 500:
                return None, f"{instance['InstanceId']}: HTTP {resp.status}"
            return json.loads(resp.read().decode()), None
    except (TimeoutError, ConnectionError, ConnectionResetError, ConnectionRefusedError) as exc:
        return None, f"{instance['InstanceId']}: {type(exc).__name__}"
    except urllib.error.HTTPError as exc:
        return None, f"{instance['InstanceId']}: HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return None, f"{instance['InstanceId']}: {exc.reason}"
    except OSError as exc:
        return None, f"{instance['InstanceId']}: {type(exc).__name__}: {exc}"


def fetch_fleet_metrics():
    instances, group = get_app_instances()
    if not instances:
        return None, group, ["no healthy InService instances"]

    readings, errors = [], []
    with ThreadPoolExecutor(max_workers=min(8, len(instances))) as pool:
        futures = {pool.submit(fetch_one, i): i["InstanceId"] for i in instances}
        for future in as_completed(futures):
            metrics, error = future.result()
            if metrics is not None:
                readings.append(metrics)
            if error:
                errors.append(error)

    if not readings:
        return None, group, errors

    def values(key):
        return [float(m.get(key, 0) or 0) for m in readings]

    # Aggregate conservatively: worst-case latency/CPU/lag, summed request load,
    # and carry the ideal-model trend features when the app publishes them.
    merged = {
        "cpu_usage": max(values("cpu_usage")),
        "active_connections": sum(values("active_connections")),
        "lag_ms": max(values("lag_ms")),
        "recent_latency_ms": max(values("recent_latency_ms")),
        "cpu_delta_15s": max(values("cpu_delta_15s")),
        "cpu_delta_30s": max(values("cpu_delta_30s")),
        "connections_delta_15s": sum(values("connections_delta_15s")),
        "connections_delta_30s": sum(values("connections_delta_30s")),
        "lag_delta_15s": max(values("lag_delta_15s")),
        "lag_delta_30s": max(values("lag_delta_30s")),
        "request_rate_rps": sum(values("request_rate_rps")),
        "recent_p95_latency_ms": max(values("recent_p95_latency_ms")),
        "instance_count": len(readings),
    }
    merged["fleet_avg_cpu"] = sum(values("cpu_usage")) / len(readings)
    merged["fleet_max_cpu"] = max(values("cpu_usage"))
    merged["fleet_responding_instances"] = len(readings)
    merged["fleet_expected_instances"] = len(instances)
    return merged, group, errors


def predict(metrics):
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(metrics),
    )
    result = json.loads(response["Body"].read().decode())
    predicted = float(result["predicted_response_time_ms"][0])
    p50 = float(result.get("p50_response_time_ms", [predicted])[0])
    p90 = float(result.get("p90_response_time_ms", [predicted])[0])
    probs = result.get("breach_probability")
    breach_probability = float(probs[0]) if probs else None
    return {
        "predicted_ms": predicted,
        "p50_ms": p50,
        "p90_ms": p90,
        "breach_probability": breach_probability,
        "horizon_s": result.get("horizon_s"),
    }


def put_metrics(items):
    cloudwatch.put_metric_data(
        Namespace="InfraWatchdog",
        MetricData=[{"MetricName": n, "Value": float(v), "Unit": u} for n, v, u in items],
    )


def set_capacity(group, desired, reason):
    current = group["DesiredCapacity"]
    desired = max(group["MinSize"], min(group["MaxSize"], int(desired)))
    if desired == current:
        return f"capacity unchanged at {current}: {reason}"
    try:
        autoscaling.set_desired_capacity(
            AutoScalingGroupName=ASG_NAME,
            DesiredCapacity=desired,
            HonorCooldown=True,
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ScalingActivityInProgress":
            return f"scaling already in progress; desired remains {current}"
        raise
    return f"scaled {current} -> {desired}: {reason}"


def scale_out(group, reason):
    return set_capacity(group, group["DesiredCapacity"] + MAX_SCALE_STEP, reason)


def scale_in(group, reason):
    floor = max(MIN_CAPACITY, group["MinSize"])
    return set_capacity(group, max(floor, group["DesiredCapacity"] - 1), reason)


def run_check(allow_scale_in=False):
    metrics, group, errors = fetch_fleet_metrics()
    unreachable = 1 if metrics is None or errors else 0

    if metrics is None:
        put_metrics([
            ("MetricsUnreachable", 1, "Count"),
            ("FleetSize", group["DesiredCapacity"], "Count"),
            ("FallbackUsed", 1, "Count"),
        ])
        outcome = scale_out(group, "all fleet metrics unavailable")
        return {"safe_for_scale_in": False, "prediction": None, "outcome": outcome, "errors": errors}

    has_traffic = metrics["active_connections"] > 0 or metrics["recent_latency_ms"] > 0 or metrics["request_rate_rps"] > 0
    fallback_used = 0
    prediction = None
    try:
        prediction = predict(metrics)
    except (ClientError, BotoCoreError, KeyError, ValueError, TypeError) as exc:
        fallback_used = 1
        prediction_error = f"{type(exc).__name__}: {exc}"[:250]
    else:
        prediction_error = None

    breach_probability = prediction.get("breach_probability") if prediction else None
    predicted_ms = prediction.get("predicted_ms") if prediction else None

    if not has_traffic:
        wants_scale_out = False
        reason = "idle: no recent business traffic"
    elif prediction is not None and breach_probability is not None:
        wants_scale_out = breach_probability >= BREACH_PROBABILITY_THRESHOLD
        reason = f"breach probability {breach_probability:.3f}"
    elif prediction is not None:
        wants_scale_out = predicted_ms > LATENCY_THRESHOLD_MS
        reason = f"forecast {predicted_ms:.1f} ms"
    else:
        wants_scale_out = (
            metrics["recent_latency_ms"] > LATENCY_THRESHOLD_MS
            or metrics["recent_p95_latency_ms"] > LATENCY_THRESHOLD_MS
            or metrics["cpu_usage"] > FALLBACK_CPU_THRESHOLD
            or bool(errors)
        )
        reason = "rule-based fallback after inference failure"

    safe_for_scale_in = (
        has_traffic
        and not wants_scale_out
        and metrics["recent_latency_ms"] < SCALE_IN_LATENCY_MS
        and metrics["recent_p95_latency_ms"] < LATENCY_THRESHOLD_MS
        and metrics["cpu_usage"] < SCALE_IN_CPU_THRESHOLD
        and not errors
    )

    if wants_scale_out:
        outcome = scale_out(group, reason)
    elif allow_scale_in and safe_for_scale_in and group["DesiredCapacity"] > max(MIN_CAPACITY, group["MinSize"]):
        outcome = scale_in(group, "two consecutive healthy checks")
    else:
        outcome = "ok" if has_traffic else reason

    metric_items = [
        ("MetricsUnreachable", unreachable, "Count"),
        ("ActualLatencyMs", metrics["recent_latency_ms"], "Milliseconds"),
        ("RecentP95LatencyMs", metrics["recent_p95_latency_ms"], "Milliseconds"),
        ("FleetAverageCPU", metrics["fleet_avg_cpu"], "Percent"),
        ("FleetMaxCPU", metrics["fleet_max_cpu"], "Percent"),
        ("FleetSize", metrics["instance_count"], "Count"),
        ("RequestRateRps", metrics["request_rate_rps"], "Count/Second"),
        ("FallbackUsed", fallback_used, "Count"),
    ]
    if prediction is not None:
        metric_items.extend([
            ("PredictedLatencyMs", prediction["predicted_ms"], "Milliseconds"),
            ("PredictedP90LatencyMs", prediction["p90_ms"], "Milliseconds"),
        ])
        if breach_probability is not None:
            metric_items.append(("BreachProbability", breach_probability, "None"))
    put_metrics(metric_items)

    print(json.dumps({
        "metrics": metrics,
        "prediction": prediction,
        "prediction_error": prediction_error,
        "errors": errors,
        "outcome": outcome,
    }, default=str))
    return {
        "safe_for_scale_in": safe_for_scale_in,
        "prediction": prediction,
        "outcome": outcome,
        "errors": errors,
    }


def lambda_handler(event, context):
    # EventBridge only schedules at one-minute granularity. Two checks in one
    # invocation approximate 30 s cadence; the second check may scale in only
    # if both checks are healthy, preventing oscillation after a transient dip.
    first = run_check(allow_scale_in=False)
    time.sleep(30)
    second = run_check(allow_scale_in=first["safe_for_scale_in"])
    return {"checks": [first, second]}
