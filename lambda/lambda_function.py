"""
Infra Watchdog - the scaling guardrail Lambda.

Triggered on a schedule (EventBridge rule, see infra/cloudformation.yaml).
Each invocation:
  1. Finds an in-service instance in the app's Auto Scaling group and polls
     its /metrics endpoint for current stress readings. (There's no load
     balancer in this architecture -- by design, to keep the stack at exactly
     four AWS services -- so the Lambda discovers the instance itself via
     the ASG + EC2 APIs instead of hitting a fixed IP.)
  2. Sends those readings to the SageMaker endpoint to get a *predicted*
     response_time_ms slightly ahead of where the server actually is.
  3. If the prediction crosses LATENCY_THRESHOLD_MS, scales the Auto Scaling
     group out by one instance.

If /metrics doesn't answer within the timeout, that is treated as overload
too, not as an error: the app serves /metrics from the same event loop that
CPU-heavy requests block, so the server goes quiet exactly when it is most
saturated. Crashing there would leave the guardrail blind at the worst time.
Only timeouts count -- a refused connection (app down) still raises.

NOTE ON SERVICE NAMES: step 3 scales the group via Amazon EC2 Auto Scaling
(the `autoscaling` boto3 client) rather than the separate "AWS Application
Auto Scaling" service. Application Auto Scaling covers resources like ECS,
DynamoDB, Aurora, and SageMaker endpoints themselves -- it does not manage
raw EC2 Auto Scaling groups, which have their own native scaling API. If you
want target-tracking or step-scaling policies instead of a blunt "+1", create
those directly on the ASG and have this Lambda publish the CloudWatch custom
metric below to drive an alarm instead of calling set_desired_capacity.

CADENCE: EventBridge's minimum schedule granularity is 1 minute
(rate(1 minute)); there is no rate(30 seconds). To approximate the requested
30-second cadence, the rule invokes this function once a minute and the
handler runs the check twice, 30 seconds apart, in one invocation (the
CloudFormation template sets the Lambda timeout to 55s to leave room for
this).
"""

import json
import os
import time
import urllib.error
import urllib.request

import boto3
from botocore.exceptions import ClientError

SAGEMAKER_ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]
ASG_NAME = os.environ["ASG_NAME"]
APP_PORT = os.environ.get("APP_PORT", "8000")
LATENCY_THRESHOLD_MS = float(os.environ.get("LATENCY_THRESHOLD_MS", "500"))
MAX_SCALE_STEP = int(os.environ.get("MAX_SCALE_STEP", "1"))

sagemaker_runtime = boto3.client("sagemaker-runtime")
autoscaling = boto3.client("autoscaling")
ec2 = boto3.client("ec2")
cloudwatch = boto3.client("cloudwatch")


def get_app_instance_ip() -> str:
    asg = autoscaling.describe_auto_scaling_groups(AutoScalingGroupNames=[ASG_NAME])[
        "AutoScalingGroups"
    ][0]
    instance_ids = [i["InstanceId"] for i in asg["Instances"] if i["LifecycleState"] == "InService"]
    if not instance_ids:
        raise RuntimeError(f"No InService instances in ASG {ASG_NAME}")

    reservations = ec2.describe_instances(InstanceIds=instance_ids[:1])["Reservations"]
    ip = reservations[0]["Instances"][0].get("PublicIpAddress")
    if not ip:
        raise RuntimeError("App instance has no public IP yet (still booting?)")
    return ip


def fetch_metrics():
    """Current readings, or None if the app is too busy to answer in time."""
    ip = get_app_instance_ip()
    try:
        with urllib.request.urlopen(f"http://{ip}:{APP_PORT}/metrics", timeout=5) as resp:
            return json.loads(resp.read().decode())
    except TimeoutError:
        return None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            return None
        raise


def predict_latency(metrics: dict) -> float:
    # Forward /metrics as-is: the endpoint picks the features its model was
    # trained on (see ml/inference.py), so adding one needs no change here.
    payload = json.dumps(metrics)
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload,
    )
    result = json.loads(response["Body"].read().decode())
    return result["predicted_response_time_ms"][0]


def put_metric(name: str, value: float, unit: str) -> None:
    cloudwatch.put_metric_data(
        Namespace="InfraWatchdog", MetricData=[{"MetricName": name, "Value": value, "Unit": unit}]
    )


def scale_out() -> str:
    group = autoscaling.describe_auto_scaling_groups(AutoScalingGroupNames=[ASG_NAME])[
        "AutoScalingGroups"
    ][0]
    current, max_cap = group["DesiredCapacity"], group["MaxSize"]
    new_capacity = min(current + MAX_SCALE_STEP, max_cap)

    if new_capacity <= current:
        return "already at max capacity"
    try:
        autoscaling.set_desired_capacity(
            AutoScalingGroupName=ASG_NAME, DesiredCapacity=new_capacity, HonorCooldown=True
        )
    except ClientError as exc:
        # With HonorCooldown, AWS rejects the call while the previous scale-out
        # is still in its cooldown. That's expected during a sustained overload
        # (the alarm fires on every check), not a failure.
        if exc.response["Error"]["Code"] == "ScalingActivityInProgress":
            return f"scale-out already in progress (cooldown), capacity {current}"
        raise
    return f"scaled {current} -> {new_capacity}"


def run_check():
    metrics = fetch_metrics()
    put_metric("MetricsUnreachable", 0 if metrics else 1, "Count")
    if metrics is None:
        outcome = scale_out()
        print(f"/metrics timed out, treating as overload; outcome={outcome}")
        return None, outcome

    predicted = predict_latency(metrics)
    put_metric("PredictedLatencyMs", predicted, "Milliseconds")
    has_traffic = metrics["active_connections"] > 0 or metrics.get("recent_latency_ms", 0) > 0
    if not has_traffic:
        outcome = "idle: no recent business traffic"
    else:
        outcome = scale_out() if predicted > LATENCY_THRESHOLD_MS else "ok"
    print(f"metrics={metrics} predicted_latency_ms={predicted:.1f} outcome={outcome}")
    return predicted, outcome


def lambda_handler(event, context):
    results = [run_check()]
    time.sleep(30)  # see CADENCE note above
    results.append(run_check())
    return {"checks": [{"predicted_latency_ms": p, "outcome": o} for p, o in results]}
