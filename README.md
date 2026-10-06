# Infra Watchdog

An AI-powered infrastructure watchdog: a regression model watches a backend's
real-time stress signals (CPU, concurrent connections, event-loop lag) and
predicts API response latency a few seconds before it degrades, so a
scaling guardrail can act before users ever see a slowdown.

```
                          poll /metrics every ~30s
   ┌─────────────┐  ───────────────────────────────►   ┌──────────────┐
   │   EC2 ASG    │                                     │  AWS Lambda  │
   │  FastAPI app │ ◄───────────────────────────────    │  guardrail   │
   │ (instrumented)│   set_desired_capacity if           └──────┬───────┘
   └──────┬───────┘   predicted latency > threshold             │
          │                                                     │ invoke_endpoint
          │ metrics_log.csv (cpu, connections, lag, resp_time)  ▼
          ▼                                            ┌──────────────────┐
   ┌─────────────┐    trains on     ┌────────────┐     │ SageMaker         │
   │  Amazon S3   │ ───────────────►│ SageMaker  │────►│ real-time endpoint│
   │ data store   │                 │ training   │     │ (RandomForest)    │
   └─────────────┘                  └────────────┘     └──────────────────┘
```

Four AWS services, on purpose: **EC2** (app), **S3** (data/model store),
**SageMaker** (train + serve the model), **Lambda** (the guardrail, on an
EventBridge schedule). No load balancer -- see "Design notes" below for how
the Lambda finds the app instance without one.

## Repo layout

```
backend/         FastAPI app instrumented with cpu/connections/lag metrics
load_testing/    Locust script that ramps traffic to generate training data
scripts/         Uploads the resulting CSV to S3
ml/              SageMaker train.py, inference.py, and launch/deploy scripts
lambda/          The scaling-guardrail Lambda function
infra/           CloudFormation template + a deploy helper script
```

## Prerequisites

- An AWS account with the AWS CLI configured (`aws configure`)
- Python 3.10+
- A VPC with at least one **public** subnet (your account's default VPC works)
- `pip install sagemaker boto3` locally, for the `ml/` launch/deploy scripts

## Deploying end to end

**1. Deploy the persistent infra** (S3 bucket, EC2 Auto Scaling group, Lambda,
EventBridge rule -- all wired together, Lambda disabled for now):

```bash
cd infra
./deploy_infra.sh <vpc-id> <subnet-id-1>,<subnet-id-2> <region>
```

This also uploads `backend/app.py` and `backend/requirements.txt` to the new
bucket, which the EC2 launch template pulls from on boot.

**2. Launch the app instance** (kept at desired capacity 0 until step 1's
upload completes, so it doesn't boot before the code exists):

```bash
aws autoscaling set-desired-capacity \
  --auto-scaling-group-name infra-watchdog-app-asg --desired-capacity 1
```

Find its public IP once it's running:

```bash
aws ec2 describe-instances --filters "Name=tag:Name,Values=infra-watchdog-app" \
  "Name=instance-state-name,Values=running" \
  --query "Reservations[].Instances[].PublicIpAddress" --output text
```

Confirm it's up: `curl http://<public-ip>:8000/health`

**3. Generate training data** with Locust:

```bash
cd load_testing
pip install -r requirements.txt
locust -f locustfile.py --host http://<public-ip>:8000 --headless
```

Let it run through the full ramp (~15 min) so the model sees both healthy and
degraded server states. This appends rows to `metrics_log.csv` **on the EC2
instance**, so either run Locust from a machine that can reach that box, or
`scp`/SSM-copy the resulting log — the important file is the one on the
server, since that's where `cpu_usage`/`active_connections`/`lag_ms` are
actually measured.

**4. Ship the CSV to S3** (run on the EC2 instance, e.g. via
`aws ssm start-session --target <instance-id>`):

```bash
python scripts/upload_metrics_to_s3.py \
  --bucket infra-watchdog-metrics-<account-id>-<region> \
  --file /opt/infra-watchdog/metrics_log.csv
```

**5. Train the model on SageMaker:**

```bash
cd ml
pip install -r requirements.txt
python launch_training_job.py \
  --role <SageMakerExecutionRoleArn from the stack outputs> \
  --bucket infra-watchdog-metrics-<account-id>-<region> \
  --data-key data/metrics_log_<timestamp>.csv
```

**6. Deploy the trained model as a live endpoint:**

```bash
python deploy_endpoint.py \
  --role <SageMakerExecutionRoleArn> \
  --model-data <model_data S3 URI printed by the previous step> \
  --endpoint-name infra-watchdog-endpoint
```

**7. Turn the guardrail on** — everything (app, model, endpoint) is live now,
so enable the schedule:

```bash
aws events enable-rule --name infra-watchdog-schedule
```

Watch it work via the `InfraWatchdog / PredictedLatencyMs` CloudWatch metric,
or `aws logs tail /aws/lambda/infra-watchdog-scaling-guardrail --follow`.

## Design notes / trade-offs

- **No load balancer.** The brief asks for exactly four AWS services, so
  there's no ALB in front of the EC2 fleet. The Lambda instead looks up an
  `InService` instance in the Auto Scaling group and polls its public IP
  directly. Fine for a single-instance demo; add an ALB (and have the Lambda
  hit it instead) if you scale this into something with real user traffic.
- **"AWS Application Auto Scaling" vs. Amazon EC2 Auto Scaling.** Application
  Auto Scaling is the service for resources *other than* raw EC2 (ECS,
  DynamoDB, Aurora, SageMaker endpoints, etc.); a plain EC2 Auto Scaling
  group is scaled through the separate `autoscaling` API, which is what
  `lambda_function.py` actually calls. Worth knowing the distinction if you
  present this project.
- **30-second cadence on a 1-minute-minimum scheduler.** EventBridge's
  `rate()` expressions bottom out at 1 minute. The Lambda is invoked once a
  minute and internally checks twice, 30 seconds apart, to approximate the
  spec's "every 30 seconds."
- **Single uvicorn worker.** `active_connections` and `lag_ms` are held in
  process memory, so the app currently must run as one worker/process per
  instance for those numbers to mean anything. Scaling *out* (more
  instances) is fine; scaling *up* worker count on one instance would need
  shared state (e.g. Redis) instead.
- **Port 8000 open to 0.0.0.0/0.** Simplest way for both Locust and the
  (non-VPC) Lambda to reach `/metrics` in a demo. For anything beyond that,
  run the Lambda inside the VPC and scope the security group to it.

## Cost / cleanup

SageMaker real-time endpoints bill by the hour whether or not they're
receiving traffic. When you're done:

```bash
aws sagemaker delete-endpoint --endpoint-name infra-watchdog-endpoint
aws cloudformation delete-stack --stack-name infra-watchdog
```

## Extending it

- Swap the blunt "+1 instance" scaling for a target-tracking policy driven
  by the `PredictedLatencyMs` CloudWatch metric this Lambda already publishes.
- Try `SageMaker Autopilot` or a gradient-boosted model (XGBoost) and compare
  MAE/RMSE against the Random Forest baseline in `ml/train.py`.
- Add a scale-*in* path (currently the guardrail only ever scales out).
