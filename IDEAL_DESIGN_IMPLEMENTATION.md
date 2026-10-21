# Infra Watchdog — Ideal Design Implementation Map

This branch implements the missing items identified in **Infra Watchdog: Ideal Design and Review (21 Oct 2026)** while keeping the original demo deployment cost-safe.

## What is implemented in the existing/demo path

- 30 s-ahead latency forecasting remains backward compatible.
- `backend/app.py` now emits trend features, request arrival rate, recent p95 latency and instance count.
- Request logging is asynchronous and local CSV files rotate instead of growing without bound.
- `lambda/lambda_function.py` now polls the whole InService fleet, treats timeout/refused/reset/HTTP failures as overload signals, uses a rule fallback if SageMaker fails, understands breach probability, publishes richer CloudWatch metrics and can scale in conservatively.
- `scripts/collect_demo_data.py` supports ramp, spike, hold and mixed profiles for 10+ slowdown evidence campaigns.
- `scripts/run_real_scaleout_test.sh` temporarily allows 1 -> 2 EC2 scale-out, records the ASG evidence and restores max=1 automatically.
- `infra/deploy_infra.sh` now deploys the upgraded Lambda and all new ML helpers so the current demo does not revert to older code on future updates.

## Ideal ML path

`ml/train_ideal.py` adds:

- CPU, connection and lag deltas over 15 s and 30 s;
- request rate, recent p95 latency and instance count;
- gradient-boosted point forecast;
- p50 / p90 quantile forecasts;
- probability of crossing the 500 ms latency threshold;
- whole-cycle walk-forward validation;
- model promotion gates: >=40% improvement over the naive baseline, >=90% breach recall and >=70% breach precision.

`ml/inference.py` serves both old RandomForest models and the new ideal bundle.

`ml/pipeline.py` defines a SageMaker Pipeline. `ml/evaluate_meta.py` extracts the walk-forward metrics and only permits registration when the quality gates pass.

`ml/deploy_endpoint.py` supports both a real-time endpoint and SageMaker Serverless Inference.

## Data lake and retraining

- `infra/firehose_parquet.yaml` creates a DirectPut Firehose stream with Glue schema and S3 Parquet conversion.
- `scripts/upload_metrics_to_firehose.py` sends CSV logs through that stream.
- `ml/build_drift_baseline.py` creates the production feature baseline.
- `lambda/drift_monitor.py` publishes `FeatureDriftScore` and `MissedBreach` and starts retraining when drift/missed breaches occur.
- `infra/ml_operations.yaml` runs drift checks, schedules weekly retraining, and retrains again after a successful ASG instance launch.

## Ideal cloud architecture

`infra/cloudformation_ideal.yaml` is intentionally separate from the student/demo stack because it creates additional billable resources.

It provides:

- internet-facing Application Load Balancer;
- optional ACM HTTPS listener and HTTP -> HTTPS redirect;
- private EC2 instances behind the ALB;
- Auto Scaling Group across supplied private subnets with minimum two instances;
- native ASG scale-out and conservative scale-in policies driven by `BreachProbability`;
- CloudWatch alarms and dashboard;
- forecaster Lambda separated from the scaling action;
- IAM roles scoped to the project resources as far as the AWS APIs allow.

The private subnets must have outbound package/S3 access (for example NAT or a pre-baked AMI plus VPC endpoints). `infra/deploy_ideal.sh` packages the forecaster and deploys this stack deliberately.

## Safety / cost behavior

The existing cost-guard Lambda remains the safety layer for the demo account. The production stack is **not enabled automatically** because an ALB, multiple EC2 instances, VPC networking and inference resources increase spend.

The controlled scale-out script always restores the demo ASG to `MaxSize=1` on exit.

## Evidence still has to be generated

Code can create the experiment, but research evidence cannot be fabricated. Run a real multi-profile campaign and report:

- at least 10 slowdown events;
- breach recall;
- breach precision;
- false alarms per hour;
- median lead time;
- MAE / baseline MAE;
- R-squared;
- endpoint p95 latency;
- one verified 1 -> 2 EC2 scale-out.

Those are measurements, not constants in the repository.
