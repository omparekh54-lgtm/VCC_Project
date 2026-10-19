# Infra Watchdog

An AI-powered infrastructure watchdog: a regression model watches a backend's
real-time stress signals (CPU, concurrent connections, event-loop lag) and
forecasts API response latency 30 seconds ahead, so a scaling guardrail
can add capacity before users ever see a slowdown.

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
ml/              SageMaker train.py, inference.py, launch/deploy scripts, and
                 compare_scaling.py (model vs. CPU-rule scaling, scored)
lambda/          The scaling-guardrail Lambda function
infra/           CloudFormation template + a deploy helper script
```

## Prerequisites

- An AWS account with the AWS CLI configured (`aws configure`)
- Python 3.10 or 3.11 for the pinned local ML tools (scikit-learn 1.2.2)
- A VPC with at least one **public** subnet (your account's default VPC works)
- `pip install sagemaker boto3` locally, for the `ml/` launch/deploy scripts

## Deploying end to end

**1. Deploy the persistent infra** (S3 bucket, EC2 Auto Scaling group, Lambda,
EventBridge rule -- all wired together, Lambda disabled for now):

```bash
cd infra
./deploy_infra.sh <vpc-id> <subnet-id-1>,<subnet-id-2> <region>
```

This also uploads the app (`backend/app.py`, `requirements.txt`,
`dashboard.html`), the server-side scripts, and `ml/train.py` +
`ml/inference.py` to the new bucket. The EC2 launch template pulls the app
and scripts on boot (into `/opt/infra-watchdog/`).

**Updating an instance that is already running.** Boot-time setup runs only
once, and redeploying the stack does not replace a running instance. After
`deploy_infra.sh`, open a session on it (`aws ssm start-session --target
<instance-id>`) and run:

```bash
sudo bash /opt/infra-watchdog/scripts/update_app_on_server.sh <bucket> --reset-data --region ap-south-1
```

`--reset-data` archives the old `metrics_log.csv`. Use it this time: logs
recorded before the latency-measurement fix (see "Measuring latency
honestly") understate latency and must not be mixed with new rows. An
instance older than the scripts folder won't have the script yet; fetch it
first with:

```bash
sudo mkdir -p /opt/infra-watchdog/scripts
sudo aws s3 cp s3://<bucket>/app/scripts/update_app_on_server.sh /opt/infra-watchdog/scripts/update_app_on_server.sh --region ap-south-1
```

The updater stages downloads, checks Python syntax, and saves the previous app
and archived metrics under `releases/`. It also adds the prediction settings
to older systemd services. For custom deployments, pass `--region`,
`--endpoint`, and `--threshold-ms`. Run the infrastructure update first: it
grants the EC2 role permission to invoke the configured SageMaker endpoint.
The updater checks health, metrics, and prediction responses; `available=false`
means the app is up but the model still needs attention.

**2. Ensure the app instance is running.** The current template starts one
instance automatically. Upload the app files before starting a fresh instance;
if an instance booted before the files were available, restart the deployment
only after confirming its cost guard has not requested shutdown:

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
pip install -r requirements-local.txt   # laptop-side tools; not shipped to SageMaker
python launch_training_job.py \
  --role <SageMakerExecutionRoleArn from the stack outputs> \
  --bucket infra-watchdog-metrics-<account-id>-<region> \
  --data-key data/metrics_log_<timestamp>.csv
```

The training log prints the holdout error next to a naive baseline
("latency stays as it is now"); the model should clearly beat it. See
"What the model predicts" below for the target definition.

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

Watch it work via the `InfraWatchdog / PredictedLatencyMs` CloudWatch metric
(and `MetricsUnreachable`, which is 1 when `/metrics` timed out -- the guardrail
treats that as overload and scales out instead of failing),
or `aws logs tail /aws/lambda/infra-watchdog-scaling-guardrail --follow`.

## What the model predicts

Each training row is one request: the readings `/metrics` showed when it
arrived and its latency. `ml/train.py` turns that into a forecasting problem:

> readings at time *t* → mean latency of requests arriving in
> [*t* + 30 s, *t* + 40 s)

so the number the Lambda gets back means "latency 30 seconds from now".

- **Features:** `cpu_usage`, `active_connections`, `lag_ms`, and
  `recent_latency_ms` (mean latency of requests that finished in the last
  10 s). The log doesn't store the last one; training rebuilds it from each
  row's arrival time and latency, with the same definition the app uses.
- **Target form:** the forest learns the *change* from recent latency to the
  future value; the forecast is recent latency + that change. Learning the
  level directly caps every forecast at the highest latency seen in
  training -- exactly where the watchdog needs to be right.
- **Settings:** `--horizon-s` and `--window-s` change the 30 s and the 10 s
  smoothing window; `--horizon-s 0` estimates current latency instead.
- **Metadata:** feature list, horizon and target form are saved in
  `model_meta.json` next to the model. `inference.py` and the app read it,
  and the Lambda forwards `/metrics` unchanged, so older three-feature
  models keep working.

Scores come from the newest 20 % of the log (a time-ordered holdout, with
training rows whose target window overlaps the test period dropped), printed
next to a naive baseline: "latency 30 s from now equals latency now". Treat
that baseline as the bar to clear. In an earlier laptop test reported with the uploaded project (raw test data
is not included here; three load
ramps, `collect_demo_data.py --peak 16`) the model's 30 s-ahead error was
about half the baseline's: MAE 599 ms vs 1,207 ms with a whole cycle held
out, 767 ms vs 1,606 ms on the newest 20 %.

## Measuring latency honestly

The model is only as good as the latency it learns from, so `/api/simulate`
runs its CPU work on a fixed worker pool (`CPU_WORKERS`, default 1 -- the
GIL lets only one thread compute anyway) and requests beyond that wait in a
queue *inside* the app. The event loop stays free to accept every request,
so the waiting is counted in `active_connections` and included in
`response_time_ms`.

The original design ran the work on the event loop. Waiting requests then
sat in the operating system's socket queue, invisible to the app: at 16
clients users waited ~3.5 s while the app recorded ~0.4 s and never more
than one active connection, so the model was learning the wrong number and
the 500 ms threshold almost never fired. With the worker pool, recorded
latency tracks what clients measure (3.44 s vs 3.55 s at 16 clients).
**Logs collected before this change understate latency; discard them and
collect new data before training.**

## Is it better than CPU-based scaling?

`ml/compare_scaling.py` answers that with numbers. It trains the model on
the earlier part of a load-test log, then replays the part the model never
saw, looking at the server every 30 s like the Lambda does, and records when
each policy would have asked for another instance:

| Policy | Rule |
|---|---|
| Model forecast | 30 s-ahead forecast > 500 ms (this project) |
| Latency now | latency of the last 10 s > 500 ms (reactive; the honest baseline) |
| CPU, 1 min | 60 s average CPU > 70 % |
| CPU, 3 min | the same for 3 minutes running (an AWS target-tracking alarm's default shape) |

For every slowdown (actual latency above the threshold) it reports how many
seconds before or after it began each policy raised the alarm, how many
slowdowns each missed, and false alarms, then draws a chart:

```bash
pip install pandas scikit-learn joblib matplotlib
python ml/compare_scaling.py --data metrics_log.csv          # trains on the first 65 %, replays the rest
python ml/compare_scaling.py --data run1.csv --test-data run2.csv   # or replay a separate run
```

It writes `scaling_comparison.png` and `scaling_comparison.json`. The load
needs several ramps for the replayed part to contain slowdowns:
`scripts/collect_demo_data.py --cycles 3` runs three gradual ramps (about
8 minutes each). On a 2-vCPU instance a single-process app can only push
host CPU to about 50 %, so try `--cpu-threshold 40` too before concluding
the CPU rules never fire -- that, though, is itself a real weakness of CPU
rules for single-threaded backends.

In the laptop test above (one held-out cycle, one slowdown), the model
forecast raised the alarm 5 s before latency crossed 500 ms, "Latency now"
25 s after it, and both CPU rules never fired; the forecast also produced
one false alarm as load dropped off. One slowdown is a small sample --
repeat on the EC2 instance with more cycles before quoting numbers.

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
- **Single uvicorn worker.** `active_connections`, `lag_ms` and the recent
  latency window are held in process memory, so the app currently must run as one worker/process per
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
  MAE/RMSE against the Random Forest baseline in `ml/train.py`, or add trend
  features (e.g. change in connections over the last 30 s) to `/metrics`.
- Add a scale-*in* path (currently the guardrail only ever scales out).

## Credit-protected demonstration deployment

The demonstration uses one Free-plan-compatible `t3.small` EC2 instance (`MaxCapacity=1`) and one small
SageMaker inference endpoint in `ap-south-1`. CPU bursting is in standard
mode to avoid surplus CPU-credit charges. With the one-instance ceiling,
the watchdog can predict latency and publish metrics but cannot add servers.

An hourly AWS cost-guard Lambda stops the EC2 group and deletes the inference
endpoint when total credit consumption reaches $65 (including earned reward credits), when the Free plan is no longer
active, or at the deployment-specific 21-day deadline in its environment.
The earlier $65-used trigger allows a buffer for delayed billing updates;
it is not an exact spending cap. Small retained S3 data and logs may still
consume credits after compute stops. The account must not be upgraded to Paid.

If the SageMaker training quota is zero, train on the EC2 app server
instead, after collecting real load-test data there:

```bash
cd /opt/infra-watchdog
python3 scripts/collect_demo_data.py --cycles 3      # ~25 min of staged load
sudo bash scripts/train_on_server.sh <bucket>
```

This trains the same Random Forest on the server, leaves the model where the
dashboard's `/prediction` picks it up (no restart needed), and uploads an
inference-ready `ml/model.tar.gz` for the endpoint.

`ml/requirements-local.txt` is deliberately not called `requirements.txt`:
SageMaker installs a `requirements.txt` from `source_dir` into its
container, and these laptop pins (pandas 2.2, numpy 1.26, the SageMaker SDK)
don't belong there -- the scikit-learn container already ships what
`train.py` and `inference.py` use.

The backend serves a browser dashboard at `/` (live readings, the
watchdog's forecast next to actual latency, 5-minute charts, and a small
load generator) and its API reference at `/docs`. The forecast comes from
`GET /prediction`, which uses `model/model.joblib` next to the app when
present (left there by `train_on_server.sh`, reloaded when it changes) and
otherwise the SageMaker endpoint named by `SAGEMAKER_ENDPOINT_NAME`, which
the launch template sets.

## Live demonstration

Deployed on 6 October 2026 in Mumbai (`ap-south-1`).

- Dashboard after the update is deployed: http://13.127.199.17:8000/
- Forecast after the update is deployed: http://13.127.199.17:8000/prediction
- Interactive API: http://13.127.199.17:8000/docs
- Health: http://13.127.199.17:8000/health
- Server readings: http://13.127.199.17:8000/metrics

The public IP can change if the EC2 instance is replaced. This demo uses HTTP.
The app fleet is capped at one EC2 instance; the SageMaker prediction endpoint
is a separate managed resource and is included in the cost estimate.
Allow approximately $50–$60 in credits for three weeks of continuous operation;
actual usage varies. Compute shuts down after $65 of account credit consumption
or at the configured 27 October 2026 deadline, whichever comes first.

## Verification

Install `backend/requirements.txt`, plus `pytest`, `httpx`, `numpy`, `pandas`,
`scikit-learn`, and `joblib` in a Python environment, then run
`python -m pytest -q tests` from the repository root. Tests cover dashboard
routing, business-only CSV logging, unavailable-model status, local and
SageMaker prediction responses, legacy models, completed-request latency
windows, and purging future targets from the time-ordered holdout. These are
local functional checks, not a measurement of AWS model accuracy.

The local checks for this ZIP update used Python 3.12, FastAPI 0.115.0,
and the available scikit-learn 1.8.0 runtime. The legacy scikit-learn 1.2.2
package requires an older Python environment for wheel installation; its
SageMaker container and the live AWS stack were not executed in this session.
Synthetic training and comparison runs verify the code path only; they are
not evidence of production forecasting accuracy.
