#!/bin/bash
# Update an existing demo instance after infra/deploy_infra.sh uploads a release.
# sudo bash update_app_on_server.sh <bucket> [--reset-data] [--region REGION]
#   [--endpoint NAME] [--threshold-ms NUMBER] [--push-cloudwatch]
set -euo pipefail
BUCKET="${1:?Provide the project S3 bucket}"
shift
REGION="${AWS_DEFAULT_REGION:-ap-south-1}"
ENDPOINT="infra-watchdog-endpoint"
THRESHOLD="500"
RESET=false
PUSH_CW=false
while (($#)); do
  case "$1" in
    --reset-data) RESET=true; shift ;;
    --push-cloudwatch) PUSH_CW=true; shift ;;
    --region) REGION="${2:?Provide a region}"; shift 2 ;;
    --endpoint) ENDPOINT="${2:?Provide an endpoint name}"; shift 2 ;;
    --threshold-ms) THRESHOLD="${2:?Provide the threshold}"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$REGION" =~ ^[a-z0-9-]+$ && "$ENDPOINT" =~ ^[a-zA-Z0-9-]+$ && "$THRESHOLD" =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 2
export AWS_DEFAULT_REGION="$REGION"
cd /opt/infra-watchdog
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/scripts"
for f in app.py requirements.txt dashboard.html; do
  aws s3 cp "s3://$BUCKET/app/$f" "$STAGE/$f"
done
for f in collect_demo_data.py train_on_server.sh upload_metrics_to_s3.py upload_metrics_to_firehose.py update_app_on_server.sh run_real_scaleout_test.sh; do
  aws s3 cp "s3://$BUCKET/app/scripts/$f" "$STAGE/scripts/$f" 2>/dev/null || true
done
venv/bin/python -m py_compile "$STAGE/app.py"
venv/bin/pip install -q -r "$STAGE/requirements.txt"

BACKUP="releases/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$BACKUP"
for f in app.py requirements.txt dashboard.html; do
  [[ -f "$f" ]] && cp "$f" "$BACKUP/"
done
systemctl stop infra-watchdog
cp "$STAGE/app.py" "$STAGE/requirements.txt" "$STAGE/dashboard.html" .
mkdir -p scripts
cp "$STAGE/scripts/"* scripts/ 2>/dev/null || true

if [[ "$RESET" == true && -f metrics_log.csv ]]; then
  mv metrics_log.csv "$BACKUP/metrics_log.csv"
  echo "Archived the old metrics log in $BACKUP"
elif [[ -f metrics_log.csv ]]; then
  # v2 adds instance_count. Keep old rows: pandas treats their missing last
  # column as NaN and train_ideal.py safely fills it with 1.
  HEADER=$(head -n 1 metrics_log.csv | tr -d '\r')
  if [[ "$HEADER" == "timestamp,cpu_usage,active_connections,lag_ms,response_time_ms" ]]; then
    TMP=$(mktemp)
    { echo "timestamp,cpu_usage,active_connections,lag_ms,response_time_ms,instance_count"; tail -n +2 metrics_log.csv; } > "$TMP"
    mv "$TMP" metrics_log.csv
    echo "Upgraded existing metrics_log.csv header for instance_count compatibility"
  fi
fi

mkdir -p /etc/systemd/system/infra-watchdog.service.d
cat > /etc/systemd/system/infra-watchdog.service.d/prediction.conf <<EOF
[Service]
Environment=SAGEMAKER_ENDPOINT_NAME=$ENDPOINT
Environment=AWS_DEFAULT_REGION=$REGION
Environment=LATENCY_THRESHOLD_MS=$THRESHOLD
Environment=PUSH_CLOUDWATCH=$PUSH_CW
Environment=CLOUDWATCH_INTERVAL_S=10
EOF
systemctl daemon-reload
systemctl start infra-watchdog
for attempt in {1..20}; do
  if curl -fsS http://127.0.0.1:8000/health; then
    echo
    curl -fsS http://127.0.0.1:8000/metrics
    echo
    curl -fsS http://127.0.0.1:8000/prediction
    echo
    echo "Application updated. Previous application saved at $BACKUP"
    exit 0
  fi
  sleep 1
done
echo "Application health check failed. Previous application is at $BACKUP" >&2
exit 1
