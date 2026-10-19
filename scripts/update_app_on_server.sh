#!/bin/bash
# Update an existing instance after infra/deploy_infra.sh uploads the release.
# sudo bash update_app_on_server.sh <bucket> [--reset-data] [--region REGION]
#   [--endpoint NAME] [--threshold-ms NUMBER]
set -euo pipefail
BUCKET="${1:?Provide the project S3 bucket}"
shift
REGION="${AWS_DEFAULT_REGION:-ap-south-1}"
ENDPOINT="infra-watchdog-endpoint"
THRESHOLD="500"
RESET=false
while (($#)); do
  case "$1" in
    --reset-data) RESET=true; shift ;;
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
for f in collect_demo_data.py train_on_server.sh upload_metrics_to_s3.py update_app_on_server.sh; do
  aws s3 cp "s3://$BUCKET/app/scripts/$f" "$STAGE/scripts/$f"
done
venv/bin/python -m py_compile "$STAGE/app.py"
venv/bin/pip install -q -r "$STAGE/requirements.txt"

# Save the previous application before stopping; preserve metrics and models.
BACKUP="releases/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$BACKUP"
for f in app.py requirements.txt dashboard.html; do
  if [[ -f "$f" ]]; then cp "$f" "$BACKUP/"; fi
done
systemctl stop infra-watchdog
cp "$STAGE/app.py" "$STAGE/requirements.txt" "$STAGE/dashboard.html" .
mkdir -p scripts
cp "$STAGE/scripts/"* scripts/
if [[ "$RESET" == true && -f metrics_log.csv ]]; then
  mv metrics_log.csv "$BACKUP/metrics_log.csv"
  echo "Archived the old metrics log in $BACKUP"
fi

# Older instances predate these settings in the launch template.
mkdir -p /etc/systemd/system/infra-watchdog.service.d
cat > /etc/systemd/system/infra-watchdog.service.d/prediction.conf <<EOF
[Service]
Environment=SAGEMAKER_ENDPOINT_NAME=$ENDPOINT
Environment=AWS_DEFAULT_REGION=$REGION
Environment=LATENCY_THRESHOLD_MS=$THRESHOLD
EOF
systemctl daemon-reload
systemctl start infra-watchdog
for attempt in {1..15}; do
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
