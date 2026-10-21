#!/usr/bin/env bash
# Controlled evidence run: temporarily allow a second EC2 instance, generate
# load, capture ASG activity, then restore the cost-safe one-instance ceiling.
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
ASG="${ASG_NAME:-infra-watchdog-app-asg}"
HOST="${HOST:-http://127.0.0.1:8000}"
DURATION="${DURATION:-600}"

restore() {
  echo ">> Restoring cost-safe ASG max=1 desired<=1"
  aws autoscaling update-auto-scaling-group \
    --auto-scaling-group-name "$ASG" --max-size 1 --region "$REGION" || true
  local desired
  desired=$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$ASG" --region "$REGION" \
    --query 'AutoScalingGroups[0].DesiredCapacity' --output text 2>/dev/null || echo 1)
  if [[ "$desired" =~ ^[0-9]+$ ]] && (( desired > 1 )); then
    aws autoscaling set-desired-capacity --auto-scaling-group-name "$ASG" --desired-capacity 1 --honor-cooldown --region "$REGION" || true
  fi
}
trap restore EXIT INT TERM

echo ">> Temporarily enabling real scale-out: max 1 -> 2"
aws autoscaling update-auto-scaling-group \
  --auto-scaling-group-name "$ASG" --max-size 2 --region "$REGION"

echo ">> Starting controlled high-load campaign for ${DURATION}s"
python3 scripts/collect_demo_data.py \
  --host "$HOST" --cycles 2 --cycle-s "$((DURATION / 2))" --profile spike --peak 28 &
LOAD_PID=$!

while kill -0 "$LOAD_PID" 2>/dev/null; do
  aws autoscaling describe-auto-scaling-groups \
    --auto-scaling-group-names "$ASG" --region "$REGION" \
    --query 'AutoScalingGroups[0].{Desired:DesiredCapacity,InService:length(Instances[?LifecycleState==`InService`]),Max:MaxSize}' \
    --output table
  sleep 30
done
wait "$LOAD_PID"

echo ">> Recent scaling activities"
aws autoscaling describe-scaling-activities \
  --auto-scaling-group-name "$ASG" --region "$REGION" --max-items 20 \
  --query 'Activities[].{Start:StartTime,Status:StatusCode,Description:Description,Cause:Cause}' \
  --output table

echo ">> Evidence run complete; ASG will now be restored by the exit trap."
