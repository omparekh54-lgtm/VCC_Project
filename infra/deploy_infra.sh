#!/bin/bash
# Deploys the persistent infra (S3, EC2 ASG, Lambda, EventBridge rule) and
# uploads the app code to the newly created bucket. Run from the infra/ dir.
#
# Usage:
#   ./deploy_infra.sh <vpc-id> <subnet-id-1,subnet-id-2,...> [region]
#
# Existing instances need the in-place updater after this upload completes.

set -euo pipefail

VPC_ID="${1:?Usage: deploy_infra.sh <vpc-id> <subnet-ids-comma-separated> [region]}"
SUBNET_IDS="${2:?Usage: deploy_infra.sh <vpc-id> <subnet-ids-comma-separated> [region]}"
REGION="${3:-ap-south-1}"
STACK_NAME="infra-watchdog"

echo ">> Deploying CloudFormation stack ($STACK_NAME) in $REGION..."
aws cloudformation deploy \
  --stack-name "$STACK_NAME" \
  --template-file cloudformation.yaml \
  --parameter-overrides VpcId="$VPC_ID" SubnetIds="$SUBNET_IDS" \
  --capabilities CAPABILITY_IAM \
  --region "$REGION" \
  --no-fail-on-empty-changeset

BUCKET=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='MetricsBucketName'].OutputValue" --output text)

echo ">> Uploading backend/ app code to s3://$BUCKET/app/ ..."
aws s3 cp ../backend/app.py "s3://$BUCKET/app/app.py" --region "$REGION"
aws s3 cp ../backend/requirements.txt "s3://$BUCKET/app/requirements.txt" --region "$REGION"
aws s3 cp ../backend/dashboard.html "s3://$BUCKET/app/dashboard.html" --region "$REGION"

echo ">> Uploading server-side helper scripts and ML code ..."
for f in collect_demo_data.py train_on_server.sh upload_metrics_to_s3.py update_app_on_server.sh; do
  aws s3 cp "../scripts/$f" "s3://$BUCKET/app/scripts/$f" --region "$REGION"
done
aws s3 cp ../ml/train.py "s3://$BUCKET/ml/train.py" --region "$REGION"
aws s3 cp ../ml/inference.py "s3://$BUCKET/ml/inference.py" --region "$REGION"

echo ">> Done. Bucket: $BUCKET"
echo ">> A NEW instance picks these files up when it boots. An instance that is"
echo "   already running keeps its old code until you run, on it:"
echo "   sudo bash /opt/infra-watchdog/scripts/update_app_on_server.sh $BUCKET --region $REGION"
