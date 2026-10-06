#!/bin/bash
# Deploys the persistent infra (S3, EC2 ASG, Lambda, EventBridge rule) and
# uploads the app code to the newly created bucket. Run from the infra/ dir.
#
# Usage:
#   ./deploy_infra.sh <vpc-id> <subnet-id-1,subnet-id-2,...> [region]
#
# After this finishes, the app instance still won't be running -- see the
# "3. Launch the app instance" step in the top-level README.

set -euo pipefail

VPC_ID="${1:?Usage: deploy_infra.sh <vpc-id> <subnet-ids-comma-separated> [region]}"
SUBNET_IDS="${2:?Usage: deploy_infra.sh <vpc-id> <subnet-ids-comma-separated> [region]}"
REGION="${3:-us-east-1}"
STACK_NAME="infra-watchdog"

echo ">> Deploying CloudFormation stack ($STACK_NAME) in $REGION..."
aws cloudformation deploy \
  --stack-name "$STACK_NAME" \
  --template-file cloudformation.yaml \
  --parameter-overrides VpcId="$VPC_ID" SubnetIds="$SUBNET_IDS" \
  --capabilities CAPABILITY_IAM \
  --region "$REGION"

BUCKET=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='MetricsBucketName'].OutputValue" --output text)

echo ">> Uploading backend/ app code to s3://$BUCKET/app/ ..."
aws s3 cp ../backend/app.py "s3://$BUCKET/app/app.py" --region "$REGION"
aws s3 cp ../backend/requirements.txt "s3://$BUCKET/app/requirements.txt" --region "$REGION"

echo ">> Done. Bucket: $BUCKET"
echo ">> Next: scale the ASG up now that the app code exists in S3, e.g.:"
echo "   aws autoscaling set-desired-capacity --auto-scaling-group-name infra-watchdog-app-asg --desired-capacity 1 --region $REGION"
