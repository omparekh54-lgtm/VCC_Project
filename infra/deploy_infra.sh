#!/bin/bash
# Deploys the cost-safe demo infrastructure and then synchronizes the current
# app/ML/Lambda source from the repository. The production-grade alternative
# is deploy_ideal.sh + cloudformation_ideal.yaml.
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
LAMBDA_NAME=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='ScalingGuardrailFunctionName'].OutputValue" --output text)

echo ">> Uploading backend app to s3://$BUCKET/app/ ..."
aws s3 cp ../backend/app.py "s3://$BUCKET/app/app.py" --region "$REGION"
aws s3 cp ../backend/requirements.txt "s3://$BUCKET/app/requirements.txt" --region "$REGION"
aws s3 cp ../backend/dashboard.html "s3://$BUCKET/app/dashboard.html" --region "$REGION"

echo ">> Uploading server-side helper scripts ..."
for f in collect_demo_data.py train_on_server.sh upload_metrics_to_s3.py upload_metrics_to_firehose.py update_app_on_server.sh run_real_scaleout_test.sh; do
  [[ -f "../scripts/$f" ]] && aws s3 cp "../scripts/$f" "s3://$BUCKET/app/scripts/$f" --region "$REGION"
done

echo ">> Uploading ML code ..."
for f in train.py train_ideal.py inference.py compare_scaling.py deploy_endpoint.py launch_training_job.py pipeline.py evaluate_meta.py build_drift_baseline.py; do
  [[ -f "../ml/$f" ]] && aws s3 cp "../ml/$f" "s3://$BUCKET/ml/$f" --region "$REGION"
done

echo ">> Updating the live guardrail Lambda from lambda/lambda_function.py ..."
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
cp ../lambda/lambda_function.py "$TMP_DIR/lambda_function.py"
(cd "$TMP_DIR" && zip -q guardrail.zip lambda_function.py)
aws lambda update-function-code \
  --function-name "$LAMBDA_NAME" \
  --zip-file "fileb://$TMP_DIR/guardrail.zip" \
  --region "$REGION" >/dev/null
aws lambda wait function-updated --function-name "$LAMBDA_NAME" --region "$REGION"

echo ">> Done. Bucket: $BUCKET"
echo ">> New EC2 instances pick up the uploaded app automatically."
echo ">> Existing EC2 instances still need the in-place updater:"
echo "   sudo bash /opt/infra-watchdog/scripts/update_app_on_server.sh $BUCKET --region $REGION"
