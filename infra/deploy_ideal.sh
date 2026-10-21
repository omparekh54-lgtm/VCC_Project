#!/usr/bin/env bash
# Deploy the production-grade Infra Watchdog architecture deliberately.
# This creates additional billable resources (ALB, >=2 EC2 instances, etc.).
# It is separate from deploy_infra.sh so the student/demo stack remains cheap.
set -euo pipefail

if (( $# < 6 )); then
  echo "Usage: $0 <vpc-id> <public-subnets-csv> <private-subnets-csv> <artifact-bucket> <sagemaker-endpoint> <region> [certificate-arn]" >&2
  exit 2
fi

VPC_ID="$1"
PUBLIC_SUBNETS="$2"
PRIVATE_SUBNETS="$3"
ARTIFACT_BUCKET="$4"
ENDPOINT="$5"
REGION="$6"
CERT_ARN="${7:-}"
STACK="infra-watchdog-ideal"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

package_lambda() {
  local src="$1" out="$2"
  cp "$src" "$BUILD_DIR/lambda_function.py"
  (cd "$BUILD_DIR" && zip -q "$out" lambda_function.py)
  rm -f "$BUILD_DIR/lambda_function.py"
}

package_lambda ../lambda/ideal_forecaster.py ideal-forecaster.zip
aws s3 cp "$BUILD_DIR/ideal-forecaster.zip" "s3://$ARTIFACT_BUCKET/lambda/ideal-forecaster.zip" --region "$REGION"

echo ">> Deploying ideal multi-AZ/ALB stack. This is not the cheap demo stack."
aws cloudformation deploy \
  --stack-name "$STACK" \
  --template-file cloudformation_ideal.yaml \
  --capabilities CAPABILITY_IAM \
  --region "$REGION" \
  --parameter-overrides \
    VpcId="$VPC_ID" \
    PublicSubnetIds="$PUBLIC_SUBNETS" \
    PrivateSubnetIds="$PRIVATE_SUBNETS" \
    ArtifactBucketName="$ARTIFACT_BUCKET" \
    SageMakerEndpointName="$ENDPOINT" \
    ForecasterCodeBucket="$ARTIFACT_BUCKET" \
    ForecasterCodeKey="lambda/ideal-forecaster.zip" \
    CertificateArn="$CERT_ARN"

aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query 'Stacks[0].Outputs' --output table
