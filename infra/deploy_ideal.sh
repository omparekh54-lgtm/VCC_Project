#!/usr/bin/env bash
# Deploy the production-grade Infra Watchdog architecture deliberately.
# This creates additional billable resources (ALB, >=2 EC2 instances, NAT, etc.).
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

# Amazon Linux 2023 already ships curl-minimal. Installing the full curl package
# can conflict with it and abort UserData, so remove that redundant package from
# older copies of the template before deployment.
sed 's/dnf install -y python3-pip awscli curl/dnf install -y python3-pip awscli/' \
  cloudformation_ideal.yaml > "$BUILD_DIR/cloudformation_ideal.yaml"

echo ">> Deploying ideal multi-AZ/ALB stack."
aws cloudformation deploy \
  --stack-name "$STACK" \
  --template-file "$BUILD_DIR/cloudformation_ideal.yaml" \
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
    CertificateArn="$CERT_ARN" \
  --no-fail-on-empty-changeset

ASG=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='AutoScalingGroupName'].OutputValue" --output text)

# UserData changes only affect newly launched instances. Refresh the fleet so
# existing nodes converge on the current Launch Template safely.
if [[ -n "$ASG" && "$ASG" != "None" ]]; then
  aws autoscaling start-instance-refresh \
    --auto-scaling-group-name "$ASG" \
    --strategy Rolling \
    --preferences MinHealthyPercentage=50,InstanceWarmup=180,SkipMatching=true \
    --region "$REGION" >/dev/null || true
fi

aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query 'Stacks[0].Outputs' --output table
