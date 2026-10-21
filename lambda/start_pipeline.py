"""Small Lambda target used by EventBridge for scheduled retraining."""

from datetime import datetime, timezone
import os
import boto3

PIPELINE_NAME = os.environ.get("PIPELINE_NAME", "InfraWatchdogWeeklyRetraining")
sm = boto3.client("sagemaker")


def lambda_handler(event, context):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    response = sm.start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineExecutionDisplayName=f"weekly-{stamp}",
    )
    return {"pipeline": PIPELINE_NAME, "execution_arn": response["PipelineExecutionArn"]}
