"""Create the ideal Infra Watchdog SageMaker Pipeline.

The pipeline trains train_ideal.py, extracts its honest walk-forward metrics,
and registers a new model package only when all quality gates pass:
  * >=40% MAE improvement over the naive latency-now baseline
  * >=90% breach recall
  * >=70% breach precision

Schedule this pipeline weekly with EventBridge, and trigger it additionally
when model/feature drift alarms fire.
"""

import argparse

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep


def build_pipeline(role, region, bucket, model_package_group="InfraWatchdogModels"):
    session = sagemaker.session.Session(boto_session=boto3.Session(region_name=region))
    training_uri = ParameterString(
        name="TrainingDataUri",
        default_value=f"s3://{bucket}/data/ideal-training/",
    )

    estimator = SKLearn(
        entry_point="train_ideal.py",
        source_dir=".",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=session,
        hyperparameters={
            "train": "/opt/ml/input/data/train",
            "threshold-ms": 500,
            "probability-threshold": 0.70,
            "min-baseline-improvement": 0.40,
        },
        disable_profiler=True,
    )
    train_step = TrainingStep(
        name="TrainIdealForecastModel",
        estimator=estimator,
        inputs={"train": TrainingInput(s3_data=training_uri, content_type="text/csv")},
    )

    evaluator = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        command=["python3"],
        sagemaker_session=session,
    )
    evaluation = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    evaluate_step = ProcessingStep(
        name="EvaluateQualityGates",
        processor=evaluator,
        code="evaluate_meta.py",
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
            )
        ],
        property_files=[evaluation],
    )

    register = RegisterModel(
        name="RegisterPromotableModel",
        estimator=estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group,
        approval_status="PendingManualApproval",
    )

    conditions = [
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=evaluate_step.name, property_file=evaluation,
                         json_path="regression_metrics.baseline_improvement_fraction"),
            right=0.40,
        ),
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=evaluate_step.name, property_file=evaluation,
                         json_path="classification_metrics.breach_recall"),
            right=0.90,
        ),
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=evaluate_step.name, property_file=evaluation,
                         json_path="classification_metrics.breach_precision"),
            right=0.70,
        ),
    ]
    gate = ConditionStep(
        name="PromoteOnlyIfBetter",
        conditions=conditions,
        if_steps=register.steps,
        else_steps=[],
    )

    return Pipeline(
        name="InfraWatchdogWeeklyRetraining",
        parameters=[training_uri],
        steps=[train_step, evaluate_step, gate],
        sagemaker_session=session,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--region", default="ap-south-1")
    p.add_argument("--model-package-group", default="InfraWatchdogModels")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    pipeline = build_pipeline(args.role, args.region, args.bucket, args.model_package_group)
    pipeline.upsert(role_arn=args.role)
    print(f"Pipeline upserted: {pipeline.name}")
    if args.execute:
        execution = pipeline.start()
        print(f"Execution started: {execution.arn}")


if __name__ == "__main__":
    main()
