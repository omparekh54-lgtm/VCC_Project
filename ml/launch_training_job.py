"""
Kicks off a SageMaker training job using the built-in scikit-learn container.

Run from inside the ml/ directory (source_dir="." picks up train.py):

    python launch_training_job.py \
        --role arn:aws:iam::<account_id>:role/InfraWatchdogSageMakerRole \
        --bucket infra-watchdog-metrics-<account_id>-<region> \
        --data-key data/metrics_log_1234567890.csv
"""

import argparse

import sagemaker
from sagemaker.sklearn.estimator import SKLearn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--data-key", required=True, help="S3 key (or prefix) of the training CSV(s)")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--job-name-prefix", default="infra-watchdog")
    args = parser.parse_args()

    session = sagemaker.Session()
    train_s3_uri = f"s3://{args.bucket}/{args.data_key}"

    estimator = SKLearn(
        entry_point="train.py",
        source_dir=".",
        role=args.role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        base_job_name=args.job_name_prefix,
        hyperparameters={"n-estimators": 200, "max-depth": 12},
        sagemaker_session=session,
    )

    estimator.fit({"train": train_s3_uri})
    print(f"Training job complete. Model artifact: {estimator.model_data}")


if __name__ == "__main__":
    main()
