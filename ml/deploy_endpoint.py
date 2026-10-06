"""
Deploys a trained model artifact (produced by launch_training_job.py) to a
live SageMaker real-time inference endpoint.

    python deploy_endpoint.py \
        --role arn:aws:iam::<account_id>:role/InfraWatchdogSageMakerRole \
        --model-data s3://infra-watchdog-metrics-<account_id>-<region>/.../model.tar.gz \
        --endpoint-name infra-watchdog-endpoint
"""

import argparse

from sagemaker.sklearn.model import SKLearnModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--model-data", required=True, help="S3 URI of model.tar.gz from training")
    parser.add_argument("--endpoint-name", default="infra-watchdog-endpoint")
    parser.add_argument("--instance-type", default="ml.t2.medium")
    args = parser.parse_args()

    model = SKLearnModel(
        model_data=args.model_data,
        role=args.role,
        entry_point="inference.py",
        source_dir=".",
        framework_version="1.2-1",
        py_version="py3",
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
    )
    print(f"Endpoint deployed: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()
