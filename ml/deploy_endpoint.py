"""Deploy an Infra Watchdog model to SageMaker.

Supports both the existing small real-time endpoint and the ideal-design
serverless option. Serverless avoids paying for an always-on instance during
low-use periods, while real-time mode remains available for predictable
latency during demonstrations.
"""

import argparse

from sagemaker.sklearn.model import SKLearnModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--model-data", required=True)
    parser.add_argument("--endpoint-name", default="infra-watchdog-endpoint")
    parser.add_argument("--mode", choices=["realtime", "serverless"], default="realtime")
    parser.add_argument("--instance-type", default="ml.t2.medium")
    parser.add_argument("--memory-mb", type=int, default=2048,
                        help="Serverless memory size; valid values depend on SageMaker region/runtime")
    parser.add_argument("--max-concurrency", type=int, default=5)
    args = parser.parse_args()

    model = SKLearnModel(
        model_data=args.model_data,
        role=args.role,
        entry_point="inference.py",
        source_dir=".",
        framework_version="1.2-1",
        py_version="py3",
    )

    if args.mode == "serverless":
        from sagemaker.serverless import ServerlessInferenceConfig
        config = ServerlessInferenceConfig(
            memory_size_in_mb=args.memory_mb,
            max_concurrency=args.max_concurrency,
        )
        predictor = model.deploy(
            serverless_inference_config=config,
            endpoint_name=args.endpoint_name,
        )
    else:
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=args.instance_type,
            endpoint_name=args.endpoint_name,
        )

    print(f"Endpoint deployed: {predictor.endpoint_name} ({args.mode})")


if __name__ == "__main__":
    main()
