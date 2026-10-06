"""
Uploads the metrics_log.csv produced by a Locust run to the S3 bucket that
acts as the training-data store.

Run this ON the EC2 instance (or anywhere with AWS creds + network access to
the file) after a load test finishes:

    python upload_metrics_to_s3.py --bucket infra-watchdog-metrics-<account_id> \
        --file /opt/infra-watchdog/metrics_log.csv
"""

import argparse
import time

import boto3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--file", default="metrics_log.csv")
    parser.add_argument(
        "--key",
        default=None,
        help="S3 key; defaults to data/metrics_log_<unix_timestamp>.csv",
    )
    args = parser.parse_args()

    key = args.key or f"data/metrics_log_{int(time.time())}.csv"
    s3 = boto3.client("s3")
    s3.upload_file(args.file, args.bucket, key)
    print(f"Uploaded {args.file} -> s3://{args.bucket}/{key}")


if __name__ == "__main__":
    main()
