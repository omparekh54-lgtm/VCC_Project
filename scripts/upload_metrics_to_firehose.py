"""Send one or more Infra Watchdog CSV logs to the Firehose Parquet data lake.

Firehose receives newline-delimited JSON and converts it to partitioned
Parquet using infra/firehose_parquet.yaml's Glue schema.
"""

import argparse
import csv
import glob
import json
import os

import boto3

FIELDS = ["timestamp", "cpu_usage", "active_connections", "lag_ms", "response_time_ms", "instance_count"]


def rows(path):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out = {}
            for field in FIELDS:
                raw = row.get(field, "1" if field == "instance_count" else "0")
                out[field] = float(raw or 0)
            yield out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV file or folder of CSV files")
    p.add_argument("--stream", default="infra-watchdog-metrics")
    p.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"))
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.csv"))) if os.path.isdir(args.data) else [args.data]
    firehose = boto3.client("firehose", region_name=args.region)
    sent, failed = 0, 0
    batch = []

    def flush():
        nonlocal sent, failed, batch
        if not batch:
            return
        resp = firehose.put_record_batch(DeliveryStreamName=args.stream, Records=batch)
        sent += len(batch) - int(resp.get("FailedPutCount", 0))
        failed += int(resp.get("FailedPutCount", 0))
        batch = []

    for file in files:
        for row in rows(file):
            payload = (json.dumps(row, separators=(",", ":")) + "\n").encode()
            batch.append({"Data": payload})
            if len(batch) == 500 or sum(len(x["Data"]) for x in batch) > 3_500_000:
                flush()
    flush()
    print(f"Firehose upload complete: sent={sent} failed={failed} files={len(files)}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
