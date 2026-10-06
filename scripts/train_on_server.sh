#!/bin/bash
# Alternative for accounts whose SageMaker training-job quota is zero.
set -euo pipefail
BUCKET="${1:?Provide the project S3 bucket}"
cd /opt/infra-watchdog
mkdir -p ml data model/code
aws s3 cp "s3://$BUCKET/ml/train.py" ml/train.py
aws s3 cp "s3://$BUCKET/ml/inference.py" model/code/inference.py
venv/bin/pip install scikit-learn==1.2.2 pandas==2.2.2 numpy==1.26.4 joblib==1.4.2
cp metrics_log.csv data/metrics_log.csv
venv/bin/python ml/train.py --train data --model-dir model
tar -czf model.tar.gz -C model .
aws s3 cp model.tar.gz "s3://$BUCKET/ml/model.tar.gz"
aws s3 cp data/metrics_log.csv "s3://$BUCKET/data/metrics_log.csv"
