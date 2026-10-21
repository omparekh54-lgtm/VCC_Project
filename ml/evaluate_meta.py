"""Extract train_ideal.py's walk-forward quality summary for a Pipeline gate."""

import argparse
import json
import os
import tarfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-tar", default="/opt/ml/processing/model/model.tar.gz")
    p.add_argument("--output", default="/opt/ml/processing/evaluation/evaluation.json")
    args = p.parse_args()

    work = Path("/tmp/infra-watchdog-model")
    work.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.model_tar, "r:gz") as tar:
        tar.extractall(work)
    meta = json.loads((work / "model_meta.json").read_text())
    summary = meta.get("summary", {})
    out = {
        "regression_metrics": {
            "mae_ms": summary.get("mean_mae_ms"),
            "baseline_mae_ms": summary.get("mean_baseline_mae_ms"),
            "baseline_improvement_fraction": summary.get("baseline_improvement_fraction"),
        },
        "classification_metrics": {
            "breach_recall": summary.get("mean_breach_recall"),
            "breach_precision": summary.get("mean_breach_precision"),
        },
        "promotable": bool(summary.get("promotable", False)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
