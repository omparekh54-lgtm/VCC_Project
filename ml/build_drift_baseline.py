"""Build a simple feature baseline used by the drift-monitor Lambda.

Run after a model is accepted. The JSON contains mean/std for the production
signals that are also emitted to CloudWatch by backend/app.py.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

FEATURES = {
    "CPUUsage": "cpu_usage",
    "ActiveConnections": "active_connections",
    "EventLoopLagMs": "lag_ms",
    "RecentLatencyMs": "recent_latency_ms",
    "RecentP95LatencyMs": "recent_p95_latency_ms",
    "RequestRateRps": "request_rate_rps",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="drift_baseline.json")
    args = p.parse_args()

    from train_ideal import add_runtime_features
    raw = pd.read_csv(args.data)
    df = add_runtime_features(raw)
    out = {"features": {}}
    for metric, col in FEATURES.items():
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        out["features"][metric] = {
            "mean": float(s.mean()),
            "std": max(float(s.std(ddof=0)), 1e-6),
            "p05": float(s.quantile(.05)),
            "p95": float(s.quantile(.95)),
        }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
