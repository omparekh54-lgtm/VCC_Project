"""Ideal Infra Watchdog model training pipeline.

Implements the ML-side recommendations from the Ideal Design review while
remaining compatible with the existing CSV logs:
- derived trend features over 15 s / 30 s
- request arrival rate, recent p95 latency and instance count
- gradient-boosted point forecast of latency change
- p50 and p90 quantile forecasts
- breach probability classifier
- whole-cycle / walk-forward evaluation support
- model is only considered promotable when it beats the naive baseline

The trained object is a dictionary saved as model.joblib; ml/inference.py is
backward compatible with both this bundle and the original RandomForest model.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

LATENCY = "response_time_ms"
RECENT_WINDOW_S = 10.0
DEFAULT_THRESHOLD_MS = 500.0
BASE_FEATURES = ["cpu_usage", "active_connections", "lag_ms", "recent_latency_ms"]
TREND_FEATURES = [
    "cpu_delta_15s", "cpu_delta_30s",
    "connections_delta_15s", "connections_delta_30s",
    "lag_delta_15s", "lag_delta_30s",
    "request_rate_rps", "recent_p95_latency_ms", "instance_count",
]
FEATURES = BASE_FEATURES + TREND_FEATURES
TARGET = "future_latency_ms"


def _window_mean(ts, values, starts, ends):
    csum = np.concatenate([[0.0], np.cumsum(values)])
    lo = np.searchsorted(ts, starts, side="left")
    hi = np.searchsorted(ts, ends, side="left")
    n = hi - lo
    out = np.full(len(starts), np.nan)
    ok = n > 0
    out[ok] = (csum[hi[ok]] - csum[lo[ok]]) / n[ok]
    return out


def _previous_value(ts, values, at):
    idx = np.searchsorted(ts, at, side="right") - 1
    idx = np.clip(idx, 0, len(ts) - 1)
    return values[idx]


def _recent_p95(df, times):
    finish = df["timestamp"].to_numpy(float) + df[LATENCY].to_numpy(float) / 1000.0
    values = df[LATENCY].to_numpy(float)
    order = np.argsort(finish, kind="stable")
    finish, values = finish[order], values[order]
    out = np.zeros(len(times), dtype=float)
    for j, t in enumerate(times):
        lo = np.searchsorted(finish, t - RECENT_WINDOW_S, side="left")
        hi = np.searchsorted(finish, t, side="left")
        out[j] = float(np.percentile(values[lo:hi], 95)) if hi > lo else 0.0
    return out


def add_runtime_features(df):
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    ts = df["timestamp"].to_numpy(float)
    latency = df[LATENCY].to_numpy(float)
    finish = ts + latency / 1000.0
    order = np.argsort(finish, kind="stable")
    recent = _window_mean(finish[order], latency[order], ts - RECENT_WINDOW_S, ts)
    df["recent_latency_ms"] = np.nan_to_num(recent, nan=0.0)

    for col, prefix in (("cpu_usage", "cpu"), ("active_connections", "connections"), ("lag_ms", "lag")):
        vals = df[col].to_numpy(float)
        for sec in (15, 30):
            df[f"{prefix}_delta_{sec}s"] = vals - _previous_value(ts, vals, ts - sec)

    arrivals = np.searchsorted(ts, ts, side="right") - np.searchsorted(ts, ts - RECENT_WINDOW_S, side="left")
    df["request_rate_rps"] = arrivals / RECENT_WINDOW_S
    df["recent_p95_latency_ms"] = _recent_p95(df, ts)
    if "instance_count" not in df.columns:
        df["instance_count"] = 1.0
    else:
        df["instance_count"] = pd.to_numeric(df["instance_count"], errors="coerce").fillna(1.0)
    return df


def load_data(path):
    files = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
    if not files:
        raise FileNotFoundError(path)
    frames = []
    for cycle, f in enumerate(files):
        d = pd.read_csv(f)
        d["cycle_id"] = Path(f).stem if len(files) > 1 else str(cycle)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    need = ["timestamp", "cpu_usage", "active_connections", "lag_ms", LATENCY]
    df = df.dropna(subset=need)
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return add_runtime_features(df.dropna(subset=need))


def add_target(df, horizon_s=30.0, window_s=10.0, threshold_ms=DEFAULT_THRESHOLD_MS):
    ts = df["timestamp"].to_numpy(float)
    rt = df[LATENCY].to_numpy(float)
    out = df.copy()
    out[TARGET] = _window_mean(ts, rt, ts + horizon_s, ts + horizon_s + window_s)
    complete = ts + horizon_s + window_s <= ts[-1]
    out = out[complete & out[TARGET].notna()].reset_index(drop=True)
    out["breach_target"] = (out[TARGET] > threshold_ms).astype(int)
    return out


def new_bundle(random_state=42):
    common = dict(n_estimators=250, max_depth=3, min_samples_leaf=3, random_state=random_state)
    return {
        "regressor": GradientBoostingRegressor(loss="huber", **common),
        "q50": GradientBoostingRegressor(loss="quantile", alpha=0.50, **common),
        "q90": GradientBoostingRegressor(loss="quantile", alpha=0.90, **common),
        "classifier": GradientBoostingClassifier(n_estimators=250, max_depth=3, min_samples_leaf=3, random_state=random_state),
    }


def fit_bundle(bundle, train):
    X = train[FEATURES].to_numpy(float)
    delta = train[TARGET].to_numpy(float) - train["recent_latency_ms"].to_numpy(float)
    bundle["regressor"].fit(X, delta)
    bundle["q50"].fit(X, delta)
    bundle["q90"].fit(X, delta)
    y_cls = train["breach_target"].to_numpy(int)
    if len(np.unique(y_cls)) >= 2:
        bundle["classifier"].fit(X, y_cls)
    else:
        bundle["classifier"] = None
    return bundle


def predict_bundle(bundle, test):
    X = test[FEATURES].to_numpy(float)
    recent = test["recent_latency_ms"].to_numpy(float)
    point = np.maximum(0.0, recent + bundle["regressor"].predict(X))
    p50 = np.maximum(0.0, recent + bundle["q50"].predict(X))
    p90 = np.maximum(0.0, recent + bundle["q90"].predict(X))
    if bundle["classifier"] is not None:
        prob = bundle["classifier"].predict_proba(X)[:, 1]
    else:
        prob = (point > DEFAULT_THRESHOLD_MS).astype(float)
    return point, p50, p90, prob


def score_fold(train, test, threshold_ms, probability_threshold):
    bundle = fit_bundle(new_bundle(), train)
    point, p50, p90, prob = predict_bundle(bundle, test)
    actual = test[TARGET].to_numpy(float)
    baseline = test["recent_latency_ms"].to_numpy(float)
    truth = actual > threshold_ms
    alarm = prob >= probability_threshold
    return {
        "rows": int(len(test)),
        "mae_ms": float(mean_absolute_error(actual, point)),
        "baseline_mae_ms": float(mean_absolute_error(actual, baseline)),
        "rmse_ms": float(np.sqrt(mean_squared_error(actual, point))),
        "r2": float(r2_score(actual, point)),
        "breach_recall": float(recall_score(truth, alarm, zero_division=0)),
        "breach_precision": float(precision_score(truth, alarm, zero_division=0)),
        "p90_coverage": float(np.mean(actual <= p90)),
    }, bundle


def walk_forward(df, threshold_ms, probability_threshold):
    cycles = list(dict.fromkeys(df["cycle_id"].astype(str)))
    results = []
    if len(cycles) >= 3:
        for i in range(1, len(cycles)):
            train_cycles = cycles[:i]
            test_cycle = cycles[i]
            train = df[df["cycle_id"].astype(str).isin(train_cycles)]
            test = df[df["cycle_id"].astype(str) == test_cycle]
            if len(train) >= 50 and len(test) >= 10:
                metrics, _ = score_fold(train, test, threshold_ms, probability_threshold)
                metrics.update({"train_cycles": train_cycles, "test_cycle": test_cycle})
                results.append(metrics)
    if not results:
        split = int(len(df) * 0.8)
        train, test = df.iloc[:split], df.iloc[split:]
        metrics, _ = score_fold(train, test, threshold_ms, probability_threshold)
        metrics.update({"train_cycles": ["earlier-80%"], "test_cycle": "newest-20%"})
        results.append(metrics)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="CSV file or folder containing separate load-cycle CSVs")
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "./model"))
    p.add_argument("--horizon-s", type=float, default=30.0)
    p.add_argument("--window-s", type=float, default=10.0)
    p.add_argument("--threshold-ms", type=float, default=500.0)
    p.add_argument("--probability-threshold", type=float, default=0.70)
    p.add_argument("--min-baseline-improvement", type=float, default=0.40,
                   help="fractional MAE improvement required before the model is promotable")
    args = p.parse_args()

    data = add_target(load_data(args.train), args.horizon_s, args.window_s, args.threshold_ms)
    if len(data) < 80:
        raise ValueError(f"Need more load data after target construction; only {len(data)} rows remain")

    folds = walk_forward(data, args.threshold_ms, args.probability_threshold)
    mean_mae = float(np.mean([f["mae_ms"] for f in folds]))
    mean_baseline = float(np.mean([f["baseline_mae_ms"] for f in folds]))
    improvement = 1.0 - mean_mae / mean_baseline if mean_baseline else 0.0
    mean_recall = float(np.mean([f["breach_recall"] for f in folds]))
    mean_precision = float(np.mean([f["breach_precision"] for f in folds]))
    promotable = improvement >= args.min_baseline_improvement and mean_recall >= 0.90 and mean_precision >= 0.70

    final_bundle = fit_bundle(new_bundle(), data)
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(final_bundle, os.path.join(args.model_dir, "model.joblib"))
    meta = {
        "model_family": "gradient_boosted_quantile_plus_classifier",
        "features": FEATURES,
        "predicts": "change_from_recent_latency",
        "horizon_s": args.horizon_s,
        "window_s": args.window_s,
        "threshold_ms": args.threshold_ms,
        "probability_threshold": args.probability_threshold,
        "trained_rows": int(len(data)),
        "walk_forward": folds,
        "summary": {
            "mean_mae_ms": mean_mae,
            "mean_baseline_mae_ms": mean_baseline,
            "baseline_improvement_fraction": improvement,
            "mean_breach_recall": mean_recall,
            "mean_breach_precision": mean_precision,
            "promotable": promotable,
        },
    }
    with open(os.path.join(args.model_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta["summary"], indent=2))
    if not promotable:
        print("[DO NOT PROMOTE] Model did not meet the ideal-version quality gates.")
    else:
        print("[PROMOTABLE] Model passed baseline, recall and precision gates.")


if __name__ == "__main__":
    main()
