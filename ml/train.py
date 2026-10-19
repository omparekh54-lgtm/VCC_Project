"""
SageMaker training entry point.

Trains a RandomForestRegressor that forecasts latency a little ahead:

    (cpu_usage, active_connections, lag_ms, recent_latency_ms) at time t
        -> mean response_time_ms of requests arriving in [t + H, t + H + W)

H (--horizon-s, default 30) is how far ahead the watchdog looks; W
(--window-s, default 10) smooths the target over a short window, because a
single request's latency is too noisy to forecast. With --horizon-s 0 the
model estimates current latency instead.

recent_latency_ms is the mean latency of requests that finished in the 10 s
before t -- what /metrics reports. The log doesn't store it; it is rebuilt
from each row's arrival time + latency, the same definition the app uses.

The forest learns the *change* from recent latency to future latency, and the
forecast is recent latency + that change (see forecast()). Learning the
level directly caps every forecast at the highest latency seen in training,
which is exactly where the watchdog needs to be right.

The feature list and target form are saved in model_meta.json, which
inference.py and the app read; the Lambda forwards /metrics unchanged.

SageMaker's scikit-learn container invokes this as roughly:
    python train.py --n-estimators 200 --max-depth 12
with training data mounted at $SM_CHANNEL_TRAIN and the model expected to be
written to $SM_MODEL_DIR. It also runs fine locally for a quick sanity check:
    python train.py --train ./data --model-dir ./model
"""

import argparse
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["cpu_usage", "active_connections", "lag_ms", "recent_latency_ms"]
LATENCY = "response_time_ms"
TARGET = "future_latency_ms"
RECENT_WINDOW_S = 10.0  # must match RECENT_WINDOW_S in backend/app.py


def load_training_data(train_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(train_dir, "*.csv")) if os.path.isdir(train_dir) else [train_dir]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {train_dir}")
    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["timestamp", "cpu_usage", "active_connections", "lag_ms", LATENCY])
    # Rows are a time series; keep them in order so targets and splits are by time.
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df["recent_latency_ms"] = recent_latency_at(df, df["timestamp"].to_numpy(dtype=float))
    return df


def recent_latency_at(df: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    """Mean latency of requests that finished in the RECENT_WINDOW_S before each time; 0 if none."""
    finish = df["timestamp"].to_numpy(dtype=float) + df[LATENCY].to_numpy(dtype=float) / 1000
    order = np.argsort(finish, kind="stable")
    out = window_mean(finish[order], df[LATENCY].to_numpy(dtype=float)[order], times - RECENT_WINDOW_S, times)
    return np.nan_to_num(out, nan=0.0)


def window_mean(ts: np.ndarray, values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Mean of `values` whose timestamp falls in [start, end), per row; NaN if none."""
    csum = np.concatenate([[0.0], np.cumsum(values)])
    lo = np.searchsorted(ts, starts, side="left")
    hi = np.searchsorted(ts, ends, side="left")
    n = hi - lo
    out = np.full(len(starts), np.nan)
    has = n > 0
    out[has] = (csum[hi[has]] - csum[lo[has]]) / n[has]
    return out


def add_targets(df: pd.DataFrame, horizon_s: float, window_s: float) -> pd.DataFrame:
    """
    Adds TARGET (latency H seconds ahead). Drops rows whose future window runs
    past the end of the data or contains no requests.
    """
    if horizon_s < 0 or window_s <= 0:
        raise ValueError("horizon_s must be nonnegative and window_s must be positive")
    if df.empty:
        raise ValueError("No valid training rows; collect a longer load test")
    ts = df["timestamp"].to_numpy(dtype=float)
    rt = df[LATENCY].to_numpy(dtype=float)
    df = df.copy()
    df[TARGET] = window_mean(ts, rt, ts + horizon_s, ts + horizon_s + window_s)
    complete = ts + horizon_s + window_s <= ts[-1]
    return df[complete & df[TARGET].notna()].reset_index(drop=True)


def time_split(df: pd.DataFrame, test_fraction: float, horizon_s: float, window_s: float):
    """
    Newest `test_fraction` of rows are the test set. Training rows whose target
    window reaches into the test period are dropped, so no test-period latency
    leaks into training through the shifted target.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    split = int(len(df) * (1 - test_fraction))
    if split == 0 or split >= len(df):
        raise ValueError("Not enough rows for a time-ordered holdout; collect a longer load test")
    split_t = df["timestamp"].iloc[split]
    train = df.iloc[:split]
    train = train[train["timestamp"] + horizon_s + window_s <= split_t]
    return train, df.iloc[split:]


def fit(model, df: pd.DataFrame):
    """Fit the forest on the change from recent latency to TARGET."""
    return model.fit(df[FEATURES].values, (df[TARGET] - df["recent_latency_ms"]).values)


def forecast(model, X: np.ndarray) -> np.ndarray:
    """Forecast latency for rows of FEATURES: recent latency + predicted change, never below 0."""
    X = np.asarray(X, dtype=float)
    return np.maximum(0.0, X[:, FEATURES.index("recent_latency_ms")] + model.predict(X))


def new_model(args) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1,
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Hyperparameters and target definition, shared with compare_scaling.py."""
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--horizon-s", type=float, default=30.0)
    parser.add_argument("--window-s", type=float, default=10.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    # SageMaker-provided paths (defaults let this run locally too)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./data"))
    args = parser.parse_args()

    df = add_targets(load_training_data(args.train), args.horizon_s, args.window_s)
    train, test = time_split(df, args.test_fraction, args.horizon_s, args.window_s)
    if len(train) < 50 or len(test) < 10:
        raise ValueError(
            f"Not enough rows after building {args.horizon_s:g}s-ahead targets "
            f"(train={len(train)}, test={len(test)}). Collect a longer load test."
        )

    # Hold out the most recent rows instead of a random 20%. Neighbouring
    # requests are near-duplicates (same load, same moment), so a random split
    # puts copies of test rows in the training set and inflates the scores.
    model = fit(new_model(args), train)
    y_test = test[TARGET].values
    preds = forecast(model, test[FEATURES].values)
    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)

    # Naive baseline: "latency H seconds from now will equal latency right now".
    # A forecaster that can't beat this isn't adding anything.
    baseline_mae = mean_absolute_error(y_test, test["recent_latency_ms"].values)

    print(f"[metrics] forecasting mean latency {args.horizon_s:g}s ahead "
          f"(over a {args.window_s:g}s window)")
    print(f"[metrics] time-ordered holdout ({len(y_test)} newest rows): "
          f"MAE={mae:.2f}ms RMSE={rmse:.2f}ms R2={r2:.4f}")
    print(f"[metrics] naive baseline (latency stays as it is now): MAE={baseline_mae:.2f}ms")
    if mae >= baseline_mae:
        print("[WARNING] the model does NOT beat the naive baseline on held-out data, so its "
              "forecasts add nothing over current latency. Usually this means too little or "
              "too uniform data: collect several full load cycles "
              "(scripts/collect_demo_data.py --cycles 3) and retrain before relying on it.")
    importances = {f: round(float(v), 3) for f, v in zip(FEATURES, model.feature_importances_)}
    print(f"[metrics] feature_importances={importances}")

    # The score above is the honest estimate; the shipped model then learns
    # from every row, including the most recent load pattern.
    model = fit(new_model(args), df)

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    meta = {
        "features": FEATURES,
        # inference.py / app.py: forecast = recent_latency_ms + model output
        "predicts": "change_from_recent_latency",
        "horizon_s": args.horizon_s,
        "window_s": args.window_s,
        "trained_rows": len(df),
        "holdout": {"rows": len(y_test), "mae_ms": mae, "rmse_ms": rmse, "r2": r2,
                    "baseline_mae_ms": baseline_mae},
    }
    with open(os.path.join(args.model_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
