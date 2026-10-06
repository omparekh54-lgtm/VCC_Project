"""
SageMaker training entry point.

Trains a RandomForestRegressor mapping
    (cpu_usage, active_connections, lag_ms) -> response_time_ms
using the CSV(s) produced by the Locust load test.

SageMaker's scikit-learn container invokes this as roughly:
    python train.py --n-estimators 200 --max-depth 12
with training data mounted at $SM_CHANNEL_TRAIN and the model expected to be
written to $SM_MODEL_DIR. It also runs fine locally for a quick sanity check:
    python train.py --train ./data --model-dir ./model
"""

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["cpu_usage", "active_connections", "lag_ms"]
TARGET = "response_time_ms"


def load_training_data(train_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(train_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {train_dir}")
    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=FEATURES + [TARGET])


def main() -> None:
    parser = argparse.ArgumentParser()
    # Hyperparameters (tune these from launch_training_job.py)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    # SageMaker-provided paths (defaults let this run locally too)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./data"))
    args = parser.parse_args()

    df = load_training_data(args.train)
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state
    )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"[metrics] MAE={mae:.2f}ms RMSE={rmse:.2f}ms R2={r2:.4f}")
    print(f"[metrics] feature_importances={dict(zip(FEATURES, model.feature_importances_))}")

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))


if __name__ == "__main__":
    main()
