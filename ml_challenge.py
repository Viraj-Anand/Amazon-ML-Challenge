"""ml_challenge.py - helpers for the Amazon ML Challenge (product length regression).

The core helpers are pure Python (stdlib only) so they can be tested anywhere.
The heavy ML dependencies (numpy, scikit-learn, TensorFlow) are imported lazily
inside the functions that need them, so importing this module never fails
because those packages are missing.
"""
from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Sequence

TARGET = "PRODUCT_LENGTH"
ID_COLUMN = "PRODUCT_ID"
TYPE_COLUMN = "PRODUCT_TYPE_ID"
EPS = 1e-12


# ---------------------------------------------------------------------------
# Statistics helpers (stdlib only)
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (matches numpy's default). q in [0, 100]."""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("cannot compute percentile of empty data")
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    pos = (n - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _iqr_bounds(values: Sequence[float], iqr_factor: float):
    if len(values) == 0:
        raise ValueError("data must not be empty")
    if iqr_factor < 0:
        raise ValueError("iqr_factor must be non-negative")
    s = sorted(float(v) for v in values)
    q1, q3 = _percentile(s, 25), _percentile(s, 75)
    iqr = q3 - q1
    return q1 - iqr_factor * iqr, q3 + iqr_factor * iqr


def detect_outliers_iqr(values: Sequence[float], iqr_factor: float = 1.5) -> List[float]:
    """Return the values outside the IQR fences. Raises ValueError on empty input."""
    lower, upper = _iqr_bounds(values, iqr_factor)
    return [float(v) for v in values if v < lower or v > upper]


def remove_outliers_iqr(values: Sequence[float], iqr_factor: float = 1.5) -> List[float]:
    """Return the values inside the IQR fences. Raises ValueError on empty input."""
    lower, upper = _iqr_bounds(values, iqr_factor)
    return [float(v) for v in values if lower <= v <= upper]


def trim_percentiles(rows: List[Dict], lower: float = 0.1, upper: float = 0.9, key: str = TARGET) -> List[Dict]:
    """Keep rows whose `key` is strictly inside the (lower, upper) quantile range.

    `rows` is a list of dicts (for example from `load_rows`). Empty input gives [].
    """
    if not (0 <= lower < upper <= 1):
        raise ValueError("Invalid percentile bounds")
    if not rows:
        return []
    if any(key not in r for r in rows):
        raise ValueError(f"rows must contain '{key}'")
    s = sorted(float(r[key]) for r in rows)
    lo_v = _percentile(s, lower * 100)
    hi_v = _percentile(s, upper * 100)
    return [r for r in rows if lo_v < float(r[key]) < hi_v]


def mape(y_true: Sequence[float], y_pred: Sequence[float], eps: float = EPS) -> float:
    """Mean absolute percentage error, as a fraction (0.1 == 10%).

    Each term is |y - p| / max(|y|, eps); the eps guard avoids division by zero.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("cannot score empty data")
    if eps <= 0:
        raise ValueError("eps must be positive")
    errors = [
        abs(float(t) - float(p)) / max(abs(float(t)), eps)
        for t, p in zip(y_true, y_pred)
    ]
    return sum(errors) / len(errors)


def challenge_score(y_true: Sequence[float], y_pred: Sequence[float], eps: float = EPS) -> float:
    """Challenge score = clamp(100 * (1 - MAPE), 0, 100)."""
    return min(100.0, max(0.0, 100.0 * (1.0 - mape(y_true, y_pred, eps))))


# ---------------------------------------------------------------------------
# CSV I/O (stdlib only)
# ---------------------------------------------------------------------------

def load_rows(path: str) -> List[Dict[str, str]]:
    """Read a CSV file into a list of dict rows. Raises FileNotFoundError or ValueError."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"file has no header row: {path}")
        return list(reader)


def save_submission(path: str, ids: Sequence, predictions: Sequence[float]) -> None:
    """Write a submission CSV with columns PRODUCT_ID, PRODUCT_LENGTH."""
    if len(ids) != len(predictions):
        raise ValueError("ids and predictions must have the same length")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([ID_COLUMN, TARGET])
        for pid, pred in zip(ids, predictions):
            writer.writerow([pid, float(pred)])


def load_submission(path: str) -> List[Dict[str, str]]:
    """Read a submission CSV and check that it has the expected columns."""
    rows = load_rows(path)
    with open(path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh), [])
    if header != [ID_COLUMN, TARGET]:
        raise ValueError(f"submission header must be {[ID_COLUMN, TARGET]}, got {header}")
    return rows


# ---------------------------------------------------------------------------
# Heavy ML helpers (lazy imports: numpy / scikit-learn / TensorFlow)
# ---------------------------------------------------------------------------

def scale_data(X_train, X_test):
    """Fit a RobustScaler on X_train only and transform both sets.

    Returns (scaler, X_train_scaled, X_test_scaled). Fitting on the training
    rows only avoids leaking test-set statistics into the model.
    """
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def build_model(input_dim: int):
    """Build the dense network used in the notebook (4 hidden layers, 1 output)."""
    if input_dim < 1:
        raise ValueError("input_dim must be at least 1")
    import tensorflow as tf
    from tensorflow.keras import layers

    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs: int = 20, verbose: int = 0):
    """Fit the model and return the Keras History object."""
    return model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=verbose)


def evaluate_model(model, X_test, y_test):
    """Return (MAE, MAPE, challenge score) on the given data."""
    import numpy as np

    y_pred = np.asarray(model.predict(X_test)).flatten()
    y_true = np.asarray(y_test, dtype=float).flatten()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape_val = mape(y_true.tolist(), y_pred.tolist())
    score = min(100.0, max(0.0, 100.0 * (1.0 - mape_val)))
    return mae, mape_val, score


def predict(model, scaler, X):
    """Scale X with a fitted scaler and return predictions as a 1-D array."""
    import numpy as np

    X_scaled = scaler.transform(X)
    return np.asarray(model.predict(X_scaled)).flatten()


# ---------------------------------------------------------------------------
# Script entry point: a quick stdlib-only sanity check (no ML libraries needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = [
        {"PRODUCT_TYPE_ID": "1", "PRODUCT_LENGTH": "10"},
        {"PRODUCT_TYPE_ID": "2", "PRODUCT_LENGTH": "20"},
        {"PRODUCT_TYPE_ID": "1", "PRODUCT_LENGTH": "12"},
        {"PRODUCT_TYPE_ID": "2", "PRODUCT_LENGTH": "22"},
        {"PRODUCT_TYPE_ID": "3", "PRODUCT_LENGTH": "30"},
        {"PRODUCT_TYPE_ID": "3", "PRODUCT_LENGTH": "31"},
    ]
    kept = trim_percentiles(demo)
    truths = [float(r[TARGET]) for r in kept]
    print(f"kept {len(kept)} of {len(demo)} rows")
    print(f"score of predicting the truth = {challenge_score(truths, truths):.2f}")
