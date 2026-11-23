"""ml_challenge.py
Utility functions for the Amazon ML Challenge notebook.
Provides outlier detection, data trimming, scaling, model building, training and evaluation.
The functions are pure and testable without external files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

# Scikit‑learn and TensorFlow imports are optional for environments where they are not installed.
# They are imported lazily inside the functions that need them.

def detect_outliers_iqr(data: np.ndarray, iqr_factor: float = 1.5) -> np.ndarray:
    """Return a 1‑D array of values that are outliers according to the IQR rule.

    Parameters
    ----------
    data: np.ndarray
        2‑D numeric array where rows are observations.
    iqr_factor: float, default 1.5
        Multiplier for the inter‑quartile range.

    Returns
    -------
    np.ndarray
        1‑D array of outlier values (flattened).
    """
    if data.size == 0:
        return np.array([])
    if iqr_factor < 0:
        raise ValueError("iqr_factor must be non‑negative")
    # Compute per‑column quartiles
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    upper = q3 + iqr_factor * iqr
    lower = q1 - iqr_factor * iqr
    mask = (data > upper) | (data < lower)
    # Return the values that are outliers, flattened to 1‑D
    return data[mask]


def remove_outliers_iqr(data: np.ndarray, iqr_factor: float = 1.5) -> np.ndarray:
    """Remove rows that contain any outlier according to the IQR rule.

    Parameters
    ----------
    data: np.ndarray
        2‑D numeric array.
    iqr_factor: float, default 1.5
        Multiplier for the inter‑quartile range.

    Returns
    -------
    np.ndarray
        Cleaned array with outlier rows removed.
    """
    if data.size == 0:
        return data.copy()
    if iqr_factor < 0:
        raise ValueError("iqr_factor must be non‑negative")
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    upper = q3 + iqr_factor * iqr
    lower = q1 - iqr_factor * iqr
    # Keep rows where *all* columns are within bounds
    mask = np.all((data >= lower) & (data <= upper), axis=1)
    return data[mask]


def trim_percentiles(df: pd.DataFrame, lower: float = 0.1, upper: float = 0.9) -> pd.DataFrame:
    """Trim rows whose ``PRODUCT_LENGTH`` is outside the given percentile range.

    Parameters
    ----------
    df: pd.DataFrame
        Must contain a ``PRODUCT_LENGTH`` column.
    lower, upper: float
        Percentile bounds (0‑1).
    """
    if "PRODUCT_LENGTH" not in df.columns:
        raise KeyError("DataFrame must contain 'PRODUCT_LENGTH' column")
    if not (0 <= lower < upper <= 1):
        raise ValueError("Invalid percentile bounds")
    low_val = df["PRODUCT_LENGTH"].quantile(lower)
    high_val = df["PRODUCT_LENGTH"].quantile(upper)
    return df[(df["PRODUCT_LENGTH"] > low_val) & (df["PRODUCT_LENGTH"] < high_val)].copy()


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into train/test sets.

    Returns X_train, X_test, y_train, y_test.
    """
    from sklearn.model_selection import train_test_split

    if "PRODUCT_LENGTH" not in df.columns:
        raise KeyError("DataFrame must contain 'PRODUCT_LENGTH' column")
    X = df.drop(columns=["PRODUCT_LENGTH"])
    y = df["PRODUCT_LENGTH"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[object, np.ndarray, np.ndarray]:
    """Scale features using RobustScaler.

    Returns the scaler instance, scaled train array, scaled test array.
    """
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def build_model(input_dim: int) -> "tf.keras.Model":
    """Build the dense neural network used in the notebook.

    Parameters
    ----------
    input_dim: int
        Number of input features.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_model(model: "tf.keras.Model", X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = 20, verbose: int = 0):
    """Fit the model and return the training history."""
    return model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=verbose)


def evaluate_model(model: "tf.keras.Model", X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float]:
    """Return MAE, MAPE and the challenge score.

    Score = max(0, 100 * (1 - MAPE)).
    """
    from sklearn import metrics
    y_pred = model.predict(X_test).flatten()
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
    score = max(0.0, 100.0 * (1.0 - mape))
    return mae, mape, score


def predict(model: "tf.keras.Model", scaler: object, X: pd.DataFrame) -> np.ndarray:
    """Scale ``X`` with ``scaler`` and return model predictions as a 1‑D array."""
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled).flatten()

# The module can be executed as a script for a quick sanity check.
if __name__ == "__main__":
    # Simple synthetic example
    import numpy as np
    import pandas as pd
    # Create a tiny dataset
    df = pd.DataFrame({
        "PRODUCT_TYPE_ID": [1, 2, 1, 2, 3, 3],
        "PRODUCT_LENGTH": [10, 20, 12, 22, 30, 31]
    })
    trimmed = trim_percentiles(df)
    X_train, X_test, y_train, y_test = split_data(trimmed)
    scaler, Xtr_s, Xte_s = scale_data(X_train, X_test)
    model = build_model(input_dim=Xtr_s.shape[1])
    train_model(model, Xtr_s, y_train.values, Xte_s, y_test.values, epochs=1, verbose=1)
    mae, mape, score = evaluate_model(model, Xte_s, y_test.values)
    print(f"MAE={mae:.2f}, MAPE={mape:.2%}, Score={score:.2f}")
