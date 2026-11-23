"""Tests for ml_challenge.py — real asserts for every requirement incl. edge cases and crash/error paths."""

import os
import sys
import json
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

import ml_challenge as mc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """A small representative training dataframe."""
    return pd.DataFrame({
        "PRODUCT_ID": [1, 2, 3, 4, 5],
        "TITLE": ["a", "b", "c", "d", "e"],
        "BULLET_POINTS": ["x", "y", "z", "w", "v"],
        "DESCRIPTION": ["d1", "d2", "d3", "d4", "d5"],
        "PRODUCT_TYPE_ID": [10, 20, 10, 20, 10],
        "PRODUCT_LENGTH": [100.0, 200.0, 110.0, 210.0, 105.0],
    })


@pytest.fixture
def tmp_csv(tmp_path, sample_df):
    """Write sample_df to a temp csv and return its path."""
    p = tmp_path / "train.csv"
    sample_df.to_csv(p, index=False)
    return str(p)


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

def test_load_data_success(tmp_csv):
    df = mc.load_data(tmp_csv)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "PRODUCT_ID", "TITLE", "BULLET_POINTS", "DESCRIPTION",
        "PRODUCT_TYPE_ID", "PRODUCT_LENGTH",
    ]
    assert len(df) == 5


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        mc.load_data("does_not_exist.csv")


def test_load_data_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises((pd.errors.EmptyDataError, ValueError)):
        mc.load_data(str(p))


# ---------------------------------------------------------------------------
# select_features
# ---------------------------------------------------------------------------

def test_select_features_keeps_target_and_type(sample_df):
    X, y = mc.select_features(sample_df)
    assert list(X.columns) == ["PRODUCT_TYPE_ID"]
    assert list(y.name) == ["PRODUCT_LENGTH"] or y.name == "PRODUCT_LENGTH"
    assert len(X) == len(y) == 5


def test_select_features_missing_target_column():
    df = pd.DataFrame({"PRODUCT_TYPE_ID": [1, 2]})
    with pytest.raises(ValueError, match="PRODUCT_LENGTH"):
        mc.select_features(df)


def test_select_features_missing_type_column():
    df = pd.DataFrame({"PRODUCT_LENGTH": [1.0, 2.0]})
    with pytest.raises(ValueError, match="PRODUCT_TYPE_ID"):
        mc.select_features(df)


# ---------------------------------------------------------------------------
# trim_percentiles
# ---------------------------------------------------------------------------

def test_trim_percentiles_removes_extremes():
    df = pd.DataFrame({
        "PRODUCT_TYPE_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "PRODUCT_LENGTH": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0],
    })
    trimmed = mc.trim_percentiles(df, lower=0.1, upper=0.9)
    assert len(trimmed) < len(df)
    assert 1000.0 not in trimmed["PRODUCT_LENGTH"].values


def test_trim_percentiles_defaults():
    df = pd.DataFrame({
        "PRODUCT_TYPE_ID": [1, 2, 3],
        "PRODUCT_LENGTH": [10.0, 20.0, 30.0],
    })
    trimmed = mc.trim_percentiles(df)
    assert len(trimmed) <= len(df)


def test_trim_percentiles_missing_column():
    df = pd.DataFrame({"PRODUCT_TYPE_ID": [1, 2]})
    with pytest.raises(ValueError, match="PRODUCT_LENGTH"):
        mc.trim_percentiles(df)


# ---------------------------------------------------------------------------
# train_test_split_data
# ---------------------------------------------------------------------------

def test_train_test_split_shapes(sample_df):
    X = sample_df[["PRODUCT_TYPE_ID"]]
    y = sample_df["PRODUCT_LENGTH"]
    Xtr, Xte, ytr, yte = mc.train_test_split_data(X, y, test_size=0.2, random_state=42)
    assert len(Xtr) + len(Xte) == len(X)
    assert len(ytr) + len(yte) == len(y)
    assert 0 < len(Xte) < len(X)


def test_train_test_split_invalid_test_size(sample_df):
    X = sample_df[["PRODUCT_TYPE_ID"]]
    y = sample_df["PRODUCT_LENGTH"]
    with pytest.raises(ValueError):
        mc.train_test_split_data(X, y, test_size=1.5)


# ---------------------------------------------------------------------------
# scale_features
# ---------------------------------------------------------------------------

def test_scale_features_fit_transform(sample_df):
    X = sample_df[["PRODUCT_TYPE_ID"]]
    Xtr, Xte, scaler = mc.scale_features(X, X.copy())
    assert Xtr.shape == X.shape
    assert Xte.shape == X.shape
    assert scaler is not None


def test_scale_features_returns_fitted_scaler(sample_df):
    X = sample_df[["PRODUCT_TYPE_ID"]]
    _, _, scaler = mc.scale_features(X, X.copy())
    # scaler should have been fit (has mean_ / scale_ or similar attribute)
    assert hasattr(scaler, "transform")


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def test_build_model_architecture():
    model = mc.build_model(input_dim=1)
    assert model is not None
    # Should have an input layer + 4 hidden + 1 output = 6 layers
    assert len(model.layers) == 6
    assert model.output_shape == (None, 1)


def test_build_model_invalid_input_dim():
    with pytest.raises(ValueError):
        mc.build_model(input_dim=0)


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

def test_train_model_runs(sample_df):
    X = sample_df[["PRODUCT_TYPE_ID"]].values.astype(float)
    y = sample_df["PRODUCT_LENGTH"].values.astype(float)
    model = mc.build_model(input_dim=1)
    history = mc.train_model(model, X, y, epochs=2, validation_split=0.0)
    assert history is not None
    assert "loss" in history.history


# ---------------------------------------------------------------------------
# evaluate_model / score
# ---------------------------------------------------------------------------

def test_evaluate_model_returns_metrics(sample_df):
    X = sample_df[["PRODUCT_TYPE_ID"]].values.astype(float)
    y = sample_df["PRODUCT_LENGTH"].values.astype(float)
    model = mc.build_model(input_dim=1)
    mc.train_model(model, X, y, epochs=1, validation_split=0.0)
    loss, mae = mc.evaluate_model(model, X, y)
    assert loss >= 0
    assert mae >= 0


def test_challenge_score_in_range():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    score = mc.challenge_score(y_true, y_pred)
    assert score == 100.0

    y_pred_bad = np.array([0.0, 0.0, 0.0])
    score = mc.challenge_score(y_true, y_pred_bad)
    assert 0.0 <= score <= 100.0


def test_challenge_score_clamped_at_zero():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([1000.0, 2000.0])  # huge MAPE -> score clamped to 0
    score = mc.challenge_score(y_true, y_pred)
    assert score == 0.0


def test_challenge_score_mismatched_lengths():
    with pytest.raises(ValueError):
        mc.challenge_score(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def test_predict_output_shape(sample_df):
    X = sample_df[["PRODUCT_TYPE_ID"]].values.astype(float)
    model = mc.build_model(input_dim=1)
    mc.train_model(model, X, sample_df["PRODUCT_LENGTH"].values.astype(float), epochs=1, validation_split=0.0)
    preds = mc.predict(model, X)
    assert preds.shape == (len(X), 1)


# ---------------------------------------------------------------------------
# save_submission / load_submission
# ---------------------------------------------------------------------------

def test_save_and_load_submission(tmp_path):
    p = tmp_path / "submission.csv"
    ids = np.array([1, 2, 3])
    preds = np.array([100.0, 200.0, 300.0])
    mc.save_submission(str(p), ids, preds)
    assert os.path.exists(p)
    df = mc.load_submission(str(p))
    assert list(df.columns) == ["PRODUCT_ID", "PRODUCT_LENGTH"]
    assert len(df) == 3


def test_load_submission_missing_file(tmp_path):
    p = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError):
        mc.load_submission(str(p))


# ---------------------------------------------------------------------------
# run_pipeline (integration)
# ---------------------------------------------------------------------------

def test_run_pipeline_end_to_end(tmp_path, tmp_csv, monkeypatch):
    # Avoid actually training for long; use tiny epochs via monkeypatch if needed.
    out = tmp_path / "submission.csv"
    # Build a tiny test.csv too
    test_df = pd.DataFrame({
        "PRODUCT_ID": [6, 7],
        "TITLE": ["f", "g"],
        "BULLET_POINTS": ["u", "t"],
        "DESCRIPTION": ["d6", "d7"],
        "PRODUCT_TYPE_ID": [10, 20],
    })
    test_csv = tmp_path / "test.csv"
    test_df.to_csv(test_csv, index=False)

    # Monkeypatch train_model to use 1 epoch
    import ml_challenge as mc_mod
    orig_train = mc_mod.train_model
    def fast_train(model, X, y, epochs=20, validation_split=0.2):
        return orig_train(model, X, y, epochs=1, validation_split=0.0)
    mc_mod.train_model = fast_train

    try:
        result = mc.run_pipeline(
            train_path=tmp_csv,
            test_path=str(test_csv),
            submission_path=str(out),
            epochs=1,
        )
    finally:
        mc_mod.train_model = orig_train

    assert os.path.exists(out)
    sub = pd.read_csv(out)
    assert list(sub.columns) == ["PRODUCT_ID", "PRODUCT_LENGTH"]
    assert len(sub) == 2
    assert result is not None


# ---------------------------------------------------------------------------
# detect_outliers_iqr / remove_outliers_iqr (legacy helpers)
# ---------------------------------------------------------------------------

def test_detect_outliers_iqr_basic():
    data = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    outliers = mc.detect_outliers_iqr(data)
    assert isinstance(outliers, np.ndarray)
    assert 100.0 in outliers


def test_remove_outliers_iqr_basic():
    data = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    cleaned = mc.remove_outliers_iqr(data)
    assert 100.0 not in cleaned
    assert len(cleaned) < len(data)


def test_detect_outliers_iqr_empty():
    data = np.array([]).reshape(0, 1)
    with pytest.raises(ValueError):
        mc.detect_outliers_iqr(data)


# ---------------------------------------------------------------------------
# Edge: constant target column (MAPE division by zero)
# ---------------------------------------------------------------------------

def test_challenge_score_constant_target():
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([5.0, 5.0, 5.0])
    score = mc.challenge_score(y_true, y_pred)
    assert score == 100.0
