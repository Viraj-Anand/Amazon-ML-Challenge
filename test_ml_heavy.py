# Tests for the numpy / scikit-learn / TensorFlow helpers in ml_challenge.py.
# Skipped (not errored) when those packages are not installed.
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("tensorflow")

import ml_challenge as mc  # noqa: E402


@pytest.fixture(scope="module")
def tiny_xy():
    X = np.array([[10.0], [20.0], [10.0], [20.0], [30.0], [30.0]])
    y = np.array([100.0, 200.0, 110.0, 210.0, 300.0, 310.0])
    return X, y


def test_scale_data_fits_on_train_only(tiny_xy):
    X, _ = tiny_xy
    X_train, X_test = X[:4], np.array([[1000.0]])
    scaler, Xtr_s, Xte_s = mc.scale_data(X_train, X_test)
    assert Xtr_s.shape == X_train.shape
    assert Xte_s.shape == X_test.shape
    # The outlier test value must not change the fitted centre or scale.
    expected_center = np.median(X_train, axis=0)
    assert np.allclose(scaler.center_, expected_center)
    assert Xte_s[0, 0] > 10  # large, because the scaler never saw 1000 during fit


def test_build_model_architecture():
    model = mc.build_model(input_dim=3)
    # Input + 4 hidden + 1 output layer. Keras reports the Input as a layer in some versions,
    # so count only the Dense layers, which is version-independent.
    dense = [layer for layer in model.layers if layer.__class__.__name__ == "Dense"]
    assert len(dense) == 5
    assert dense[-1].units == 1
    assert model.output_shape == (None, 1)


def test_build_model_rejects_invalid_input_dim():
    with pytest.raises(ValueError):
        mc.build_model(input_dim=0)


def test_train_model_returns_history(tiny_xy):
    X, y = tiny_xy
    model = mc.build_model(input_dim=1)
    history = mc.train_model(model, X, y, X, y, epochs=2, verbose=0)
    assert "loss" in history.history
    assert len(history.history["loss"]) == 2


def test_evaluate_model_matches_challenge_score(tiny_xy):
    X, y = tiny_xy
    model = mc.build_model(input_dim=1)
    mc.train_model(model, X, y, X, y, epochs=1, verbose=0)
    mae, mape, score = mc.evaluate_model(model, X, y)
    assert mae >= 0
    assert mape >= 0
    assert 0.0 <= score <= 100.0
    # The score must agree with the stdlib formula applied to the same predictions.
    preds = np.asarray(model.predict(X, verbose=0)).flatten().tolist()
    assert score == pytest.approx(mc.challenge_score(y.tolist(), preds), rel=1e-6)


def test_predict_returns_one_value_per_row(tiny_xy):
    X, y = tiny_xy
    scaler, Xs, _ = mc.scale_data(X, X)
    model = mc.build_model(input_dim=1)
    mc.train_model(model, Xs, y, Xs, y, epochs=1, verbose=0)
    preds = mc.predict(model, scaler, X)
    assert preds.shape == (len(X),)
