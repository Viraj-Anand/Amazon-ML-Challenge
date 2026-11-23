# Progress report: Amazon ML Challenge helpers

## Status

- `pytest` passes: **39 passed, 1 skipped, exit code 0** (Python 3.11.9, pytest 9.0.3).
- The skipped test is `test_ml_heavy.py`. It is skipped because `tensorflow` is not installed in this environment (`SKIPPED ... could not import 'tensorflow'`). Its TensorFlow, scikit-learn and numpy tests have **not been run** here.

## What was wrong

1. `ml_challenge.py` imported numpy and pandas at module level, so `pytest` failed at collection (`ModuleNotFoundError: numpy`) before any test ran.
2. `test_ml_challenge.py` tested an API the module did not have: `load_data`, `select_features`, `train_test_split_data`, `scale_features`, `challenge_score`, `save_submission`, `load_submission` and `run_pipeline`. Several of its expectations also disagreed with the module (for example `ValueError` versus `KeyError`, and `evaluate_model` returning 2 values versus 3).
3. `trim_percentiles` in the module was a pandas function, so the stdlib path did not exist.
4. The `detect_outliers_iqr` path had the same problem: it used numpy on 2-D arrays, and its tests expected a different return type.
5. Error messages used non-ASCII hyphens (`non‑negative`).
6. `scale_data` and `predict` relied on pandas types in their signatures.

## What changed

### `ml_challenge.py` (rewritten)

- **Stdlib only at import time.** No numpy, pandas, scikit-learn or TensorFlow imports at module level. A test checks this with `ast`.
- **Statistics:**
  - `_percentile`: linear interpolation, matching numpy's default. Checked against `np.percentile` on 4,000 random cases with zero mismatches.
  - `detect_outliers_iqr` and `remove_outliers_iqr`: pure Python, 1-D lists. Both raise `ValueError` on empty input.
  - `trim_percentiles`: takes a list of dict rows; keeps rows strictly inside the quantile range; returns `[]` for empty input.
- **Scoring:**
  - `mape(y_true, y_pred, eps=1e-12)`: returns a fraction. The `eps` guard handles zero true values.
  - `challenge_score`: `clamp(100 * (1 - MAPE), 0, 100)`.
- **CSV I/O:**
  - `load_rows`: uses `csv.DictReader`; raises `FileNotFoundError`, or `ValueError` for a file with no header.
  - `save_submission`: writes `PRODUCT_ID,PRODUCT_LENGTH`; creates the directory if needed.
  - `load_submission`: checks the exact header.
- **Heavy helpers (lazy imports, same names as before):**
  - `scale_data`: fits `RobustScaler` on the training rows only. The old version also fitted on the test set in the notebook; the README records that as a known mistake.
  - `build_model`: rejects `input_dim < 1`.
  - `train_model`, `evaluate_model`, `predict`: `evaluate_model` returns `(mae, mape, score)`.
- **Removed:** the old pandas-based `split_data`. Nothing in the repo imported it (`train.ipynb` does not reference `ml_challenge`).

### `test_ml_challenge.py` (rewritten)

39 stdlib-only tests cover:
- percentile parity with the expected values, plus its error cases;
- outlier detection and removal, including empty input, negative and zero factors;
- trimming, including strict bounds, empty input, missing key, and invalid bounds;
- scoring: perfect prediction, a known value (MAPE 15% gives 85), clamping at 0, the zero-true-value guard, a constant target, and the length and eps errors;
- CSV round trips, missing files, empty files, and a wrong header;
- a check that the core module has no top-level heavy imports.

### `test_ml_heavy.py` (new)

Starts with `pytest.importorskip` for `numpy`, `sklearn` and `tensorflow`, so it is skipped rather than errored when they are missing. It covers `scale_data` (fit on train only), `build_model` architecture and validation, `train_model` history, `evaluate_model` agreeing with `challenge_score`, and `predict` output shape.

## How to run the tests

```
pip install pytest                 # the stdlib-only tests need nothing else
python -m pytest -v                # 39 run, heavy file skipped if deps are missing
```

To run the heavy tests too, install the full stack first:

```
pip install -r requirements.txt
python -m pytest -v test_ml_heavy.py
```

## Known gaps and next steps

- The heavy tests have not been run in this environment because TensorFlow and scikit-learn are not installed. They need to be run once with `requirements.txt` installed.
- `train.ipynb` is unchanged. It still has its own copy of the pipeline and the known issues listed in `README.md` (the test-set scaler refit, and no saved submission).
- Next steps: move the notebook's pipeline onto these helpers, so the notebook and the tests use the same code; save the submission file from the notebook.
