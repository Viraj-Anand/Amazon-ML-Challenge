"""Stdlib-only tests for ml_challenge.py.

These tests need no numpy, pandas, scikit-learn or TensorFlow. The tests that
do need those packages live in test_ml_heavy.py and are skipped when they are
missing.
"""

import math

import pytest

import ml_challenge as mc


SAMPLE_CSV = (
    "PRODUCT_ID,TITLE,BULLET_POINTS,DESCRIPTION,PRODUCT_TYPE_ID,PRODUCT_LENGTH\n"
    "1,a,x,d1,10,100.0\n"
    "2,b,y,d2,20,200.0\n"
    "3,c,z,d3,10,110.0\n"
    "4,d,w,d4,20,210.0\n"
    "5,e,v,d5,10,105.0\n"
)


# ---------------------------------------------------------------------------
# _percentile / outliers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q, expected", [(0, 1.0), (25, 1.75), (50, 2.5), (75, 3.25), (100, 4.0)])
def test_percentile_matches_numpy_linear_interpolation(q, expected):
    assert mc._percentile([1.0, 2.0, 3.0, 4.0], q) == pytest.approx(expected)


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        mc._percentile([], 50)


def test_percentile_out_of_range_raises():
    with pytest.raises(ValueError):
        mc._percentile([1.0, 2.0], 101)


def test_detect_outliers_iqr_finds_extreme_value():
    assert mc.detect_outliers_iqr([1.0, 2.0, 3.0, 4.0, 100.0]) == [100.0]


def test_remove_outliers_iqr_drops_extreme_value():
    cleaned = mc.remove_outliers_iqr([1.0, 2.0, 3.0, 4.0, 100.0])
    assert 100.0 not in cleaned
    assert cleaned == [1.0, 2.0, 3.0, 4.0]


def test_detect_and_remove_are_complementary():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 50.0, -40.0]
    outliers = mc.detect_outliers_iqr(data)
    kept = mc.remove_outliers_iqr(data)
    assert sorted(outliers + kept) == sorted(data)


def test_outliers_empty_input_raises():
    with pytest.raises(ValueError):
        mc.detect_outliers_iqr([])
    with pytest.raises(ValueError):
        mc.remove_outliers_iqr([])


def test_outliers_negative_factor_raises():
    with pytest.raises(ValueError):
        mc.detect_outliers_iqr([1.0, 2.0], iqr_factor=-1)


def test_outliers_zero_factor_flags_values_outside_quartiles():
    # With factor 0 the fences are Q1 and Q3, so the middle values are kept.
    assert mc.remove_outliers_iqr([1.0, 2.0, 3.0, 4.0, 5.0], iqr_factor=0) == [2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# trim_percentiles
# ---------------------------------------------------------------------------

def _rows(lengths):
    return [{"PRODUCT_TYPE_ID": str(i), "PRODUCT_LENGTH": str(v)} for i, v in enumerate(lengths)]


def test_trim_percentiles_removes_extremes():
    rows = _rows([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000])
    trimmed = mc.trim_percentiles(rows, lower=0.1, upper=0.9)
    lengths = [float(r["PRODUCT_LENGTH"]) for r in trimmed]
    assert len(trimmed) < len(rows)
    assert 1000.0 not in lengths
    assert 1.0 not in lengths


def test_trim_percentiles_keeps_rows_strictly_inside_bounds():
    rows = _rows([10.0, 20.0, 30.0])
    trimmed = mc.trim_percentiles(rows)
    # 10th percentile is 12.0 and 90th is 28.0, so only 20.0 is strictly inside.
    assert [float(r["PRODUCT_LENGTH"]) for r in trimmed] == [20.0]


def test_trim_percentiles_empty_returns_empty():
    assert mc.trim_percentiles([]) == []


def test_trim_percentiles_missing_key_raises():
    with pytest.raises(ValueError, match="PRODUCT_LENGTH"):
        mc.trim_percentiles([{"PRODUCT_TYPE_ID": "1"}])


@pytest.mark.parametrize("lower, upper", [(0.9, 0.1), (-0.1, 0.5), (0.2, 1.5), (0.5, 0.5)])
def test_trim_percentiles_invalid_bounds_raise(lower, upper):
    with pytest.raises(ValueError):
        mc.trim_percentiles(_rows([1, 2, 3]), lower=lower, upper=upper)


# ---------------------------------------------------------------------------
# challenge_score / mape
# ---------------------------------------------------------------------------

def test_challenge_score_perfect_prediction_is_100():
    assert mc.challenge_score([100.0, 200.0, 300.0], [100.0, 200.0, 300.0]) == 100.0


def test_challenge_score_known_value():
    # Errors are 10% and 20%, so MAPE is 15% and the score is 85.
    assert mc.challenge_score([100.0, 100.0], [110.0, 80.0]) == pytest.approx(85.0)


def test_challenge_score_clamped_at_zero():
    # MAPE is 99 (9900%), so the raw score is negative and must be clamped to 0.
    assert mc.challenge_score([10.0, 20.0], [1000.0, 2000.0]) == 0.0


def test_challenge_score_always_in_range():
    score = mc.challenge_score([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    assert 0.0 <= score <= 100.0


def test_challenge_score_zero_true_value_uses_eps_guard():
    score = mc.challenge_score([0.0], [0.0])
    assert score == 100.0
    assert math.isfinite(mc.challenge_score([0.0], [1.0]))


def test_challenge_score_constant_target():
    assert mc.challenge_score([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]) == 100.0


def test_challenge_score_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        mc.challenge_score([1.0, 2.0], [1.0])


def test_challenge_score_empty_raises():
    with pytest.raises(ValueError):
        mc.challenge_score([], [])


def test_challenge_score_bad_eps_raises():
    with pytest.raises(ValueError):
        mc.challenge_score([1.0], [1.0], eps=0)


def test_mape_is_fraction():
    assert mc.mape([100.0, 100.0], [110.0, 80.0]) == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# load_rows / save_submission / load_submission
# ---------------------------------------------------------------------------

def test_load_rows_reads_csv(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text(SAMPLE_CSV, encoding="utf-8")
    rows = mc.load_rows(str(p))
    assert len(rows) == 5
    assert list(rows[0].keys()) == [
        "PRODUCT_ID", "TITLE", "BULLET_POINTS", "DESCRIPTION",
        "PRODUCT_TYPE_ID", "PRODUCT_LENGTH",
    ]
    assert rows[0]["PRODUCT_LENGTH"] == "100.0"


def test_load_rows_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mc.load_rows(str(tmp_path / "does_not_exist.csv"))


def test_load_rows_empty_file_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        mc.load_rows(str(p))


def test_save_and_load_submission_round_trip(tmp_path):
    p = tmp_path / "out" / "submission.csv"
    mc.save_submission(str(p), [1, 2, 3], [100.0, 200.0, 300.5])
    rows = mc.load_submission(str(p))
    assert [r["PRODUCT_ID"] for r in rows] == ["1", "2", "3"]
    assert [float(r["PRODUCT_LENGTH"]) for r in rows] == [100.0, 200.0, 300.5]


def test_save_submission_mismatched_lengths_raise(tmp_path):
    with pytest.raises(ValueError):
        mc.save_submission(str(tmp_path / "s.csv"), [1, 2], [1.0])


def test_load_submission_wrong_header_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("PRODUCT_ID,SOMETHING_ELSE\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        mc.load_submission(str(p))


def test_load_submission_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mc.load_submission(str(tmp_path / "nope.csv"))


# ---------------------------------------------------------------------------
# Module contract
# ---------------------------------------------------------------------------

HEAVY = {"numpy", "pandas", "sklearn", "tensorflow"}


def test_no_heavy_imports_at_module_level():
    # The core module must stay importable without the ML stack, so heavy
    # imports may only appear inside function bodies.
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(mc))
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.append(node.module.split(".")[0])
    assert HEAVY.isdisjoint(top_level), f"heavy imports at module level: {HEAVY & set(top_level)}"
