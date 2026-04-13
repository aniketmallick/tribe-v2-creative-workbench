import pytest

np = pytest.importorskip("numpy")

import compare


def test_align_truncates_to_shorter() -> None:
    pred_a = np.zeros((5, compare.EXPECTED_VERTICES), dtype=float)
    pred_b = np.ones((3, compare.EXPECTED_VERTICES), dtype=float)

    aligned_a, aligned_b, aligned_t = compare.align_predictions(pred_a, pred_b)

    assert aligned_t == 3
    assert aligned_a.shape == (3, compare.EXPECTED_VERTICES)
    assert aligned_b.shape == (3, compare.EXPECTED_VERTICES)
    np.testing.assert_array_equal(aligned_a, pred_a[:3])
    np.testing.assert_array_equal(aligned_b, pred_b[:3])


def test_align_rejects_zero_timesteps() -> None:
    pred_a = np.zeros((0, compare.EXPECTED_VERTICES), dtype=float)
    pred_b = np.zeros((3, compare.EXPECTED_VERTICES), dtype=float)

    with pytest.raises(ValueError, match="zero timesteps"):
        compare.align_predictions(pred_a, pred_b)


def test_compute_difference_shape() -> None:
    pred_a = np.zeros((2, compare.EXPECTED_VERTICES), dtype=float)
    pred_b = np.full((2, compare.EXPECTED_VERTICES), 3.0, dtype=float)

    diff = compare.compute_difference(pred_a, pred_b)

    assert diff.shape == (2, compare.EXPECTED_VERTICES)
    assert float(diff.mean()) == pytest.approx(3.0)
    assert float(diff.max()) == pytest.approx(3.0)


def test_validate_predictions_wrong_ndim() -> None:
    pred = np.zeros((compare.EXPECTED_VERTICES,), dtype=float)

    with pytest.raises(ValueError, match="must be 2D"):
        compare._validate_predictions(pred, label="pred")


def test_validate_predictions_wrong_vertices() -> None:
    pred = np.zeros((2, 100), dtype=float)

    with pytest.raises(ValueError, match=str(compare.EXPECTED_VERTICES)):
        compare._validate_predictions(pred, label="pred")
