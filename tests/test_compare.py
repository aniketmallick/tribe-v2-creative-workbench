import json

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


def test_validate_cortical_array_wrong_ndim() -> None:
    pred = np.zeros((compare.EXPECTED_VERTICES,), dtype=float)

    with pytest.raises(ValueError, match="must be 2D"):
        compare._validate_cortical_array(pred, label="pred")


def test_validate_cortical_array_wrong_vertices() -> None:
    pred = np.zeros((2, 100), dtype=float)

    with pytest.raises(ValueError, match=str(compare.EXPECTED_VERTICES)):
        compare._validate_cortical_array(pred, label="pred")


def test_iter_segment_items_handles_to_dict_records() -> None:
    class SegmentFrame:
        def to_dict(self, orient: str = "records") -> list[dict[str, float]]:
            assert orient == "records"
            return [{"start": 0.0, "duration": 1.0}]

    items = compare._iter_segment_items(SegmentFrame())

    assert items == [{"start": 0.0, "duration": 1.0}]


def test_iter_segment_items_handles_mapping_string_iterable_and_fallback() -> None:
    mapping_items = compare._iter_segment_items({"start": 0.0, "duration": 1.0})
    string_items = compare._iter_segment_items("not-segments")
    iterable_items = compare._iter_segment_items((1, 2, 3))

    class SegmentObject:
        pass

    obj = SegmentObject()
    fallback_items = compare._iter_segment_items(obj)

    assert mapping_items == [{"start": 0.0, "duration": 1.0}]
    assert string_items == []
    assert iterable_items == [1, 2, 3]
    assert fallback_items == [obj]


def test_segment_timing_rows_limit_and_coercion() -> None:
    class SegmentObject:
        def __init__(self, start: object, duration: object) -> None:
            self.start = start
            self.duration = duration

    segments = [
        {"start": "0.25", "duration": 1},
        {"start": "bad", "duration": None},
        SegmentObject(2.5, "3.5"),
    ]

    rows = compare._segment_timing_rows(segments, limit=2)

    assert rows == [
        {"index": 0, "start": 0.25, "duration": 1.0},
        {"index": 1, "start": None, "duration": None},
    ]


def test_save_outputs_writes_arrays_metadata_and_segments(tmp_path) -> None:
    class SegmentObject:
        def __init__(self, start: object, duration: object) -> None:
            self.start = start
            self.duration = duration

    pred_a = np.zeros((2, compare.EXPECTED_VERTICES), dtype=float)
    pred_b = np.ones((2, compare.EXPECTED_VERTICES), dtype=float)
    diff = compare.compute_difference(pred_a, pred_b)
    metadata = {"diff_mean": 123.0, "diff_max": 456.0, "note": "keep-user-stats"}

    compare.save_outputs(
        tmp_path,
        pred_a,
        pred_b,
        diff,
        metadata,
        segments_a=[{"start": 0.0, "duration": 1.0}, {"start": 2.0, "duration": 1.0}],
        segments_b=[SegmentObject("0.1", "0.5"), SegmentObject("bad", None)],
        aligned_t=1,
    )

    pred_a_file = tmp_path / "pred_A.npy"
    pred_b_file = tmp_path / "pred_B.npy"
    diff_file = tmp_path / "diff.npy"
    metadata_file = tmp_path / "metadata.json"
    segments_a_file = tmp_path / "segments_A.json"
    segments_b_file = tmp_path / "segments_B.json"

    assert pred_a_file.exists()
    assert pred_b_file.exists()
    assert diff_file.exists()
    assert metadata_file.exists()
    assert segments_a_file.exists()
    assert segments_b_file.exists()

    np.testing.assert_array_equal(np.load(pred_a_file), pred_a)
    np.testing.assert_array_equal(np.load(pred_b_file), pred_b)
    np.testing.assert_array_equal(np.load(diff_file), diff)

    metadata_payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert metadata_payload["diff_mean"] == pytest.approx(123.0)
    assert metadata_payload["diff_max"] == pytest.approx(456.0)
    assert metadata_payload["segments_a_count"] == 1
    assert metadata_payload["segments_b_count"] == 1
    assert metadata_payload["saved_files"]["segments_A"].endswith("segments_A.json")
    assert metadata_payload["saved_files"]["segments_B"].endswith("segments_B.json")

    segments_a_payload = json.loads(segments_a_file.read_text(encoding="utf-8"))
    segments_b_payload = json.loads(segments_b_file.read_text(encoding="utf-8"))
    assert segments_a_payload["segment_count"] == 1
    assert segments_b_payload["segment_count"] == 1
    assert segments_a_payload["segments"][0] == {"index": 0, "start": 0.0, "duration": 1.0}
    assert segments_b_payload["segments"][0] == {"index": 0, "start": 0.1, "duration": 0.5}


def test_save_outputs_rejects_diff_shape_mismatch(tmp_path) -> None:
    pred_a = np.zeros((2, compare.EXPECTED_VERTICES), dtype=float)
    pred_b = np.zeros((2, compare.EXPECTED_VERTICES), dtype=float)
    diff = np.zeros((1, compare.EXPECTED_VERTICES), dtype=float)

    with pytest.raises(ValueError, match="diff shape must match"):
        compare.save_outputs(tmp_path, pred_a, pred_b, diff, {})
