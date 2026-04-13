from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
gr = pytest.importorskip("gradio")

import app
import compare


def _predictions(timesteps: int, value: float) -> np.ndarray:
    return np.full((timesteps, compare.EXPECTED_VERTICES), value, dtype=float)


def test_app_object_and_builder_exist() -> None:
    assert isinstance(app.app, gr.Blocks)
    built = app.build_app()
    assert isinstance(built, gr.Blocks)


def test_validate_uploaded_video_accepts_path_and_video_dict(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")

    resolved = video.resolve()
    assert app._validate_uploaded_video(str(video), "Ad A") == resolved
    assert app._validate_uploaded_video({"path": str(video)}, "Ad A") == resolved
    assert app._validate_uploaded_video({"video": {"path": str(video)}}, "Ad A") == resolved


def test_validate_uploaded_video_rejects_missing_upload() -> None:
    with pytest.raises(ValueError, match="Missing upload"):
        app._validate_uploaded_video(None, "Ad A")


def test_validate_prediction_array_rejects_invalid_shape() -> None:
    bad = np.zeros((compare.EXPECTED_VERTICES,), dtype=float)

    with pytest.raises(ValueError, match="Invalid inference output"):
        app._validate_prediction_array(bad, "Ad A")


def test_compute_summary_stats_returns_expected_ordering() -> None:
    diff = np.zeros((4, compare.EXPECTED_VERTICES), dtype=float)
    diff[0, :] = 0.1
    diff[1, :] = 1.0
    diff[2, :] = 0.5
    diff[3, :] = 2.0

    summary = app._compute_summary_stats(diff)

    assert summary["aligned_shape"] == [4, compare.EXPECTED_VERTICES]
    assert summary["overall_mean_difference"] == pytest.approx(0.9)
    assert summary["overall_max_difference"] == pytest.approx(2.0)
    top_steps = [row["time_step"] for row in summary["top_5_time_steps_by_mean_cortical_difference"]]
    assert top_steps == [3, 1, 2, 0]


def test_update_time_step_requires_state() -> None:
    with pytest.raises(gr.Error, match="Run Comparison first"):
        app.update_time_step(0, None, None, None)


def test_update_time_step_rejects_out_of_range() -> None:
    pred_a = _predictions(2, 0.0)
    pred_b = _predictions(2, 1.0)
    diff = np.abs(pred_a - pred_b)

    with pytest.raises(gr.Error, match="out of range"):
        app.update_time_step(5, pred_a, pred_b, diff)


def test_update_time_step_calls_render_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    pred_a = _predictions(3, 0.0)
    pred_b = _predictions(3, 1.0)
    diff = np.abs(pred_a - pred_b)

    called = {"step": None}

    def fake_render_comparison(pred_a_arg, pred_b_arg, diff_arg, time_step):
        called["step"] = time_step
        assert pred_a_arg.shape == pred_a.shape
        assert pred_b_arg.shape == pred_b.shape
        assert diff_arg.shape == diff.shape
        return "fig_a", "fig_b", "fig_diff"

    monkeypatch.setattr(app.viz, "render_comparison", fake_render_comparison)

    result = app.update_time_step(2, pred_a, pred_b, diff)

    assert result == ("fig_a", "fig_b", "fig_diff")
    assert called["step"] == 2


def test_run_comparison_progress_and_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyModel:
        pass

    def fake_load_model():
        return DummyModel()

    def fake_run_video_inference(model, video_path):
        path = Path(video_path)
        if path.name.startswith("a"):
            return _predictions(4, 0.0), None
        return _predictions(3, 2.0), None

    def fake_render_comparison(*, pred_a, pred_b, diff, time_step):
        return (
            {"plot": "a", "time_step": int(time_step), "shape": list(pred_a.shape)},
            {"plot": "b", "time_step": int(time_step), "shape": list(pred_b.shape)},
            {"plot": "diff", "time_step": int(time_step), "shape": list(diff.shape)},
        )

    monkeypatch.setattr(app.compare, "load_model", fake_load_model)
    monkeypatch.setattr(app.compare, "run_video_inference", fake_run_video_inference)
    monkeypatch.setattr(app.viz, "render_comparison", fake_render_comparison)

    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"b")

    updates = list(app.run_comparison(str(video_a), str(video_b)))
    statuses = [update[0] for update in updates]

    assert statuses == [
        "Loading model...",
        "Processing Ad A...",
        "Processing Ad B...",
        "Aligning predictions...",
        "Computing difference...",
        "Done",
    ]

    final = updates[-1]
    pred_a, pred_b, diff = final[1], final[2], final[3]

    assert pred_a.shape == (3, compare.EXPECTED_VERTICES)
    assert pred_b.shape == (3, compare.EXPECTED_VERTICES)
    assert diff.shape == (3, compare.EXPECTED_VERTICES)

    slider = final[5]
    assert slider["minimum"] == 0
    assert slider["maximum"] == 2
    assert slider["interactive"] is True

    summary = final[9]
    assert summary["aligned_shape"] == [3, compare.EXPECTED_VERTICES]
    assert "overall_mean_difference" in summary
    assert "overall_max_difference" in summary


def test_run_comparison_missing_upload_returns_error() -> None:
    updates = list(app.run_comparison(None, None))

    assert len(updates) == 1
    assert updates[0][0].startswith("Error:")
    assert "Missing upload for Ad A" in updates[0][9]["error"]
