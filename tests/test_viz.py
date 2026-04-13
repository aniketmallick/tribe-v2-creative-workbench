from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

import viz


def _fake_surface_mesh() -> dict[str, np.ndarray]:
    coords = np.zeros((viz.EXPECTED_VERTICES, 3), dtype=float)
    coords[:, 0] = np.linspace(-1.0, 1.0, viz.EXPECTED_VERTICES)
    faces = np.array([[0, 1, 2], [2, 3, 4], [10, 11, 12]], dtype=np.int64)
    return {
        "coords": coords,
        "faces": faces,
        "n_left_vertices": viz.EXPECTED_VERTICES // 2,
        "n_right_vertices": viz.EXPECTED_VERTICES // 2,
    }


def test_validate_predictions_accepts_valid_shape() -> None:
    predictions = np.zeros((2, viz.EXPECTED_VERTICES), dtype=float)
    viz.validate_predictions(predictions)


def test_validate_predictions_rejects_wrong_ndim() -> None:
    predictions = np.zeros((viz.EXPECTED_VERTICES,), dtype=float)
    with pytest.raises(ValueError, match="must be 2D"):
        viz.validate_predictions(predictions)


def test_validate_predictions_rejects_empty_timesteps() -> None:
    predictions = np.zeros((0, viz.EXPECTED_VERTICES), dtype=float)
    with pytest.raises(ValueError, match="empty"):
        viz.validate_predictions(predictions)


def test_validate_predictions_rejects_wrong_vertices() -> None:
    predictions = np.zeros((2, 100), dtype=float)
    with pytest.raises(ValueError, match=str(viz.EXPECTED_VERTICES)):
        viz.validate_predictions(predictions)


def test_validate_time_step_bounds() -> None:
    viz._validate_time_step(0, 2)
    with pytest.raises(ValueError, match="out of range"):
        viz._validate_time_step(2, 2)


def test_resolve_color_range_defaults_and_padding() -> None:
    lo, hi = viz._resolve_color_range(np.array([1.0, 1.0]), vmin=None, vmax=None)
    assert lo < 1.0 < hi

    lo2, hi2 = viz._resolve_color_range(np.array([1.0, 2.0]), vmin=None, vmax=None)
    assert lo2 == pytest.approx(1.0)
    assert hi2 == pytest.approx(2.0)


def test_resolve_color_range_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="vmin"):
        viz._resolve_color_range(np.array([0.0, 1.0]), vmin=2.0, vmax=1.0)


def test_load_prediction_file_validation(tmp_path) -> None:
    good = tmp_path / "pred.npy"
    bad_ext = tmp_path / "pred.txt"

    np.save(good, np.zeros((1, viz.EXPECTED_VERTICES), dtype=float))
    bad_ext.write_text("x", encoding="utf-8")

    loaded = viz.load_prediction_file(good)
    assert loaded.shape == (1, viz.EXPECTED_VERTICES)

    with pytest.raises(ValueError, match=".npy"):
        viz.load_prediction_file(bad_ext)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        viz.load_prediction_file(tmp_path / "missing.npy")


def test_load_fsaverage5_combines_hemispheres(monkeypatch: pytest.MonkeyPatch) -> None:
    left_coords = np.zeros((10_242, 3), dtype=float)
    right_coords = np.ones((10_242, 3), dtype=float)
    left_faces = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    right_faces = np.array([[0, 1, 2]], dtype=np.int64)

    class FakeDatasets:
        @staticmethod
        def fetch_surf_fsaverage(mesh: str) -> SimpleNamespace:
            assert mesh == "fsaverage5"
            return SimpleNamespace(pial_left="left", pial_right="right")

    def fake_load_mesh_arrays(path: str) -> tuple[np.ndarray, np.ndarray]:
        if path == "left":
            return left_coords, left_faces
        if path == "right":
            return right_coords, right_faces
        raise AssertionError(f"Unexpected mesh path: {path}")

    monkeypatch.setattr(viz, "_require_nilearn", lambda: (FakeDatasets(), object()))
    monkeypatch.setattr(viz, "_load_mesh_arrays", fake_load_mesh_arrays)
    viz.load_fsaverage5.cache_clear()

    combined = viz.load_fsaverage5()

    assert combined["coords"].shape == (viz.EXPECTED_VERTICES, 3)
    np.testing.assert_array_equal(combined["faces"][: len(left_faces)], left_faces)
    np.testing.assert_array_equal(
        combined["faces"][len(left_faces) :],
        right_faces + 10_242,
    )
    assert combined["n_left_vertices"] == 10_242
    assert combined["n_right_vertices"] == 10_242

    viz.load_fsaverage5.cache_clear()


def test_load_fsaverage5_rejects_unexpected_vertex_count(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDatasets:
        @staticmethod
        def fetch_surf_fsaverage(mesh: str) -> SimpleNamespace:
            return SimpleNamespace(pial_left="left", pial_right="right")

    def fake_load_mesh_arrays(path: str) -> tuple[np.ndarray, np.ndarray]:
        coords = np.zeros((3, 3), dtype=float)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        return coords, faces

    monkeypatch.setattr(viz, "_require_nilearn", lambda: (FakeDatasets(), object()))
    monkeypatch.setattr(viz, "_load_mesh_arrays", fake_load_mesh_arrays)
    viz.load_fsaverage5.cache_clear()

    with pytest.raises(ValueError, match="vertex count"):
        viz.load_fsaverage5()

    viz.load_fsaverage5.cache_clear()


@pytest.mark.skipif(viz.go is None, reason="plotly not installed")
def test_render_brain_returns_mesh_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(viz, "load_fsaverage5", lambda: _fake_surface_mesh())
    predictions = np.ones((1, viz.EXPECTED_VERTICES), dtype=float)

    figure = viz.render_brain(predictions, time_step=0, title="test")

    assert len(figure.data) == 1
    mesh = figure.data[0]
    assert mesh.type == "mesh3d"
    assert mesh.cmin < mesh.cmax


@pytest.mark.skipif(viz.go is None, reason="plotly not installed")
def test_render_brain_rejects_time_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(viz, "load_fsaverage5", lambda: _fake_surface_mesh())
    predictions = np.ones((1, viz.EXPECTED_VERTICES), dtype=float)
    with pytest.raises(ValueError, match="out of range"):
        viz.render_brain(predictions, time_step=1)


@pytest.mark.skipif(viz.go is None, reason="plotly not installed")
def test_render_comparison_uses_shared_raw_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(viz, "load_fsaverage5", lambda: _fake_surface_mesh())

    pred_a = np.full((1, viz.EXPECTED_VERTICES), 1.0, dtype=float)
    pred_b = np.full((1, viz.EXPECTED_VERTICES), 3.0, dtype=float)
    diff = np.abs(pred_a - pred_b)

    fig_a, fig_b, fig_diff = viz.render_comparison(pred_a, pred_b, diff, time_step=0)
    mesh_a = fig_a.data[0]
    mesh_b = fig_b.data[0]
    mesh_diff = fig_diff.data[0]

    assert mesh_a.cmin == pytest.approx(mesh_b.cmin)
    assert mesh_a.cmax == pytest.approx(mesh_b.cmax)
    assert mesh_diff.cmin >= 0.0
    assert mesh_diff.colorscale[0][1] == "#000004"


def test_render_comparison_rejects_shape_mismatch() -> None:
    pred_a = np.zeros((1, viz.EXPECTED_VERTICES), dtype=float)
    pred_b = np.zeros((2, viz.EXPECTED_VERTICES), dtype=float)
    diff = np.zeros((1, viz.EXPECTED_VERTICES), dtype=float)

    with pytest.raises(ValueError, match="Shape mismatch"):
        viz.render_comparison(pred_a, pred_b, diff, time_step=0)
