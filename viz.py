from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any, TYPE_CHECKING

try:
    import numpy as np
except ModuleNotFoundError as exc:
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    from nilearn import datasets, surface
except ModuleNotFoundError as exc:
    datasets = None  # type: ignore[assignment]
    surface = None  # type: ignore[assignment]
    _NILEARN_IMPORT_ERROR = exc
else:
    _NILEARN_IMPORT_ERROR = None

try:
    import plotly.graph_objects as go
except ModuleNotFoundError as exc:
    go = None  # type: ignore[assignment]
    _PLOTLY_IMPORT_ERROR = exc
else:
    _PLOTLY_IMPORT_ERROR = None

if TYPE_CHECKING:
    import numpy as np_typing
    import plotly.graph_objects as go_typing

EXPECTED_VERTICES = 20_484
DEFAULT_OUTPUT_DIR = Path("outputs")

__all__ = [
    "load_fsaverage5",
    "validate_predictions",
    "render_brain",
    "render_comparison",
    "load_prediction_file",
]


def _require_numpy() -> Any:
    if np is None:
        raise ModuleNotFoundError(
            "numpy is required for viz.py. Install dependencies first."
        ) from _NUMPY_IMPORT_ERROR
    return np


def _require_nilearn() -> tuple[Any, Any]:
    if datasets is None or surface is None:
        raise ModuleNotFoundError(
            "nilearn is required for surface loading. Install dependencies first "
            "(for example: pip install -r requirements.viz.txt)."
        ) from _NILEARN_IMPORT_ERROR
    return datasets, surface


def _require_plotly() -> Any:
    if go is None:
        raise ModuleNotFoundError(
            "plotly is required for rendering. Install it with: pip install plotly"
        ) from _PLOTLY_IMPORT_ERROR
    return go


def _validate_time_step(time_step: int, timesteps: int) -> None:
    if not 0 <= time_step < timesteps:
        raise ValueError(
            f"time_step out of range: got {time_step}, valid range is 0 to {timesteps - 1}."
        )


def _resolve_color_range(
    values: "np_typing.ndarray[Any, Any]",
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    np_mod = _require_numpy()
    values_np = np_mod.asarray(values, dtype=float)
    finite_values = values_np[np_mod.isfinite(values_np)]
    if finite_values.size == 0:
        raise ValueError("Cannot compute color range: all values are non-finite.")

    resolved_vmin = float(np_mod.min(finite_values)) if vmin is None else float(vmin)
    resolved_vmax = float(np_mod.max(finite_values)) if vmax is None else float(vmax)

    if resolved_vmax < resolved_vmin:
        raise ValueError(
            f"Invalid color range: vmin ({resolved_vmin}) must be <= vmax ({resolved_vmax})."
        )

    if resolved_vmax == resolved_vmin:
        padding = max(abs(resolved_vmax) * 0.05, 1e-6)
        resolved_vmin -= padding
        resolved_vmax += padding

    return resolved_vmin, resolved_vmax


def _load_mesh_arrays(mesh_path: str | Path) -> tuple["np_typing.ndarray[Any, Any]", "np_typing.ndarray[Any, Any]"]:
    np_mod = _require_numpy()
    _, surface_mod = _require_nilearn()
    coords, faces = surface_mod.load_surf_mesh(mesh_path)
    coords_np = np_mod.asarray(coords, dtype=float)
    faces_np = np_mod.asarray(faces, dtype=np_mod.int64)

    if coords_np.ndim != 2 or coords_np.shape[1] != 3:
        raise ValueError(f"Unexpected mesh coordinates shape: {coords_np.shape}")
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError(f"Unexpected mesh faces shape: {faces_np.shape}")
    return coords_np, faces_np


@lru_cache(maxsize=1)
def load_fsaverage5() -> dict[str, Any]:
    """Load and combine fsaverage5 left/right hemisphere geometry."""
    np_mod = _require_numpy()
    datasets_mod, _ = _require_nilearn()
    fsaverage = datasets_mod.fetch_surf_fsaverage(mesh="fsaverage5")

    left_coords, left_faces = _load_mesh_arrays(fsaverage.pial_left)
    right_coords, right_faces = _load_mesh_arrays(fsaverage.pial_right)

    left_vertices = int(left_coords.shape[0])
    right_vertices = int(right_coords.shape[0])
    total_vertices = left_vertices + right_vertices

    if total_vertices != EXPECTED_VERTICES:
        raise ValueError(
            "Combined fsaverage5 vertex count does not match TRIBE v2 output size: "
            f"expected {EXPECTED_VERTICES}, got {total_vertices} "
            f"({left_vertices} left + {right_vertices} right)."
        )

    # Assumption: TRIBE v2 cortical outputs are ordered [left hemisphere, right hemisphere].
    # Validate this ordering against upstream TRIBE references if rendering looks hemisphere-swapped.
    combined_coords = np_mod.vstack([left_coords, right_coords])
    combined_faces = np_mod.vstack([left_faces, right_faces + left_vertices])

    return {
        "coords": combined_coords,
        "faces": combined_faces,
        "n_left_vertices": left_vertices,
        "n_right_vertices": right_vertices,
    }


def validate_predictions(predictions: np.ndarray) -> None:
    """Validate TRIBE v2 cortical prediction array shape."""
    np_mod = _require_numpy()
    predictions_np = np_mod.asarray(predictions)

    if predictions_np.ndim != 2:
        raise ValueError(
            f"predictions must be 2D with shape (T, {EXPECTED_VERTICES}), got {predictions_np.shape}."
        )
    if predictions_np.shape[0] == 0:
        raise ValueError("predictions is empty: expected at least one time step.")
    if predictions_np.shape[1] != EXPECTED_VERTICES:
        raise ValueError(
            "predictions has incorrect vertex dimension: "
            f"expected {EXPECTED_VERTICES}, got {predictions_np.shape[1]}."
        )


def render_brain(
    predictions: np.ndarray,
    time_step: int,
    title: str = "",
    colorscale: str = "Viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> "go_typing.Figure":
    """Render a single time step of cortical predictions on fsaverage5."""
    np_mod = _require_numpy()
    go_mod = _require_plotly()

    predictions_np = np_mod.asarray(predictions)
    validate_predictions(predictions_np)
    _validate_time_step(time_step, predictions_np.shape[0])

    surface_data = load_fsaverage5()
    coords = surface_data["coords"]
    faces = surface_data["faces"]

    intensities = np_mod.asarray(predictions_np[time_step], dtype=float)
    if intensities.shape[0] != int(coords.shape[0]):
        raise ValueError(
            "Prediction vector size does not match mesh vertex count: "
            f"predictions={intensities.shape[0]}, mesh={coords.shape[0]}."
        )

    resolved_vmin, resolved_vmax = _resolve_color_range(intensities, vmin=vmin, vmax=vmax)
    plot_title = title or f"Brain prediction (t={time_step})"

    mesh = go_mod.Mesh3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=intensities,
        colorscale=colorscale,
        cmin=resolved_vmin,
        cmax=resolved_vmax,
        showscale=True,
        colorbar={
            "title": {"text": "Cortical Response (a.u.)", "font": {"color": "#F5F5F5"}},
            "len": 0.85,
            "thickness": 16,
            "tickfont": {"color": "#F5F5F5"},
        },
        lighting={
            "ambient": 0.5,
            "diffuse": 0.6,
            "specular": 0.1,
            "roughness": 0.85,
            "fresnel": 0.05,
        },
        lightposition={"x": 120, "y": 120, "z": 220},
        hovertemplate=(
            "x=%{x:.2f}<br>"
            "y=%{y:.2f}<br>"
            "z=%{z:.2f}<br>"
            "value=%{intensity:.5f}<extra></extra>"
        ),
    )

    figure = go_mod.Figure(data=[mesh])
    figure.update_layout(
        title={"text": plot_title, "x": 0.5},
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font={"color": "#F5F5F5"},
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
        scene={
            "bgcolor": "#111111",
            "aspectmode": "data",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "camera": {
                "eye": {"x": 1.85, "y": 0.0, "z": 0.4},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
            },
        },
    )
    return figure


def render_comparison(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    diff: np.ndarray,
    time_step: int,
    pred_vmin: float | None = None,
    pred_vmax: float | None = None,
    diff_vmin: float | None = None,
    diff_vmax: float | None = None,
) -> tuple["go_typing.Figure", "go_typing.Figure", "go_typing.Figure"]:
    """Render Ad A, Ad B, and Difference figures for the same time step.

    Pass ``pred_vmin``/``pred_vmax`` and ``diff_vmin``/``diff_vmax`` to lock the
    colour scale globally (e.g. computed across all timesteps) so frames are
    directly comparable.  When omitted, the range is derived from the slice at
    ``time_step``.
    """
    np_mod = _require_numpy()

    pred_a_np = np_mod.asarray(pred_a)
    pred_b_np = np_mod.asarray(pred_b)
    diff_np = np_mod.asarray(diff)

    validate_predictions(pred_a_np)
    validate_predictions(pred_b_np)
    validate_predictions(diff_np)

    if pred_a_np.shape != pred_b_np.shape:
        raise ValueError(f"Shape mismatch: pred_a {pred_a_np.shape} vs pred_b {pred_b_np.shape}.")
    if diff_np.shape != pred_a_np.shape:
        raise ValueError(
            f"Shape mismatch: diff {diff_np.shape} must match predictions {pred_a_np.shape}."
        )

    _validate_time_step(time_step, pred_a_np.shape[0])

    if pred_vmin is None or pred_vmax is None:
        raw_slice = np_mod.concatenate([pred_a_np[time_step], pred_b_np[time_step]])
        pred_vmin, pred_vmax = _resolve_color_range(raw_slice, vmin=pred_vmin, vmax=pred_vmax)

    if diff_vmin is None or diff_vmax is None:
        diff_slice = np_mod.asarray(diff_np[time_step], dtype=float)
        diff_vmin, diff_vmax = _resolve_color_range(diff_slice, vmin=diff_vmin, vmax=diff_vmax)

    diff_colorscale = "Inferno"

    fig_a = render_brain(
        pred_a_np,
        time_step=time_step,
        title=f"Ad A (t={time_step})",
        colorscale="Viridis",
        vmin=pred_vmin,
        vmax=pred_vmax,
    )
    fig_b = render_brain(
        pred_b_np,
        time_step=time_step,
        title=f"Ad B (t={time_step})",
        colorscale="Viridis",
        vmin=pred_vmin,
        vmax=pred_vmax,
    )
    fig_diff = render_brain(
        diff_np,
        time_step=time_step,
        title=f"Difference (t={time_step})",
        colorscale=diff_colorscale,
        vmin=diff_vmin,
        vmax=diff_vmax,
    )
    return fig_a, fig_b, fig_diff


def load_prediction_file(path: str | Path) -> np.ndarray:
    """Load and validate a prediction .npy file."""
    np_mod = _require_numpy()
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Prediction path is not a file: {resolved_path}")
    if resolved_path.suffix.lower() != ".npy":
        raise ValueError(f"Prediction file must be a .npy file, got: {resolved_path}")

    predictions = np_mod.load(resolved_path, allow_pickle=False)
    validate_predictions(predictions)
    return np_mod.asarray(predictions)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render TRIBE v2 cortical predictions on fsaverage5 with Plotly."
    )
    parser.add_argument("--pred-a", type=str, required=True, help="Path to pred_A.npy")
    parser.add_argument("--pred-b", type=str, default=None, help="Path to pred_B.npy")
    parser.add_argument("--diff", type=str, default=None, help="Path to diff.npy")
    parser.add_argument(
        "--time-step",
        type=int,
        default=0,
        help="Time step index to render (default: 0).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    pred_a = load_prediction_file(args.pred_a)

    output_dir = DEFAULT_OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_requested = args.pred_b is not None or args.diff is not None
    if not comparison_requested:
        figure = render_brain(pred_a, time_step=args.time_step, title=f"Ad A (t={args.time_step})")
        output_path = output_dir / "viz_single.html"
        figure.write_html(output_path, include_plotlyjs="cdn")
        print(f"Saved visualization: {output_path}")
        return

    if args.pred_b is None or args.diff is None:
        raise ValueError(
            "Invalid arguments: provide either only --pred-a, "
            "or all three --pred-a, --pred-b, and --diff."
        )

    pred_b = load_prediction_file(args.pred_b)
    diff = load_prediction_file(args.diff)
    fig_a, fig_b, fig_diff = render_comparison(
        pred_a=pred_a,
        pred_b=pred_b,
        diff=diff,
        time_step=args.time_step,
    )

    ad_a_path = output_dir / "viz_ad_a.html"
    ad_b_path = output_dir / "viz_ad_b.html"
    diff_path = output_dir / "viz_diff.html"

    fig_a.write_html(ad_a_path, include_plotlyjs="cdn")
    fig_b.write_html(ad_b_path, include_plotlyjs="cdn")
    fig_diff.write_html(diff_path, include_plotlyjs="cdn")

    print("Saved visualizations:")
    print(f"  - {ad_a_path}")
    print(f"  - {ad_b_path}")
    print(f"  - {diff_path}")


if __name__ == "__main__":
    main()
