from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "gradio is required for app.py. Install it with: pip install gradio"
    ) from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "numpy is required for app.py. Install dependencies first."
    ) from exc

import compare
import viz

RunOutput = tuple[
    str,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any],
    dict[str, Any],
    Any,
    Any,
    Any,
    dict[str, Any],
]


def _disabled_time_slider() -> dict[str, Any]:
    return gr.update(minimum=0, maximum=0, value=0, step=1, interactive=False)


def _enabled_time_slider(timesteps: int) -> dict[str, Any]:
    if timesteps <= 0:
        return _disabled_time_slider()
    return gr.update(minimum=0, maximum=timesteps - 1, value=0, step=1, interactive=True)


def _default_summary(message: str = "Run a comparison to see summary stats.") -> dict[str, Any]:
    return {"message": message}


def _slider_config(timesteps: int) -> dict[str, Any]:
    if timesteps <= 0:
        return {"minimum": 0, "maximum": 0, "value": 0, "step": 1, "interactive": False}
    return {
        "minimum": 0,
        "maximum": timesteps - 1,
        "value": 0,
        "step": 1,
        "interactive": True,
    }


def _resolve_reference_video(path_like: Any) -> str | None:
    if path_like is None:
        return None

    if not isinstance(path_like, (str, Path)):
        raise ValueError("Demo video path must be a string or pathlib.Path.")

    resolved = Path(path_like).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"Demo video path does not exist: {resolved}")
    return str(resolved)


def _initialize_demo_data(demo_data: dict[str, Any]) -> dict[str, Any]:
    pred_a = _validate_prediction_array(demo_data.get("pred_a"), label="Demo Ad A")
    pred_b = _validate_prediction_array(demo_data.get("pred_b"), label="Demo Ad B")
    diff = _validate_prediction_array(demo_data.get("diff"), label="Demo Difference")

    if pred_a.shape != pred_b.shape:
        raise ValueError(
            "Invalid demo data: pred_a and pred_b must have identical shapes, "
            f"got {pred_a.shape} vs {pred_b.shape}."
        )
    if diff.shape != pred_a.shape:
        raise ValueError(
            "Invalid demo data: diff must match pred_a/pred_b shape, "
            f"got diff={diff.shape}, pred={pred_a.shape}."
        )

    timesteps = int(diff.shape[0])
    if timesteps <= 0:
        raise ValueError("Invalid demo data: arrays must contain at least one timestep.")

    fig_a, fig_b, fig_diff = viz.render_comparison(
        pred_a=pred_a,
        pred_b=pred_b,
        diff=diff,
        time_step=0,
    )

    metadata = dict(demo_data.get("metadata") or {})
    metadata.setdefault("timesteps", timesteps)
    metadata.setdefault("aligned_shape", list(diff.shape))
    summary = _compute_summary_stats(diff)
    if metadata:
        summary = {"demo_metadata": metadata, **summary}

    return {
        "status": "Demo mode: loaded cached sample predictions.",
        "pred_a": pred_a,
        "pred_b": pred_b,
        "diff": diff,
        "metadata": metadata,
        "slider": _slider_config(timesteps),
        "fig_a": fig_a,
        "fig_b": fig_b,
        "fig_diff": fig_diff,
        "summary": summary,
        "video_a": _resolve_reference_video(demo_data.get("video_a")),
        "video_b": _resolve_reference_video(demo_data.get("video_b")),
    }


def _coerce_upload_path(upload: Any) -> str | Path | None:
    if isinstance(upload, (str, Path)):
        return upload

    if isinstance(upload, dict):
        for key in ("path", "video", "name"):
            value = upload.get(key)
            if isinstance(value, (str, Path)):
                return value
            if isinstance(value, dict):
                nested_path = value.get("path")
                if isinstance(nested_path, (str, Path)):
                    return nested_path

    if isinstance(upload, (tuple, list)) and upload:
        first = upload[0]
        if isinstance(first, (str, Path)):
            return first
        if isinstance(first, dict):
            nested_path = first.get("path")
            if isinstance(nested_path, (str, Path)):
                return nested_path

    return None


def _validate_uploaded_video(upload_value: Any, label: str) -> Path:
    upload_path = _coerce_upload_path(upload_value)
    if upload_path is None:
        raise ValueError(f"Missing upload for {label}.")

    path = Path(upload_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise ValueError(f"Invalid upload for {label}: {path}")
    return path


def _validate_prediction_array(array: Any, label: str) -> np.ndarray:
    array_np = np.asarray(array)
    try:
        viz.validate_predictions(array_np)
    except Exception as exc:
        raise ValueError(f"Invalid inference output for {label}: {exc}") from exc
    return array_np


def _compute_summary_stats(diff: np.ndarray) -> dict[str, Any]:
    diff_np = np.asarray(diff)
    if diff_np.ndim != 2 or diff_np.shape[0] == 0:
        raise ValueError("diff must be a non-empty 2D array.")

    mean_by_time = np.mean(diff_np, axis=1)
    top_indices = np.argsort(mean_by_time)[::-1][:5]

    top_time_steps = [
        {
            "time_step": int(time_index),
            "mean_cortical_difference": float(mean_by_time[time_index]),
        }
        for time_index in top_indices
    ]

    return {
        "aligned_shape": list(diff_np.shape),
        "overall_mean_difference": float(np.mean(diff_np)),
        "overall_max_difference": float(np.max(diff_np)),
        "top_5_time_steps_by_mean_cortical_difference": top_time_steps,
    }


def _progress_yield(message: str) -> RunOutput:
    return (
        message,
        None,
        None,
        None,
        {},
        _disabled_time_slider(),
        None,
        None,
        None,
        _default_summary(),
    )


def _error_yield(exc: Exception) -> RunOutput:
    return (
        f"Error: {exc}",
        None,
        None,
        None,
        {},
        _disabled_time_slider(),
        None,
        None,
        None,
        {"error": str(exc)},
    )


def _success_yield(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    diff: np.ndarray,
    aligned_t: int,
    fig_a: Any,
    fig_b: Any,
    fig_diff: Any,
) -> RunOutput:
    metadata = {
        "timesteps": int(aligned_t),
        "aligned_shape": list(diff.shape),
    }

    return (
        "Done",
        pred_a,
        pred_b,
        diff,
        metadata,
        _enabled_time_slider(aligned_t),
        fig_a,
        fig_b,
        fig_diff,
        _compute_summary_stats(diff),
    )


def _comparison_workflow(video_a: Path, video_b: Path) -> Generator[str, None, RunOutput]:
    yield "Loading model..."
    model = compare.load_model()

    yield "Processing Ad A..."
    pred_a_raw, _ = compare.run_video_inference(model=model, video_path=video_a)
    pred_a = _validate_prediction_array(pred_a_raw, label="Ad A")

    yield "Processing Ad B..."
    pred_b_raw, _ = compare.run_video_inference(model=model, video_path=video_b)
    pred_b = _validate_prediction_array(pred_b_raw, label="Ad B")

    yield "Aligning predictions..."
    aligned_a, aligned_b, aligned_t = compare.align_predictions(pred_a, pred_b)

    yield "Computing difference..."
    diff_raw = compare.compute_difference(aligned_a, aligned_b)
    diff = _validate_prediction_array(diff_raw, label="Difference")

    if diff.shape[0] != aligned_t:
        raise ValueError("Invalid aligned output: aligned timestep count does not match diff shape.")

    fig_a, fig_b, fig_diff = viz.render_comparison(
        pred_a=aligned_a,
        pred_b=aligned_b,
        diff=diff,
        time_step=0,
    )

    return _success_yield(aligned_a, aligned_b, diff, aligned_t, fig_a, fig_b, fig_diff)


def run_comparison(ad_a_upload: Any, ad_b_upload: Any):
    try:
        video_a = _validate_uploaded_video(ad_a_upload, label="Ad A")
        video_b = _validate_uploaded_video(ad_b_upload, label="Ad B")
    except Exception as exc:
        yield _error_yield(exc)
        return

    workflow = _comparison_workflow(video_a, video_b)
    while True:
        try:
            yield _progress_yield(next(workflow))
        except StopIteration as done:
            yield done.value
            return
        except Exception as exc:
            yield _error_yield(exc)
            return


def update_time_step(
    time_step: int,
    pred_a: np.ndarray | None,
    pred_b: np.ndarray | None,
    diff: np.ndarray | None,
):
    if pred_a is None or pred_b is None or diff is None:
        raise gr.Error("Run Comparison first.")

    try:
        step = int(time_step)
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"Invalid time step: {time_step}") from exc

    try:
        pred_a_np = _validate_prediction_array(pred_a, label="Ad A")
        pred_b_np = _validate_prediction_array(pred_b, label="Ad B")
        diff_np = _validate_prediction_array(diff, label="Difference")
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    if not 0 <= step < diff_np.shape[0]:
        raise gr.Error(
            f"time_step out of range: got {step}, valid range is 0 to {diff_np.shape[0] - 1}."
        )

    try:
        return viz.render_comparison(pred_a_np, pred_b_np, diff_np, time_step=step)
    except Exception as exc:
        raise gr.Error(f"Failed to render comparison plots: {exc}") from exc


def build_app(demo_data: dict[str, Any] | None = None) -> gr.Blocks:
    demo_mode = demo_data is not None
    initial = _initialize_demo_data(demo_data) if demo_mode else None
    initial_slider = initial["slider"] if initial else _slider_config(0)

    with gr.Blocks(title="TRIBE v2 Comparison MVP") as demo:
        gr.Markdown("## TRIBE v2 Comparison MVP")
        if demo_mode:
            gr.Markdown(
                "Demo mode is active. Cached predictions are preloaded and upload/inference controls are disabled."
            )

        with gr.Row():
            ad_a_input = gr.Video(
                label="Ad A (reference only)" if demo_mode else "Ad A",
                sources=["upload"],
                value=initial["video_a"] if initial else None,
                interactive=not demo_mode,
            )
            ad_b_input = gr.Video(
                label="Ad B (reference only)" if demo_mode else "Ad B",
                sources=["upload"],
                value=initial["video_b"] if initial else None,
                interactive=not demo_mode,
            )

        run_button = gr.Button("Run Comparison", variant="primary", interactive=not demo_mode)
        status_box = gr.Textbox(
            label="Status / Progress",
            lines=2,
            value=initial["status"] if initial else "Upload two videos and click Run Comparison.",
            interactive=False,
        )

        time_slider = gr.Slider(
            label="time_step",
            minimum=initial_slider["minimum"],
            maximum=initial_slider["maximum"],
            value=initial_slider["value"],
            step=initial_slider["step"],
            interactive=initial_slider["interactive"],
        )

        with gr.Row():
            plot_a = gr.Plot(label="Ad A", value=initial["fig_a"] if initial else None)
            plot_b = gr.Plot(label="Ad B", value=initial["fig_b"] if initial else None)
            plot_diff = gr.Plot(label="Difference", value=initial["fig_diff"] if initial else None)

        summary_stats = gr.JSON(
            label="Summary Stats",
            value=initial["summary"] if initial else _default_summary(),
        )

        pred_a_state = gr.State(value=initial["pred_a"] if initial else None)
        pred_b_state = gr.State(value=initial["pred_b"] if initial else None)
        diff_state = gr.State(value=initial["diff"] if initial else None)
        metadata_state = gr.State(value=initial["metadata"] if initial else {})

        if not demo_mode:
            run_button.click(
                fn=run_comparison,
                inputs=[ad_a_input, ad_b_input],
                outputs=[
                    status_box,
                    pred_a_state,
                    pred_b_state,
                    diff_state,
                    metadata_state,
                    time_slider,
                    plot_a,
                    plot_b,
                    plot_diff,
                    summary_stats,
                ],
            )

        time_slider.change(
            fn=update_time_step,
            inputs=[time_slider, pred_a_state, pred_b_state, diff_state],
            outputs=[plot_a, plot_b, plot_diff],
        )

    return demo.queue()


if __name__ == "__main__":
    build_app().launch()
