from __future__ import annotations

import http.server
import json
import struct
import threading
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

    timing_a: list[float] = demo_data.get("timing_a") or [float(i) for i in range(timesteps)]
    timing_b: list[float] = demo_data.get("timing_b") or [float(i) for i in range(timesteps)]

    video_a = _resolve_reference_video(demo_data.get("video_a"))
    video_b = _resolve_reference_video(demo_data.get("video_b"))

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
        "video_a": video_a,
        "video_b": video_b,
        "timing_a": timing_a,
        "timing_b": timing_b,
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


def _write_intensities_binary(path: Path, pred_a: np.ndarray, pred_b: np.ndarray, diff: np.ndarray) -> None:
    """Write intensities + per-frame color scales as flat float32 binary.

    Layout:
      [T: int32, N: int32]                          — 8 bytes header
      [pred_a_flat: float32 × T×N]                  — intensity data
      [pred_b_flat: float32 × T×N]
      [diff_flat:   float32 × T×N]
      [pred_vmin: float32 × T]                      — per-frame color scales
      [pred_vmax: float32 × T]
      [diff_vmin: float32 × T]
      [diff_vmax: float32 × T]
    """
    T, N = int(pred_a.shape[0]), int(pred_a.shape[1])

    # Per-frame color ranges matching viz.render_comparison's per-frame logic.
    pred_vmin = np.minimum(pred_a.min(axis=1), pred_b.min(axis=1)).astype(np.float32)
    pred_vmax = np.maximum(pred_a.max(axis=1), pred_b.max(axis=1)).astype(np.float32)
    diff_vmin = diff.min(axis=1).astype(np.float32)
    diff_vmax = diff.max(axis=1).astype(np.float32)

    with path.open("wb") as f:
        f.write(struct.pack("<2i", T, N))
        f.write(pred_a.astype(np.float32).tobytes())
        f.write(pred_b.astype(np.float32).tobytes())
        f.write(diff.astype(np.float32).tobytes())
        f.write(pred_vmin.tobytes())
        f.write(pred_vmax.tobytes())
        f.write(diff_vmin.tobytes())
        f.write(diff_vmax.tobytes())


def update_timestamp_only(
    time_step: int,
    timing_a: list[float] | None,
    timing_b: list[float] | None,
) -> str:
    return _timestamp_label(int(time_step), timing_a, timing_b)


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


def _timestamp_label(step: int, timing_a: list[float] | None, timing_b: list[float] | None) -> str:
    t_a = timing_a[step] if timing_a and step < len(timing_a) else float(step)
    t_b = timing_b[step] if timing_b and step < len(timing_b) else float(step)
    if abs(t_a - t_b) < 0.1:
        return f"**t = {step}** &nbsp;·&nbsp; ≈ **{t_a:.1f}s** into both videos"
    return f"**t = {step}** &nbsp;·&nbsp; Ad A ≈ **{t_a:.1f}s** &nbsp;|&nbsp; Ad B ≈ **{t_b:.1f}s**"


def update_time_step(
    time_step: int,
    pred_a: np.ndarray | None,
    pred_b: np.ndarray | None,
    diff: np.ndarray | None,
    timing_a: list[float] | None = None,
    timing_b: list[float] | None = None,
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
        figs = viz.render_comparison(pred_a_np, pred_b_np, diff_np, time_step=step)
    except Exception as exc:
        raise gr.Error(f"Failed to render comparison plots: {exc}") from exc

    return (*figs, _timestamp_label(step, timing_a, timing_b))


_file_servers: dict[str, int] = {}  # directory → port


def _find_plotly_js() -> Path | None:
    """Locate the bundled plotly.min.js shipped with the plotly Python package."""
    try:
        import plotly as _plotly_pkg  # noqa: PLC0415
        candidate = Path(_plotly_pkg.__file__).parent / "package_data" / "plotly.min.js"
        return candidate if candidate.exists() else None
    except Exception:
        return None


def _start_video_file_server(directory: str, port: int = 7862) -> int:
    """Start a per-directory background HTTP server. Returns the port used.

    Each unique directory gets its own server so multiple demo sessions in the
    same process don't cross-serve stale assets from the wrong folder.
    """
    canonical = str(Path(directory).expanduser().resolve())
    if canonical in _file_servers:
        return _file_servers[canonical]

    plotly_js_path = _find_plotly_js()

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=directory, **kwargs)

        def end_headers(self) -> None:  # type: ignore[override]
            # CORS headers on every response so fetch() from Gradio's port works.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            super().end_headers()

        def do_OPTIONS(self) -> None:  # type: ignore[override]
            self.send_response(200)
            self.end_headers()

        def do_GET(self) -> None:  # type: ignore[override]
            # Serve plotly.min.js from the venv so the browser can set window.Plotly.
            if self.path == "/plotly.min.js" and plotly_js_path:
                content = plotly_js_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            super().do_GET()

        def log_message(self, *args: Any) -> None:
            pass  # silence request logs

    for candidate in range(port, port + 20):
        try:
            server = http.server.HTTPServer(("127.0.0.1", candidate), _Handler)
        except OSError:
            continue
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _file_servers[canonical] = candidate
        return candidate

    raise RuntimeError(
        f"Could not bind file server to any port in range {port}–{port + 19}. "
        "Free one of those ports and retry."
    )


def _make_show_step_js() -> str:
    """Standalone JS that defines window._tribeShowStep. No braces need escaping."""
    return (
        "  window._tribeShowStep = function(step) {\n"
        "    const f = window._tribeFrm, P = window.Plotly;\n"
        "    if (!f || !P) return;\n"
        "    const { T, N, data, scales } = f;\n"
        "    if (step < 0 || step >= T) return;\n"
        # scales layout: [pred_vmin×T, pred_vmax×T, diff_vmin×T, diff_vmax×T]
        "    const pVmin = scales[step],       pVmax = scales[T + step];\n"
        "    const dVmin = scales[2*T + step], dVmax = scales[3*T + step];\n"
        "    ['tribe-plot-a', 'tribe-plot-b', 'tribe-plot-diff'].forEach((id, p) => {\n"
        "      const div = document.querySelector('#' + id + ' .js-plotly-plot');\n"
        "      if (!div) return;\n"
        "      const off = p * T * N + step * N;\n"
        "      const vmin = p < 2 ? pVmin : dVmin;\n"
        "      const vmax = p < 2 ? pVmax : dVmax;\n"
        "      P.update(div,\n"
        "        { intensity: [data.subarray(off, off + N)], cmin: vmin, cmax: vmax },\n"
        "        { 'title.text': ['Ad A','Ad B','Difference'][p] + ' (t=' + step + ')' },\n"
        "        [0]);\n"
        "    });\n"
        "    const t = window._tribeTiming;\n"
        "    const ta = t ? (t.a[step] ?? step) : step;\n"
        "    const c = document.getElementById('tribe-clock');\n"
        "    if (c) c.textContent = 't=' + step + '  \u00b7  ' + ta.toFixed(1) + 's';\n"
        "  };\n"
    )


def _init_js(port: int, timing_a: list[float], timing_b: list[float]) -> str:
    """gr.Blocks(js=...) init: fetch assets + define window._tribeShowStep at
    page load so the slider works before the user ever clicks Play."""
    tp = json.dumps({"a": timing_a, "b": timing_b})
    has_plotly = _find_plotly_js() is not None
    psrc = f"http://127.0.0.1:{port}/plotly.min.js"
    load_plotly = (
        "  const _ps = document.createElement('script');\n"
        f"  _ps.src = '{psrc}'; _ps.onload = _def; document.head.appendChild(_ps);\n"
        if has_plotly
        else "  _def();\n"
    )
    return (
        "() => {\n"
        f"  window._tribeTiming = {tp};\n"
        f"  fetch('http://127.0.0.1:{port}/intensities.bin')\n"
        "    .then(r => r.arrayBuffer())\n"
        "    .then(buf => {\n"
        "      const h = new Int32Array(buf, 0, 2), T = h[0], N = h[1];\n"
        "      const dataLen = 3 * T * N;\n"
        "      window._tribeFrm = {\n"
        "        T, N,\n"
        "        data:   new Float32Array(buf, 8, dataLen),\n"
        "        scales: new Float32Array(buf, 8 + dataLen * 4, 4 * T),\n"
        "      };\n"
        "    }).catch(e => console.warn('[tribe] init fetch:', e));\n"
        "  function _def() {\n"
        + _make_show_step_js()
        + "  }\n"
        "  if (window.Plotly) { _def(); } else {\n"
        + load_plotly
        + "  }\n"
        "}"
    )


def _play_js(port: int, timing_a: list[float], timing_b: list[float]) -> str:
    """Play-button handler: loads assets lazily, (re-)defines _tribeShowStep,
    attaches timeupdate sync, plays both videos."""
    tp = json.dumps({"a": timing_a, "b": timing_b})
    has_plotly = _find_plotly_js() is not None
    psrc = f"http://127.0.0.1:{port}/plotly.min.js"
    load_plotly_js = (
        "    const _ps = document.createElement('script');\n"
        f"    _ps.src = '{psrc}';\n"
        "    _ps.onload = () => { (window._tribePending||[]).forEach(f=>f()); };\n"
        "    _ps.onerror = rej; document.head.appendChild(_ps);\n"
        if has_plotly
        else "    res();\n"
    )
    return (
        "() => {\n"
        "  const va = document.getElementById('tribe-vid-a') || document.querySelectorAll('video')[0];\n"
        "  const vb = document.getElementById('tribe-vid-b') || document.querySelectorAll('video')[1];\n"
        "  if (!va || !vb) return;\n"
        "\n"
        "  const loadPlotly = window.Plotly ? Promise.resolve()\n"
        "    : new Promise((res, rej) => {\n"
        "        if (window._tribePending) { window._tribePending.push(res); return; }\n"
        "        window._tribePending = [res];\n"
        + load_plotly_js
        + "      });\n"
        "\n"
        f"  const loadData = window._tribeFrm ? Promise.resolve()\n"
        f"    : fetch('http://127.0.0.1:{port}/intensities.bin')\n"
        "        .then(r => r.arrayBuffer())\n"
        "        .then(buf => {\n"
        "          const h = new Int32Array(buf, 0, 2), T = h[0], N = h[1];\n"
        "          const dataLen = 3 * T * N;\n"
        "          window._tribeFrm = {\n"
        "            T, N,\n"
        "            data:   new Float32Array(buf, 8, dataLen),\n"
        "            scales: new Float32Array(buf, 8 + dataLen * 4, 4 * T),\n"
        "          };\n"
        f"          window._tribeTiming = {tp};\n"
        "        });\n"
        "\n"
        "  Promise.all([loadPlotly, loadData])\n"
        "    .then(() => {\n"
        + _make_show_step_js().replace("\n", "\n  ")
        + "    })\n"
        "    .catch(e => console.warn('[tribe] asset load failed:', e));\n"
        "\n"
        "  if (!va._tribeSync) {\n"
        "    va._tribeSync = () => {\n"
        "      const c = document.getElementById('tribe-clock');\n"
        "      if (c) c.textContent = va.currentTime.toFixed(1) + 's';\n"
        "      if (Math.abs(va.currentTime - vb.currentTime) > 0.3) vb.currentTime = va.currentTime;\n"
        "      const step = Math.floor(va.currentTime);\n"
        "      if (step !== va._lastStep) {\n"
        "        va._lastStep = step;\n"
        "        if (window._tribeShowStep) window._tribeShowStep(step);\n"
        "        const wrap = document.getElementById('tribe-time-slider');\n"
        "        const slider = wrap ? wrap.querySelector('input[type=\"range\"]') : document.querySelector('input[type=\"range\"]');\n"
        "        if (slider && step <= parseInt(slider.max)) slider.value = step;\n"
        "      }\n"
        "    };\n"
        "    va.addEventListener('timeupdate', va._tribeSync);\n"
        "    va.addEventListener('ended', () => vb.pause());\n"
        "  }\n"
        "  Promise.all([va.play(), vb.play()]).catch(() => {});\n"
        "}\n"
    )


# ── Shared JS snippets ─────────────────────────────────────────────────────────

# Fallback _PLAY_JS used in full (non-demo) mode where no video server runs.
_PLAY_JS = """
() => {
  const va = document.getElementById('tribe-vid-a') || document.querySelectorAll('video')[0];
  const vb = document.getElementById('tribe-vid-b') || document.querySelectorAll('video')[1];
  if (!va || !vb) return;
  if (!va._tribeSync) {
    va._tribeSync = () => {
      if (Math.abs(va.currentTime - vb.currentTime) > 0.3) vb.currentTime = va.currentTime;
    };
    va.addEventListener('timeupdate', va._tribeSync);
    va.addEventListener('ended', () => vb.pause());
  }
  Promise.all([va.play(), vb.play()]).catch(() => {});
}
"""

_PAUSE_JS = """
() => {
  const va = document.getElementById('tribe-vid-a') || document.querySelectorAll('video')[0];
  const vb = document.getElementById('tribe-vid-b') || document.querySelectorAll('video')[1];
  if (va) va.pause();
  if (vb) vb.pause();
}
"""

# For manual slider drag: update plots client-side if intensities are loaded,
# seek videos, then pass inputs through to Python (for timestamp update).
_SEEK_JS = """
(step, ...rest) => {
  if (window._tribeShowStep) window._tribeShowStep(parseInt(step));
  const va = document.getElementById('tribe-vid-a') || document.querySelectorAll('video')[0];
  const vb = document.getElementById('tribe-vid-b') || document.querySelectorAll('video')[1];
  if (va && va.paused) va.currentTime = step;
  if (vb && vb.paused) vb.currentTime = step;
  const c = document.getElementById('tribe-clock');
  if (c && !window._tribeShowStep) c.textContent = parseFloat(step).toFixed(1) + 's';
  return [step, ...rest];
}
"""


def _video_elements_html(video_a: str, video_b: str) -> str:
    """Pure HTML — no script tags. JS is injected separately via gr.Blocks(js=...)."""
    directory = str(Path(video_a).parent)
    port = _start_video_file_server(directory)
    url_a = f"http://127.0.0.1:{port}/{Path(video_a).name}"
    url_b = f"http://127.0.0.1:{port}/{Path(video_b).name}"
    return f"""
<div style="display:flex;gap:12px;padding:12px;background:#1a1a1a;border-radius:10px">
  <div style="flex:1;min-width:0">
    <p style="color:#ccc;text-align:center;margin:0 0 6px;font-weight:600;font-size:13px">Ad A</p>
    <video id="tribe-vid-a" src="{url_a}"
      style="width:100%;border-radius:6px;display:block" preload="auto"></video>
  </div>
  <div style="flex:1;min-width:0">
    <p style="color:#ccc;text-align:center;margin:0 0 6px;font-weight:600;font-size:13px">Ad B</p>
    <video id="tribe-vid-b" src="{url_b}"
      style="width:100%;border-radius:6px;display:block" preload="auto"></video>
  </div>
</div>
<div style="text-align:center;padding:8px 0">
  <span id="tribe-clock"
    style="color:#888;font-family:monospace;font-size:13px">0.0s</span>
</div>
"""


def build_app(demo_data: dict[str, Any] | None = None) -> gr.Blocks:
    demo_mode = demo_data is not None
    initial = _initialize_demo_data(demo_data) if demo_mode else None
    initial_slider = initial["slider"] if initial else _slider_config(0)

    # Prepare client-side acceleration assets when videos are available.
    play_js_str: str | None = None
    if demo_mode and initial and initial.get("video_a") and initial.get("video_b"):
        directory = str(Path(initial["video_a"]).parent)
        port = _start_video_file_server(directory)
        _write_intensities_binary(
            Path(directory) / "intensities.bin",
            initial["pred_a"],
            initial["pred_b"],
            initial["diff"],
        )
        play_js_str = _play_js(port, initial["timing_a"], initial["timing_b"])

    _launch_js = _init_js(port, initial["timing_a"], initial["timing_b"]) if play_js_str else None

    with gr.Blocks(title="TRIBE v2 Comparison Workbench") as demo:
        gr.Markdown("# TRIBE v2 Comparison Workbench")

        if demo_mode:
            # ── Synced video players ───────────────────────────────────────────
            video_a = initial.get("video_a")
            video_b = initial.get("video_b")
            if video_a and video_b:
                gr.HTML(_video_elements_html(video_a, video_b))
                with gr.Row():
                    play_btn = gr.Button("▶ Play both", variant="primary", size="sm", visible=True)
                    pause_btn = gr.Button("⏸ Pause both", variant="secondary", size="sm", visible=False)

                play_btn.click(
                    fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
                    outputs=[play_btn, pause_btn],
                    js=play_js_str or _PLAY_JS,
                )
                pause_btn.click(
                    fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                    outputs=[play_btn, pause_btn],
                    js=_PAUSE_JS,
                )
        else:
            # ── Upload + inference controls (full mode only) ───────────────────
            gr.Markdown("### Upload Videos")
            with gr.Row():
                ad_a_input = gr.Video(label="Ad A", sources=["upload"])
                ad_b_input = gr.Video(label="Ad B", sources=["upload"])
            run_button = gr.Button("Run Comparison", variant="primary")
            status_box = gr.Textbox(
                label="Status",
                lines=2,
                value="Upload two videos and click Run Comparison.",
                interactive=False,
            )

        # ── Timestamp display + slider ─────────────────────────────────────────
        gr.Markdown("### Brain Predictions")
        timestamp_display = gr.Markdown(
            _timestamp_label(0, initial.get("timing_a") if initial else None,
                             initial.get("timing_b") if initial else None)
        )
        time_slider = gr.Slider(
            label="Time step",
            minimum=initial_slider["minimum"],
            maximum=initial_slider["maximum"],
            value=initial_slider["value"],
            step=initial_slider["step"],
            interactive=initial_slider["interactive"],
            elem_id="tribe-time-slider",
        )

        # ── Brain plots ────────────────────────────────────────────────────────
        with gr.Row():
            plot_a = gr.Plot(label="Ad A", value=initial["fig_a"] if initial else None, elem_id="tribe-plot-a")
            plot_b = gr.Plot(label="Ad B", value=initial["fig_b"] if initial else None, elem_id="tribe-plot-b")
            plot_diff = gr.Plot(label="Difference", value=initial["fig_diff"] if initial else None, elem_id="tribe-plot-diff")

        # ── Summary stats ──────────────────────────────────────────────────────
        summary_stats = gr.JSON(
            label="Summary Stats",
            value=initial["summary"] if initial else _default_summary(),
        )

        # ── State ──────────────────────────────────────────────────────────────
        pred_a_state = gr.State(value=initial["pred_a"] if initial else None)
        pred_b_state = gr.State(value=initial["pred_b"] if initial else None)
        diff_state = gr.State(value=initial["diff"] if initial else None)
        metadata_state = gr.State(value=initial["metadata"] if initial else {})
        timing_a_state = gr.State(value=initial["timing_a"] if initial else None)
        timing_b_state = gr.State(value=initial["timing_b"] if initial else None)

        # ── Event handlers ─────────────────────────────────────────────────────
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

        if demo_mode and play_js_str:
            # Plots update client-side via Plotly.restyle (zero round-trips).
            # Python only updates the timestamp label.
            time_slider.change(
                fn=update_timestamp_only,
                inputs=[time_slider, timing_a_state, timing_b_state],
                outputs=[timestamp_display],
                js=_SEEK_JS,
            )
        else:
            time_slider.change(
                fn=update_time_step,
                inputs=[time_slider, pred_a_state, pred_b_state, diff_state,
                        timing_a_state, timing_b_state],
                outputs=[plot_a, plot_b, plot_diff, timestamp_display],
                js=_SEEK_JS,
            )

    app = demo.queue()
    app._tribe_launch_js = _launch_js  # passed to launch() by the caller
    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(js=app._tribe_launch_js)
