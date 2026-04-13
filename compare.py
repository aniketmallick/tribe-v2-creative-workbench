from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import numpy as np
except ModuleNotFoundError as exc:
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

if TYPE_CHECKING:
    from tribev2.demo_utils import TribeModel


EXPECTED_VERTICES = 20_484
MODEL_ID = "facebook/tribev2"


def _require_numpy() -> Any:
    if np is None:
        raise ModuleNotFoundError(
            "numpy is required for compare.py. Install dependencies first "
            "(for example: pip install -r requirements.inference.txt)."
        ) from _NUMPY_IMPORT_ERROR
    return np


def _validate_video_path(video_path: str | Path) -> Path:
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video path does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Video path is not a file: {path}")
    return path


def _validate_predictions(predictions: np.ndarray, label: str) -> None:
    if predictions.ndim != 2:
        raise ValueError(
            f"{label} must be 2D with shape (T, {EXPECTED_VERTICES}), got {predictions.shape}."
        )
    if predictions.shape[1] != EXPECTED_VERTICES:
        raise ValueError(
            f"{label} must have {EXPECTED_VERTICES} vertices in axis 1, got {predictions.shape}."
        )


@lru_cache(maxsize=None)
def _load_model_cached(cache_dir: str) -> "TribeModel":
    from tribev2.demo_utils import TribeModel

    return TribeModel.from_pretrained(MODEL_ID, cache_folder=Path(cache_dir))


def load_model(cache_dir: str | Path = "./cache") -> "TribeModel":
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    return _load_model_cached(str(cache_path))


def run_video_inference(model: Any, video_path: str | Path) -> tuple[np.ndarray, object]:
    np_mod = _require_numpy()
    resolved_video_path = _validate_video_path(video_path)
    events = model.get_events_dataframe(video_path=resolved_video_path)
    predictions, segments = model.predict(events=events)
    predictions_np = np_mod.asarray(predictions)
    _validate_predictions(predictions_np, label=f"predictions for {resolved_video_path}")
    return predictions_np, segments


def align_predictions(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    mode: str = "truncate",
) -> tuple[np.ndarray, np.ndarray, int]:
    np_mod = _require_numpy()
    if mode != "truncate":
        raise ValueError("Only mode='truncate' is currently supported.")

    pred_a_np = np_mod.asarray(pred_a)
    pred_b_np = np_mod.asarray(pred_b)
    _validate_predictions(pred_a_np, label="pred_a")
    _validate_predictions(pred_b_np, label="pred_b")

    aligned_t = min(pred_a_np.shape[0], pred_b_np.shape[0])
    if aligned_t == 0:
        raise ValueError("Cannot align predictions with zero timesteps.")

    return pred_a_np[:aligned_t], pred_b_np[:aligned_t], aligned_t


def compute_difference(pred_a: np.ndarray, pred_b: np.ndarray) -> np.ndarray:
    np_mod = _require_numpy()
    pred_a_np = np_mod.asarray(pred_a)
    pred_b_np = np_mod.asarray(pred_b)
    _validate_predictions(pred_a_np, label="pred_a")
    _validate_predictions(pred_b_np, label="pred_b")
    if pred_a_np.shape != pred_b_np.shape:
        raise ValueError(f"Shape mismatch: pred_a {pred_a_np.shape} vs pred_b {pred_b_np.shape}.")

    return np_mod.abs(pred_a_np - pred_b_np)


def save_outputs(
    output_dir: str | Path,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    diff: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    np_mod = _require_numpy()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    pred_a_np = np_mod.asarray(pred_a)
    pred_b_np = np_mod.asarray(pred_b)
    diff_np = np_mod.asarray(diff)
    _validate_predictions(pred_a_np, label="pred_A")
    _validate_predictions(pred_b_np, label="pred_B")
    _validate_predictions(diff_np, label="diff")

    pred_a_file = output_path / "pred_A.npy"
    pred_b_file = output_path / "pred_B.npy"
    diff_file = output_path / "diff.npy"
    metadata_file = output_path / "metadata.json"

    np_mod.save(pred_a_file, pred_a_np)
    np_mod.save(pred_b_file, pred_b_np)
    np_mod.save(diff_file, diff_np)

    metadata_payload = dict(metadata)
    metadata_payload.setdefault("diff_mean", float(np_mod.mean(diff_np)))
    metadata_payload.setdefault("diff_max", float(np_mod.max(diff_np)))
    metadata_payload["saved_files"] = {
        "pred_A": str(pred_a_file),
        "pred_B": str(pred_b_file),
        "diff": str(diff_file),
        "metadata": str(metadata_file),
    }

    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRIBE v2 comparison engine.")
    parser.add_argument("video_a", type=str, help="Path to first video file.")
    parser.add_argument("video_b", type=str, help="Path to second video file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save outputs (default: outputs).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Cache directory for model files (default: ./cache).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    np_mod = _require_numpy()

    video_a_path = _validate_video_path(args.video_a)
    video_b_path = _validate_video_path(args.video_b)

    model = load_model(cache_dir=args.cache_dir)

    pred_a, _segments_a = run_video_inference(model=model, video_path=video_a_path)
    pred_b, _segments_b = run_video_inference(model=model, video_path=video_b_path)

    original_shape_a = list(pred_a.shape)
    original_shape_b = list(pred_b.shape)

    aligned_a, aligned_b, _aligned_t = align_predictions(pred_a, pred_b, mode="truncate")
    diff = compute_difference(aligned_a, aligned_b)

    diff_mean = float(np_mod.mean(diff))
    diff_max = float(np_mod.max(diff))
    aligned_shape = list(diff.shape)

    metadata = {
        "video_a_path": str(video_a_path),
        "video_b_path": str(video_b_path),
        "original_shape_a": original_shape_a,
        "original_shape_b": original_shape_b,
        "aligned_shape": aligned_shape,
        "diff_mean": diff_mean,
        "diff_max": diff_max,
    }
    save_outputs(args.output_dir, aligned_a, aligned_b, diff, metadata)

    output_dir = Path(args.output_dir).expanduser().resolve()
    saved_files = [
        output_dir / "pred_A.npy",
        output_dir / "pred_B.npy",
        output_dir / "diff.npy",
        output_dir / "metadata.json",
    ]

    print("Comparison complete.")
    print(f"Original shapes: A={tuple(original_shape_a)}, B={tuple(original_shape_b)}")
    print(f"Aligned shape: {tuple(aligned_shape)}")
    print(f"Diff mean: {diff_mean:.6f}")
    print(f"Diff max: {diff_max:.6f}")
    print("Saved files:")
    for file_path in saved_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()
