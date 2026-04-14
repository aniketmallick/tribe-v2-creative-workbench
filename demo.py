from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import app as comparison_app

DEFAULT_SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"
REQUIRED_DEMO_FILES = ("pred_A.npy", "pred_B.npy", "diff.npy")


def find_missing_required_assets(sample_dir: str | Path) -> list[str]:
    sample_path = Path(sample_dir).expanduser().resolve()
    return [name for name in REQUIRED_DEMO_FILES if not (sample_path / name).is_file()]


def _load_optional_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"metadata.json must contain a JSON object, got {type(payload).__name__}.")
    return payload


def load_demo_assets(sample_dir: str | Path = DEFAULT_SAMPLE_DIR) -> dict[str, Any]:
    sample_path = Path(sample_dir).expanduser().resolve()
    missing = find_missing_required_assets(sample_path)
    if missing:
        raise FileNotFoundError(
            f"Missing required demo files in {sample_path}: {', '.join(missing)}."
        )

    pred_a = np.load(sample_path / "pred_A.npy", allow_pickle=False)
    pred_b = np.load(sample_path / "pred_B.npy", allow_pickle=False)
    diff = np.load(sample_path / "diff.npy", allow_pickle=False)
    metadata = _load_optional_metadata(sample_path / "metadata.json")

    data: dict[str, Any] = {
        "pred_a": pred_a,
        "pred_b": pred_b,
        "diff": diff,
        "metadata": metadata,
    }

    for video_key, filename in (("video_a", "ad_a.mp4"), ("video_b", "ad_b.mp4")):
        p = sample_path / filename
        if p.is_file():
            data[video_key] = p

    for timing_key, filename in (("timing_a", "timing_a.json"), ("timing_b", "timing_b.json")):
        p = sample_path / filename
        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                data[timing_key] = json.load(f)

    return data


def _print_missing_assets_instructions(sample_path: Path, missing: list[str]) -> None:
    print(f"Demo assets not found in: {sample_path}")
    print(f"Missing: {', '.join(missing)}")
    print("Create them by running one real comparison and saving outputs to sample_data/:")
    print(f"  python compare.py /path/to/ad_a.mp4 /path/to/ad_b.mp4 --output-dir {sample_path}")
    print("Optional files for richer demo context: metadata.json, ad_a.mp4, ad_b.mp4")


def main() -> int:
    sample_path = DEFAULT_SAMPLE_DIR.expanduser().resolve()
    missing = find_missing_required_assets(sample_path)
    if missing:
        _print_missing_assets_instructions(sample_path, missing)
        return 1

    try:
        demo_data = load_demo_assets(sample_path)
        demo_app = comparison_app.build_app(demo_data=demo_data)
    except Exception as exc:
        print(f"Failed to load demo mode from {sample_path}: {exc}")
        return 1

    demo_app.launch(js=getattr(demo_app, "_tribe_launch_js", None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
