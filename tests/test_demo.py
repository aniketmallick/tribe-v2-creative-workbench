from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("gradio")

import compare
import demo


def _write_demo_arrays(output_dir: Path) -> None:
    pred_a = np.zeros((2, compare.EXPECTED_VERTICES), dtype=float)
    pred_b = np.ones((2, compare.EXPECTED_VERTICES), dtype=float)
    diff = np.abs(pred_a - pred_b)
    np.save(output_dir / "pred_A.npy", pred_a)
    np.save(output_dir / "pred_B.npy", pred_b)
    np.save(output_dir / "diff.npy", diff)


def test_load_demo_assets_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing required demo files"):
        demo.load_demo_assets(tmp_path)


def test_find_missing_required_assets_reports_expected_files(tmp_path: Path) -> None:
    missing_before = demo.find_missing_required_assets(tmp_path)
    assert missing_before == ["pred_A.npy", "pred_B.npy", "diff.npy"]

    _write_demo_arrays(tmp_path)
    missing_after = demo.find_missing_required_assets(tmp_path)
    assert missing_after == []


def test_load_demo_assets_success_with_optional_files(tmp_path: Path) -> None:
    _write_demo_arrays(tmp_path)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"campaign": "demo", "version": 2}),
        encoding="utf-8",
    )
    (tmp_path / "ad_a.mp4").write_bytes(b"a")
    (tmp_path / "ad_b.mp4").write_bytes(b"b")

    assets = demo.load_demo_assets(tmp_path)

    assert assets["pred_a"].shape == (2, compare.EXPECTED_VERTICES)
    assert assets["pred_b"].shape == (2, compare.EXPECTED_VERTICES)
    assert assets["diff"].shape == (2, compare.EXPECTED_VERTICES)
    assert assets["metadata"]["campaign"] == "demo"
    assert Path(assets["video_a"]).name == "ad_a.mp4"
    assert Path(assets["video_b"]).name == "ad_b.mp4"


def test_main_returns_one_when_assets_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(demo, "DEFAULT_SAMPLE_DIR", tmp_path)
    exit_code = demo.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Demo assets not found in:" in captured.out


def test_main_builds_and_launches_when_assets_exist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_demo_arrays(tmp_path)
    monkeypatch.setattr(demo, "DEFAULT_SAMPLE_DIR", tmp_path)

    launched = {"called": False}

    class DummyApp:
        def launch(self) -> None:
            launched["called"] = True

    def fake_build_app(*, demo_data):
        assert demo_data["pred_a"].shape == (2, compare.EXPECTED_VERTICES)
        return DummyApp()

    monkeypatch.setattr(demo.comparison_app, "build_app", fake_build_app)
    exit_code = demo.main()

    assert exit_code == 0
    assert launched["called"] is True
