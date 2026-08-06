from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_traditional_sweep import (
    DEFAULT_CONFIG,
    build_runs,
    load_config,
    run_sweep,
)


def test_sweep_config_has_unique_non_redundant_runs() -> None:
    config = load_config(DEFAULT_CONFIG)
    hog_runs = build_runs(config, "hog")
    sift_runs = build_runs(config, "sift")

    assert [run.name for run in hog_runs] == [
        "baseline",
        "cell-16",
        "trees-100",
        "trees-300",
        "depth-20",
        "depth-50",
    ]
    assert len(sift_runs) == 9
    assert len({run.command for run in hog_runs}) == len(hog_runs)
    assert all("--output-dir" in run.command for run in hog_runs + sift_runs)
    assert all("--output_dir" not in run.command for run in hog_runs + sift_runs)


def test_sweep_selection_and_dry_run(capsys) -> None:
    config = load_config(DEFAULT_CONFIG)
    runs = build_runs(config, "hog", ["trees-100"])
    assert [run.name for run in runs] == ["trees-100"]
    assert run_sweep(runs, dry_run=True) == 0
    assert "trees-100" in capsys.readouterr().out


def test_sweep_rejects_unknown_run() -> None:
    config = load_config(DEFAULT_CONFIG)
    with pytest.raises(ValueError, match="Unknown hog sweep runs"):
        build_runs(config, "hog", ["missing"])


def test_sweep_config_path_is_repository_relative() -> None:
    expected = Path(__file__).resolve().parents[1] / "configs" / "traditional_sweeps.yaml"
    assert DEFAULT_CONFIG == expected
