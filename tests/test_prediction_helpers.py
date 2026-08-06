from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.common.prediction import run_torch_prediction, write_prediction_artifacts


class IdentityLogitModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def test_shared_prediction_helper_writes_and_merges_artifacts(tmp_path: Path) -> None:
    loader = [
        (
            torch.tensor([[4.0, 1.0], [0.5, 2.0]]),
            torch.tensor([0, 1]),
            torch.tensor([101, 102]),
        )
    ]
    result = run_torch_prediction(
        IdentityLogitModel(),
        loader,
        device=torch.device("cpu"),
        method_name="identity",
    )
    assert result.accuracy == 1.0
    assert result.scores.shape == (2, 2)

    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps({"training_time_seconds": 12.5, "hardware": {"gpu": None}}),
        encoding="utf-8",
    )
    write_prediction_artifacts(
        tmp_path,
        result,
        hardware={"gpu": "should-not-overwrite"},
        software={"python": "test"},
        prediction_config={"batch_size": 2},
    )

    with (tmp_path / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["image_id"] for row in rows] == ["101", "102"]

    scores = np.load(tmp_path / "scores.npz")
    np.testing.assert_array_equal(scores["class_indices"], [0, 1])

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert runtime["training_time_seconds"] == 12.5
    assert runtime["hardware"] == {"gpu": None}
    assert runtime["method_name"] == "identity"
    assert runtime["num_test_images"] == 2
