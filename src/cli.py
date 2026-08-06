"""Single command router for project workflows."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Command:
    description: str
    module: str | None = None
    prefix_args: tuple[str, ...] = ()
    executable: tuple[str, ...] = ()

    def argv(self, forwarded: Sequence[str]) -> list[str]:
        if self.module is not None:
            prefix = [sys.executable, "-m", self.module]
        else:
            prefix = [part.format(project_root=PROJECT_ROOT) for part in self.executable]
        return [*prefix, *self.prefix_args, *forwarded]


COMMANDS: dict[str, Command] = {
    "check-environment": Command(
        "Validate Python and dependency versions.",
        module="scripts.check_environment",
    ),
    "test": Command("Run the project test suite.", module="pytest"),
    "build-demo": Command("Rebuild the evaluation MVP fixture.", module="scripts.build_mvp_demo"),
    "demo-robustness": Command(
        "Run the dummy robustness matrix.",
        module="demo.evaluation_mvp.dummy_robustness",
    ),
    "build-split": Command(
        "Build the fixed class split and metadata.",
        module="scripts.build_dataset",
    ),
    "build-longtail": Command(
        "Build deterministic long-tail training metadata.",
        module="scripts.build_longtail",
    ),
    "copy-selected-images": Command(
        "Copy split-referenced images into a compact directory.",
        module="scripts.copy_selected_images",
    ),
    "build-full-subset": Command(
        "Build full-training-subset metadata.",
        module="scripts.build_full_train_subset",
    ),
    "extract-full-subset": Command(
        "Stream and extract the selected full-training images.",
        executable=("bash", "{project_root}/scripts/extract_full_train_subset.sh"),
    ),
    "scan-data": Command(
        "Scan dataset integrity and quality.", module="scripts.scan_dataset_quality"
    ),
    "train-hog": Command(
        "Train or evaluate the HOG + Random Forest model.",
        module="src.traditional.hog_random_forest",
    ),
    "train-sift": Command(
        "Train or evaluate the SIFT + BoVW + SVM model.",
        module="src.traditional.sift_bovw_svm",
    ),
    "sweep-hog": Command(
        "Run the configured HOG development sweep.",
        module="scripts.run_traditional_sweep",
        prefix_args=("--method", "hog"),
    ),
    "sweep-sift": Command(
        "Run the configured SIFT development sweep.",
        module="scripts.run_traditional_sweep",
        prefix_args=("--method", "sift"),
    ),
    "train-scratch": Command("Train the scratch ResNet18 model.", module="src.scratch.train"),
    "predict-scratch": Command("Run scratch ResNet18 inference.", module="src.scratch.predict"),
    "train-transfer": Command(
        "Train a transfer-learning ResNet18 model.", module="src.transfer.train"
    ),
    "predict-transfer": Command(
        "Run transfer-learning ResNet18 inference.", module="src.transfer.predict"
    ),
    "evaluate": Command(
        "Evaluate one standard prediction artifact.",
        module="src.evaluation.evaluate",
    ),
    "compare": Command(
        "Build the validated cross-method comparison.",
        module="src.evaluation.compare",
    ),
    "evaluate-robustness": Command(
        "Aggregate existing robustness runs.",
        module="src.evaluation.robustness",
    ),
    "run-robustness": Command(
        "Run the configured real-model robustness matrix.",
        module="scripts.run_final_robustness",
    ),
    "transfer-metrics": Command(
        "Compute transfer-learning metrics.", module="src.transfer.compute_metrics"
    ),
    "transfer-curves": Command(
        "Plot transfer-learning training curves.",
        module="src.transfer.plot_training_curves",
    ),
    "transfer-gradcam": Command(
        "Generate transfer-learning Grad-CAM analysis.",
        module="src.transfer.gradcam_analysis",
    ),
    "transfer-gradcam-compare": Command(
        "Compare two transfer-model Grad-CAM outputs.",
        module="src.transfer.compare_gradcam_models",
    ),
    "transfer-confusions": Command(
        "Analyze transfer-model confusable species pairs.",
        module="src.transfer.confusable_pairs_analysis",
    ),
    "transfer-crop-fusion": Command(
        "Run the Grad-CAM crop-fusion experiment.",
        module="src.transfer.gradcam_crop_fusion",
    ),
}


def format_help() -> str:
    width = max(len(name) for name in COMMANDS)
    commands = "\n".join(
        f"  {name:<{width}}  {command.description}" for name, command in COMMANDS.items()
    )
    return (
        "Usage: comp9517 <command> [arguments]\n\n"
        "Project commands:\n"
        f"{commands}\n\n"
        "Use `comp9517 <command> --help` for command-specific options.\n"
        "Legacy snake_case options remain accepted; new examples use kebab-case."
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        if len(arguments) == 2 and arguments[0] == "help":
            arguments = [arguments[1], "--help"]
        else:
            print(format_help())
            return 0

    command_name, *forwarded = arguments
    command = COMMANDS.get(command_name)
    if command is None:
        print(f"Unknown command: {command_name}\n", file=sys.stderr)
        print(format_help(), file=sys.stderr)
        return 2

    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    completed = subprocess.run(
        command.argv(forwarded),
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
