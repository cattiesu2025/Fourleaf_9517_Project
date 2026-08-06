from __future__ import annotations

from src.cli import COMMANDS, main
from src.common.cli import ArgumentParser


def test_argument_parser_accepts_new_and_legacy_flag_styles() -> None:
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, required=True)

    assert parser.parse_args(["--batch-size", "8"]).batch_size == 8
    assert parser.parse_args(["--batch_size", "16"]).batch_size == 16


def test_project_cli_lists_commands_without_loading_model_modules(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "sweep-hog" in output
    assert "train-transfer" in output
    assert "run-robustness" in output


def test_sweep_command_injects_the_selected_method() -> None:
    argv = COMMANDS["sweep-hog"].argv(["--dry-run"])
    assert argv[-3:] == ["--method", "hog", "--dry-run"]


def test_project_cli_rejects_unknown_command(capsys) -> None:
    assert main(["does-not-exist"]) == 2
    assert "Unknown command" in capsys.readouterr().err
