#!/usr/bin/env python3
"""Run reproducible HOG or SIFT development sweeps from one YAML config."""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.cli import ArgumentParser  # noqa: I001


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "traditional_sweeps.yaml"


@dataclass(frozen=True)
class SweepRun:
    name: str
    command: tuple[str, ...]
    output_dir: Path
    log_file: Path


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_config(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Sweep config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = _require_mapping(payload, str(path))
    if config.get("version") != 1:
        raise ValueError(f"Unsupported sweep config version in {path}")
    _require_mapping(config.get("methods"), "methods")
    return config


def _format_path(template: str, *, method: str, name: str) -> Path:
    rendered = template.format(method=method, name=name)
    path = Path(rendered).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _cli_arguments(values: Mapping[str, Any]) -> list[str]:
    arguments: list[str] = []
    for key, value in values.items():
        if value is None or value is False:
            continue
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            arguments.append(flag)
        elif isinstance(value, list):
            arguments.extend([flag, *(str(item) for item in value)])
        else:
            arguments.extend([flag, str(value)])
    return arguments


def build_runs(
    config: Mapping[str, Any],
    method: str,
    selected_names: Iterable[str] | None = None,
) -> list[SweepRun]:
    methods = _require_mapping(config["methods"], "methods")
    if method not in methods:
        raise ValueError(f"Unknown sweep method {method!r}; choose from {sorted(methods)}")
    method_config = _require_mapping(methods[method], f"methods.{method}")
    module = method_config.get("module")
    if not isinstance(module, str) or not module:
        raise ValueError(f"methods.{method}.module must be a non-empty string")

    common_args = dict(_require_mapping(config.get("common_args", {}), "common_args"))
    fixed_args = dict(
        _require_mapping(method_config.get("fixed_args", {}), f"methods.{method}.fixed_args")
    )
    run_specs = method_config.get("runs")
    if not isinstance(run_specs, list) or not run_specs:
        raise ValueError(f"methods.{method}.runs must be a non-empty list")

    output_template = method_config.get("output_dir")
    log_template = method_config.get("log_file")
    if not isinstance(output_template, str) or not output_template:
        raise ValueError(f"methods.{method}.output_dir must be a non-empty string")
    if not isinstance(log_template, str) or not log_template:
        raise ValueError(f"methods.{method}.log_file must be a non-empty string")
    requested = set(selected_names or [])
    runs: list[SweepRun] = []
    seen: set[str] = set()
    for index, raw_spec in enumerate(run_specs):
        spec = _require_mapping(raw_spec, f"methods.{method}.runs[{index}]")
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"methods.{method}.runs[{index}].name is required")
        if name in seen:
            raise ValueError(f"Duplicate {method} sweep run name: {name}")
        seen.add(name)
        if requested and name not in requested:
            continue

        arguments = {**common_args, **fixed_args}
        arguments.update(_require_mapping(spec.get("args", {}), f"run {name}.args"))
        output_dir = _format_path(output_template, method=method, name=name)
        log_file = _format_path(log_template, method=method, name=name)
        arguments["output_dir"] = str(output_dir)
        runs.append(
            SweepRun(
                name=name,
                command=(sys.executable, "-m", module, *_cli_arguments(arguments)),
                output_dir=output_dir,
                log_file=log_file,
            )
        )

    missing = requested - seen
    if missing:
        raise ValueError(f"Unknown {method} sweep runs: {sorted(missing)}")
    return runs


def run_sweep(
    runs: Iterable[SweepRun],
    *,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> int:
    failures: list[str] = []
    for run in runs:
        print(f"[{run.name}] {shlex.join(run.command)}", flush=True)
        print(f"[{run.name}] log: {run.log_file}", flush=True)
        if dry_run:
            continue

        run.output_dir.mkdir(parents=True, exist_ok=True)
        run.log_file.parent.mkdir(parents=True, exist_ok=True)
        with run.log_file.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                run.command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode == 0:
            print(f"[{run.name}] complete", flush=True)
            continue

        failures.append(run.name)
        print(
            f"[{run.name}] failed with exit code {completed.returncode}; see {run.log_file}",
            file=sys.stderr,
            flush=True,
        )
        if not continue_on_error:
            return completed.returncode

    if failures:
        print(f"Failed sweep runs: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def parse_args() -> Any:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("hog", "sift"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        help="Run only this named configuration; repeat to select multiple runs.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the selected method's run names without executing them.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    runs = build_runs(config, args.method, args.runs)
    if args.list:
        for run in runs:
            print(run.name)
        return 0
    return run_sweep(
        runs,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    raise SystemExit(main())
