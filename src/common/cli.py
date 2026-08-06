"""Command-line helpers shared by the project executables."""

from __future__ import annotations

import argparse
from typing import Any


class ArgumentParser(argparse.ArgumentParser):
    """Use kebab-case flags while accepting legacy snake_case aliases.

    Older project commands exposed options such as ``--batch_size``. New help
    text leads with ``--batch-size``, but both spellings map to the same
    ``args.batch_size`` attribute so existing experiment commands keep working.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)
        kwargs.setdefault("formatter_class", argparse.ArgumentDefaultsHelpFormatter)
        super().__init__(*args, **kwargs)

    def add_argument(self, *name_or_flags: str, **kwargs: Any) -> argparse.Action:
        expanded: list[str] = []
        for name_or_flag in name_or_flags:
            if name_or_flag.startswith("--") and "_" in name_or_flag:
                kebab_case = name_or_flag.replace("_", "-")
                if kebab_case not in expanded:
                    expanded.append(kebab_case)
            if name_or_flag not in expanded:
                expanded.append(name_or_flag)
        return super().add_argument(*expanded, **kwargs)
