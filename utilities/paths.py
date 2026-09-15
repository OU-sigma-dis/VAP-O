"""Portable paths for locally prepared, non-redistributed resources."""

import os
from pathlib import Path


def switchboard_dataset_root() -> Path:
    """Return the prepared Switchboard dataset directory.

    Set ``VAPO_DATA_ROOT`` to override the default relative location.
    The corpus and derived features are intentionally not distributed with VAP-O.
    """
    return Path(
        os.environ.get("VAPO_DATA_ROOT", "../data/switchboard/vap-o_dataset")
    ).expanduser()


def switchboard_paths() -> tuple[Path, Path, Path, Path]:
    """Return train CSV, validation CSV, test CSV, and CPC-feature directory."""
    root = switchboard_dataset_root()
    return root / "train.csv", root / "val.csv", root / "test.csv", root / "cpc_features"
