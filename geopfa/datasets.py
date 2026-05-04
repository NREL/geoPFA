"""Functions to fetch sample data for geoPFA."""

from pathlib import Path
import shutil

import pooch
from pooch.processors import Unzip


dogbert_newberry = pooch.create(
    path=pooch.os_cache("geoPFA"),
    base_url="https://github.com/NatLabRockies/geoPFA/releases/download/{version}/",
    version="v0.0.20",
    registry={
        "newberry_tutorial_data.zip": "sha256:d9168678b1e63f52a2e73b18d8b26e93ac502b3025e197391ccb26f3081f6c4c",
    },
)


def setup_newberry_tutorial_data(target_dir: Path) -> None:
    """
    Download and extract the Newberry tutorial dataset into a target directory.

    The contents of the zip are copied into `target_dir` exactly as structured.
    """
    filename = "newberry_tutorial_data.zip"

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = dogbert_newberry.fetch(filename, processor=Unzip())

    if not extracted_files:
        raise RuntimeError(f"No files extracted from {filename}")

    extracted_paths = [Path(p).resolve() for p in extracted_files]

    # Find the common extraction root of all extracted files
    common_root = Path(
        __import__("os").path.commonpath([str(p) for p in extracted_paths])
    )

    # If the zip contains a single top-level folder (for example "data/"),
    # unwrap that folder so we do not create data/data/.
    top_level_items = list(common_root.iterdir())
    if len(top_level_items) == 1 and top_level_items[0].is_dir():
        source_dir = top_level_items[0]
    else:
        source_dir = common_root

    # Copy the full directory structure into target_dir
    for item in source_dir.iterdir():
        dest = target_dir / item.name

        if dest.exists():
            continue

        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    print(f"Data extracted to: {target_dir}")
