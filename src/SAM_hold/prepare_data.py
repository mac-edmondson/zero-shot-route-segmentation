"""Extract the ZIP-wrapped JPEG files supplied with the hold dataset."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from PIL import Image


def extract_image_archives(data_dir: str | Path) -> list[Path]:
    """Replace each valid ZIP-wrapped JPEG with its validated JPEG payload."""
    directory = Path(data_dir)
    extracted: list[Path] = []
    for archive_path in sorted(directory.glob("*.jpg")):
        if not zipfile.is_zipfile(archive_path):
            continue
        temporary_path = archive_path.with_suffix(".jpg.extracting")
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if names != [archive_path.name]:
                raise ValueError(f"{archive_path} must contain exactly {archive_path.name}.")
            with archive.open(archive_path.name) as source, temporary_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        try:
            with Image.open(temporary_path) as image:
                image.verify()
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        temporary_path.replace(archive_path)
        extracted.append(archive_path)
    return extracted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", default=Path(__file__).parent / "data", type=Path)
    args = parser.parse_args()
    for path in extract_image_archives(args.data_dir):
        print(f"Extracted {path}")
