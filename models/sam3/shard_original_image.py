#!/usr/bin/env python3
"""Split one safetensors checkpoint into Transformers-compatible shards.

The source checkpoint is removed after successful sharding.
Usage: uv run python models/sam3/shard_original_image.py PATH
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

MAX_SHARD_SIZE = 1_000_000_000

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}


def _tensor_nbytes(tensor_slice) -> int:
    dtype = tensor_slice.get_dtype()
    try:
        bytes_per_element = _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype {dtype!r}") from exc
    return math.prod(tensor_slice.get_shape()) * bytes_per_element


def _group_tensors(tensor_sizes: dict[str, int]) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    current_size = 0

    for name, size in sorted(
        tensor_sizes.items(), key=lambda item: (-item[1], item[0])
    ):
        if current and current_size + size > MAX_SHARD_SIZE:
            groups.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += size

    if current:
        groups.append(current)
    return groups


def shard_model(source: Path) -> list[Path]:
    source = source.expanduser()
    if not source.is_file():
        raise ValueError(f"source is not a file: {source}")
    source = source.resolve()

    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        names = list(checkpoint.keys())
        if not names:
            raise ValueError(f"source contains no tensors: {source}")
        metadata = checkpoint.metadata()
        tensor_sizes = {
            name: _tensor_nbytes(checkpoint.get_slice(name)) for name in names
        }
        groups = _group_tensors(tensor_sizes)
        total_size = sum(tensor_sizes.values())

        shard_count = len(groups)
        shard_paths = [
            source.parent / f"model-{number:05d}-of-{shard_count:05d}.safetensors"
            for number in range(1, shard_count + 1)
        ]
        index_path = source.parent / "model.safetensors.index.json"
        destinations = [*shard_paths, index_path]
        if any(path.resolve() == source for path in destinations):
            raise ValueError("refusing to overwrite the source with an output path")
        existing = [path for path in destinations if path.exists()]
        if existing:
            names = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"output already exists: {names}")

        weight_map = {
            name: shard_paths[index].name
            for index, group in enumerate(groups)
            for name in group
        }
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}

        temporary_dir = Path(
            tempfile.mkdtemp(prefix=".shard-original-image-", dir=source.parent)
        )
        try:
            for number, group in enumerate(groups, start=1):
                tensors = {name: checkpoint.get_tensor(name) for name in group}
                temporary_path = temporary_dir / shard_paths[number - 1].name
                save_file(tensors, temporary_path, metadata=metadata or None)
                del tensors

            temporary_index = temporary_dir / index_path.name
            with temporary_index.open("w", encoding="utf-8") as handle:
                json.dump(index, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")

            for shard_path in shard_paths:
                os.replace(temporary_dir / shard_path.name, shard_path)
            os.replace(temporary_index, index_path)
            source.unlink()
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)

    return [*shard_paths, index_path]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("source", type=Path, help="original .safetensors checkpoint")
    args = parser.parse_args(argv)
    outputs = shard_model(args.source)
    print(
        f"Wrote {len(outputs) - 1} shards and {outputs[-1].name} in {outputs[-1].parent}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
