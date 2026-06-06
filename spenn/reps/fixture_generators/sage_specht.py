"""Optional SageMath Specht fixture generation.

Run this module from the project Python and point it at a Sage executable, for
example::

    uv run python -m spenn.reps.fixture_generators.sage_specht \
      --sage-executable /n/sw/sage-10.3/sage \
      --max-order 3 \
      --out-json spenn/cache/irreps.json \
      --out-cache spenn/cache/irreps_m3.pt

Normal runtime modules and tests must not import Sage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from spenn.data.partition import integer_partitions
from spenn.data.permutation import Permutation, all_permutations
from spenn.reps.irreps import IrrepMetadata, dimension_key, irrep_dimension, permutation_cache_key


def require_sage() -> tuple[Any, Any, str]:
    """Return Sage APIs or raise a fixture-generation error."""

    try:
        from sage.all import Permutation as SagePermutation
        from sage.all import SymmetricGroupRepresentation
        try:
            from sage.env import SAGE_VERSION
        except Exception:  # pragma: no cover - depends on Sage installation details
            SAGE_VERSION = "unknown"
    except ImportError as exc:
        raise RuntimeError(
            "SageMath is required only for irrep fixture generation. "
            "Run from project Python with `--sage-executable`, or run inside "
            "Sage's Python with torch available."
        ) from exc
    return SagePermutation, SymmetricGroupRepresentation, SAGE_VERSION


def generate_sage_irrep_matrix_data(
    *,
    max_order: int = 3,
) -> dict[str, object]:
    """Generate JSON-serializable Specht matrices using in-process Sage.

    Parameters
    ----------
    max_order : int, optional
        Maximum symmetric-group order to include.

    Returns
    -------
    dict
        Sage-generated representation matrices as nested Python lists.
    """

    if max_order <= 0:
        raise ValueError(f"max_order must be positive, got {max_order}")
    SagePermutation, SymmetricGroupRepresentation, sage_version = require_sage()

    representations: dict[str, dict[str, list[list[float]]]] = {}
    for order in range(1, max_order + 1):
        for partition in integer_partitions(order):
            key = dimension_key(partition)
            representation = SymmetricGroupRepresentation(list(partition.parts), "orthogonal")
            representations[key] = {}
            for permutation in all_permutations(order):
                sage_permutation = SagePermutation(_sage_one_based_image(permutation))
                matrix = representation(sage_permutation)
                representations[key][permutation_cache_key(permutation)] = _sage_matrix_to_rows(matrix)

    return {
        "schema_version": IrrepMetadata.schema_version,
        "max_order": max_order,
        "basis_convention": IrrepMetadata.basis_convention,
        "generator": "sage_specht",
        "sage_version": str(sage_version),
        "representations": representations,
    }


def generate_sage_irrep_matrix_data_subprocess(
    *,
    max_order: int = 3,
    sage_executable: str = "sage",
) -> dict[str, object]:
    """Generate Specht matrices by invoking Sage as a subprocess.

    This is the preferred path on clusters where Sage's Python does not include
    the project's PyTorch dependency.

    Parameters
    ----------
    max_order : int, optional
        Maximum symmetric-group order to include.
    sage_executable : str, optional
        Sage launcher. It must accept ``-python -c``.

    Returns
    -------
    dict
        Sage-generated representation matrices as nested Python lists.
    """

    if max_order <= 0:
        raise ValueError(f"max_order must be positive, got {max_order}")
    env = dict(os.environ)
    env.setdefault("DOT_SAGE", str(Path(tempfile.gettempdir()) / f"spenn-sage-{os.getuid()}"))
    script = _sage_subprocess_script(max_order=max_order)
    result = subprocess.run(
        [sage_executable, "-python", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout)


def generate_sage_irrep_tensor_cache(
    *,
    max_order: int = 3,
    dtype: Any = "float64",
    sage_executable: str | None = None,
) -> dict[str, object]:
    """Generate Specht representation tensor cache entries using SageMath.

    Parameters
    ----------
    max_order : int, optional
        Maximum symmetric-group order to include.
    dtype : str or torch.dtype, optional
        Torch dtype or dtype name for saved matrices.
    sage_executable : str or None, optional
        Sage launcher. When supplied, Sage is invoked as a subprocess and this
        function can be run from the project Python environment.

    Returns
    -------
    dict
        Tensor cache containing Sage-generated representation matrices.
    """

    import torch

    if sage_executable is None:
        try:
            data = generate_sage_irrep_matrix_data(max_order=max_order)
        except RuntimeError:
            data = generate_sage_irrep_matrix_data_subprocess(max_order=max_order)
    else:
        data = generate_sage_irrep_matrix_data_subprocess(
            max_order=max_order,
            sage_executable=sage_executable,
        )
    torch_dtype = _torch_dtype(dtype, torch)
    representations = data["representations"]
    if not isinstance(representations, dict):
        raise TypeError("Sage matrix data must contain a representations mapping")
    tensor_representations: dict[str, dict[str, torch.Tensor]] = {}
    for partition_key, matrices in representations.items():
        if not isinstance(matrices, dict):
            raise TypeError(f"Representation block {partition_key!r} must be a mapping")
        tensor_representations[partition_key] = {
            permutation_key: torch.tensor(matrix, dtype=torch_dtype)
            for permutation_key, matrix in matrices.items()
        }
    return {
        "schema_version": IrrepMetadata.schema_version,
        "max_order": max_order,
        "basis_convention": IrrepMetadata.basis_convention,
        "generator": "sage_specht",
        "sage_version": str(data.get("sage_version", "unknown")),
        "representations": tensor_representations,
    }


def _sage_one_based_image(permutation: Permutation) -> list[int]:
    return [value + 1 for value in permutation.image]


def _sage_matrix_to_rows(matrix) -> list[list[float]]:
    return [[float(entry) for entry in row] for row in matrix]


def _torch_dtype(name: Any, torch_module: Any) -> Any:
    if isinstance(name, torch_module.dtype):
        return name
    if isinstance(name, str):
        try:
            dtype = getattr(torch_module, name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported torch dtype name {name!r}") from exc
        if isinstance(dtype, torch_module.dtype):
            return dtype
    raise ValueError(f"Unsupported torch dtype {name!r}")


def _sage_subprocess_script(*, max_order: int) -> str:
    return textwrap.dedent(
        f"""
        import json
        from itertools import permutations

        from sage.all import Permutation as SagePermutation
        from sage.all import SymmetricGroupRepresentation

        try:
            from sage.env import SAGE_VERSION
        except Exception:
            SAGE_VERSION = "unknown"

        def integer_partitions(order, max_part=None):
            if max_part is None:
                max_part = order
            if order == 0:
                return [()]
            out = []
            for part in range(min(max_part, order), 0, -1):
                for suffix in integer_partitions(order - part, part):
                    out.append((part,) + suffix)
            return out

        def dimension_key(parts):
            return str(sum(parts)) + "|" + ",".join(str(part) for part in parts)

        def permutation_key(image):
            return ",".join(str(value) for value in image)

        representations = {{}}
        for order in range(1, {max_order} + 1):
            for parts in integer_partitions(order):
                key = dimension_key(parts)
                representation = SymmetricGroupRepresentation(list(parts), "orthogonal")
                representations[key] = {{}}
                for image in permutations(range(order)):
                    sage_permutation = SagePermutation([value + 1 for value in image])
                    matrix = representation(sage_permutation)
                    representations[key][permutation_key(image)] = [
                        [float(entry) for entry in row]
                        for row in matrix
                    ]

        print(json.dumps({{
            "schema_version": {IrrepMetadata.schema_version},
            "max_order": {max_order},
            "basis_convention": {IrrepMetadata.basis_convention!r},
            "generator": "sage_specht",
            "sage_version": str(SAGE_VERSION),
            "representations": representations,
        }}, sort_keys=True))
        """
    )


def _write_metadata(
    max_order: int,
    out_json: Path,
    out_cache: Path | None,
    *,
    sage_version: str = "unknown",
) -> None:
    metadata = IrrepMetadata.generate(
        max_order=max_order,
        tensor_cache=None if out_cache is None else out_cache.name,
    )
    data = metadata.to_json_data()
    data["generator"] = "sage_specht"
    data["sage_version"] = sage_version
    data["dimensions"] = {
        dimension_key(partition): irrep_dimension(partition)
        for partitions in metadata.partitions.values()
        for partition in partitions
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> None:
    """Generate irrep JSON and tensor-cache files."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--sage-executable", default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-cache", type=Path, required=True)
    args = parser.parse_args()

    cache = generate_sage_irrep_tensor_cache(
        max_order=args.max_order,
        dtype=args.dtype,
        sage_executable=args.sage_executable,
    )
    import torch

    args.out_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out_cache)
    _write_metadata(
        args.max_order,
        args.out_json,
        args.out_cache,
        sage_version=str(cache.get("sage_version", "unknown")),
    )


if __name__ == "__main__":
    main()
