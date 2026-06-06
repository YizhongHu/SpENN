"""Symmetric-group irrep metadata helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from spenn.data.partition import Partition, as_partition, integer_partitions
from spenn.data.permutation import Permutation


@dataclass(frozen=True)
class SpechtIrrepInfo:
    """Store lightweight metadata for one Specht irrep.

    Parameters
    ----------
    partition : Partition
        Specht partition label.
    dimension : int
        Dimension of the local Specht irrep.
    basis : str, optional
        Basis convention for representation matrices. The scaffold only
        supports ``"orthogonal"``.
    """

    partition: Partition
    dimension: int
    basis: str = "orthogonal"

    @property
    def order(self) -> int:
        """Return the partition order.

        Returns
        -------
        int
            Integer partitioned by :attr:`partition`.
        """

        return self.partition.order


class IrrepMetadata:
    """Thin reader/constructor for saved Specht irrep metadata.

    Parameters
    ----------
    path : str or pathlib.Path
        JSON metadata file. Tensor-valued caches are optional and loaded only
        when the JSON names an existing cache file.
    """

    schema_version = 1
    partition_order_version = "lexicographic-v1"
    basis_convention = "orthogonal"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = self._load_json(self.path)
        self.max_order = int(self.data["max_order"])
        self.partitions = self._parse_partitions(self.data["partitions"])
        self.dimensions = self._parse_dimensions(self.data["dimensions"])
        self.tensor_cache_path = self._tensor_cache_path(self.data)
        self.tensor_cache = self._load_tensor_cache(self.tensor_cache_path)

    @classmethod
    def load(cls, path: str | Path) -> "IrrepMetadata":
        """Load irrep metadata from `path`."""

        return cls(path)

    @classmethod
    def generate(cls, *, max_order: int, tensor_cache: str | None = "irreps_m3.pt") -> "IrrepMetadata":
        """Generate deterministic discrete irrep metadata.

        Parameters
        ----------
        max_order : int
            Maximum symmetric-group order.
        tensor_cache : str or None, optional
            Optional cache filename recorded in JSON. The file is not generated
            implicitly.
        """

        if max_order <= 0:
            raise ValueError(f"max_order must be positive, got {max_order}")
        metadata = cls.__new__(cls)
        metadata.path = None
        metadata.max_order = int(max_order)
        metadata.partitions = {order: list(integer_partitions(order)) for order in range(1, max_order + 1)}
        metadata.dimensions = {
            dimension_key(partition): irrep_dimension(partition)
            for partitions in metadata.partitions.values()
            for partition in partitions
        }
        metadata.tensor_cache_path = None if tensor_cache is None else Path(tensor_cache)
        metadata.tensor_cache = {}
        metadata.data = metadata.to_json_data()
        return metadata

    def save(self, path: str | Path) -> None:
        """Write this metadata as stable JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_json_data(), indent=2, sort_keys=True) + "\n")
        self.path = output

    def dimension(self, partition: Partition | tuple[int, ...] | list[int] | str | int) -> int:
        """Return the dimension recorded for `partition`."""

        key = dimension_key(as_partition(partition))
        try:
            return self.dimensions[key]
        except KeyError as exc:
            raise KeyError(f"Partition {key} is absent from irrep metadata") from exc

    def representation_matrix(
        self,
        partition: Partition | tuple[int, ...] | list[int] | str | int,
        permutation: Permutation,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return a cached Specht representation matrix.

        Parameters
        ----------
        partition : Partition or partition-like
            Specht partition label.
        permutation : Permutation
            Permutation represented in the cached basis.
        dtype : torch.dtype or None, optional
            Optional output dtype.
        device : torch.device, str, or None, optional
            Optional output device.

        Returns
        -------
        torch.Tensor
            Cached representation matrix.
        """

        normalized = as_partition(partition)
        if len(permutation) != normalized.order:
            raise ValueError(
                f"Permutation size {len(permutation)} does not match partition order {normalized.order}"
            )
        representations = self._cache_mapping("representations")
        key = dimension_key(normalized)
        permutation_key = permutation_cache_key(permutation)
        try:
            matrix = representations[key][permutation_key]
        except KeyError as exc:
            raise KeyError(
                f"Representation matrix for partition {key} and permutation {permutation_key} "
                "is absent from irrep tensor cache"
            ) from exc
        return matrix.to(device=device, dtype=dtype)

    def _cache_mapping(self, name: str) -> dict[str, Any]:
        value = self.tensor_cache.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"Irrep tensor cache does not contain mapping {name!r}")
        return value

    def to_json_data(self) -> dict[str, object]:
        """Return JSON-serializable metadata."""

        data: dict[str, object] = {
            "schema_version": self.schema_version,
            "max_order": self.max_order,
            "partition_order_version": self.partition_order_version,
            "basis_convention": self.basis_convention,
            "partitions": {
                str(order): [list(partition.parts) for partition in partitions]
                for order, partitions in sorted(self.partitions.items())
            },
            "dimensions": dict(sorted(self.dimensions.items())),
        }
        if self.tensor_cache_path is not None:
            data["tensor_cache"] = self.tensor_cache_path.name
        return data

    @classmethod
    def _load_json(cls, path: Path) -> dict[str, object]:
        data = json.loads(path.read_text())
        if int(data.get("schema_version", -1)) != cls.schema_version:
            raise ValueError(f"Unsupported irrep metadata schema in {path}")
        if data.get("partition_order_version") != cls.partition_order_version:
            raise ValueError(f"Unsupported partition ordering in {path}")
        if data.get("basis_convention") != cls.basis_convention:
            raise ValueError(f"Unsupported irrep basis convention in {path}")
        return data

    @staticmethod
    def _parse_partitions(data: object) -> dict[int, list[Partition]]:
        if not isinstance(data, dict):
            raise TypeError("irrep partitions metadata must be a mapping")
        return {
            int(order): [Partition(tuple(int(part) for part in parts)) for parts in partitions]
            for order, partitions in data.items()
        }

    @staticmethod
    def _parse_dimensions(data: object) -> dict[str, int]:
        if not isinstance(data, dict):
            raise TypeError("irrep dimensions metadata must be a mapping")
        return {str(key): int(value) for key, value in data.items()}

    def _tensor_cache_path(self, data: dict[str, object]) -> Path | None:
        cache = data.get("tensor_cache")
        if cache is None:
            return None
        cache_path = Path(str(cache))
        if not cache_path.is_absolute() and self.path is not None:
            cache_path = self.path.parent / cache_path
        return cache_path

    @staticmethod
    def _load_tensor_cache(path: Path | None) -> dict[str, object]:
        if path is None or not path.exists():
            return {}
        return torch.load(path, map_location="cpu")


class SpechtIrrep:
    """Placeholder symmetric-group irrep object.

    Parameters
    ----------
    partition : Partition or partition-like
        Specht partition label.
    dimension : int or None, optional
        Irrep dimension. If ``None``, use the scaffold tail-shape convention.
    basis : str, optional
        Basis convention. Only ``"orthogonal"`` is supported for new Specht
        module fixtures.
    """

    def __init__(
        self,
        partition: Partition | tuple[int, ...] | list[int] | str | int,
        dimension: int | None = None,
        *,
        basis: str = "orthogonal",
    ) -> None:
        self.partition = as_partition(partition)
        self.dimension = irrep_dimension(self.partition) if dimension is None else int(dimension)
        if basis != "orthogonal":
            raise ValueError("SpechtIrrep scaffold only supports the orthogonal basis")
        self.basis = basis

    @property
    def order(self) -> int:
        """Return the partition order.

        Returns
        -------
        int
            Integer partitioned by :attr:`partition`.
        """

        return self.partition.order

    def metadata(self) -> SpechtIrrepInfo:
        """Return static metadata for this irrep.

        Returns
        -------
        SpechtIrrepInfo
            Partition and dimension metadata.
        """

        return SpechtIrrepInfo(partition=self.partition, dimension=self.dimension, basis=self.basis)

    def representation(self, permutation: Permutation) -> torch.Tensor:
        """Return the representation matrix for `permutation`.

        Parameters
        ----------
        permutation : Permutation
            Permutation to represent.

        Returns
        -------
        torch.Tensor
            Orthogonal-basis representation matrix.

        """

        return load_default_irrep_metadata().representation_matrix(self.partition, permutation)


def irrep_dimension(partition: Partition) -> int:
    """Return the Specht irrep dimension by the hook-length formula.

    Parameters
    ----------
    partition : Partition
        Specht partition label.

    Returns
    -------
    int
        Dimension used by the existing tensor tail convention.
    """

    if partition.order == 0:
        return 1
    factorial = 1
    for value in range(2, partition.order + 1):
        factorial *= value
    hook_product = 1
    for row, row_length in enumerate(partition.parts):
        for col in range(row_length):
            below = sum(1 for lower in partition.parts[row + 1 :] if lower > col)
            right = row_length - col - 1
            hook_product *= right + below + 1
    return factorial // hook_product


def dimension_key(partition: Partition) -> str:
    """Return the stable dimension/cache key for a Specht partition."""

    return f"{partition.order}|{','.join(str(part) for part in partition.parts)}"


def permutation_cache_key(permutation: Permutation) -> str:
    """Return the stable cache key for a zero-based permutation."""

    return ",".join(str(value) for value in permutation.image)


def default_irrep_metadata_path() -> Path:
    """Return the checked-in irrep metadata cache path."""

    return Path(__file__).resolve().parents[1] / "cache" / "irreps.json"


def load_default_irrep_metadata() -> IrrepMetadata:
    """Load checked-in irrep metadata and tensor cache files."""

    return IrrepMetadata.load(default_irrep_metadata_path())


def generate_irrep_tensor_cache(
    *,
    max_order: int = 3,
    dtype: torch.dtype | str = torch.float64,
    sage_executable: str | None = None,
) -> dict[str, object]:
    """Generate tensor-valued irrep metadata cache entries with SageMath.

    Parameters
    ----------
    max_order : int, optional
        Maximum symmetric-group order to include. The current small fixture set
        supports orders up to 3.
    dtype : torch.dtype or str, optional
        Tensor dtype for saved matrices.
    sage_executable : str or None, optional
        Sage launcher. When supplied, Sage is invoked as a subprocess.

    Returns
    -------
    dict
        Cache containing representation matrices and scaffold Fourier maps.
    """

    from spenn.reps.fixture_generators.sage_specht import generate_sage_irrep_tensor_cache

    return generate_sage_irrep_tensor_cache(
        max_order=max_order,
        dtype=dtype,
        sage_executable=sage_executable,
    )


def specht_irrep(
    partition: Partition | tuple[int, ...] | list[int] | str | int,
    *,
    basis: str = "orthogonal",
) -> SpechtIrrep:
    """Construct a :class:`SpechtIrrep` placeholder.

    Parameters
    ----------
    partition : Partition or partition-like
        Specht partition label.
    basis : str, optional
        Basis convention. Only ``"orthogonal"`` is supported.

    Returns
    -------
    SpechtIrrep
        Placeholder irrep object.
    """

    return SpechtIrrep(partition, basis=basis)


__all__ = [
    "IrrepMetadata",
    "SpechtIrrep",
    "SpechtIrrepInfo",
    "dimension_key",
    "load_default_irrep_metadata",
    "permutation_cache_key",
    "generate_irrep_tensor_cache",
    "irrep_dimension",
    "specht_irrep",
]
