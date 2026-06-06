"""Permutation helpers for particle-label actions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations


@dataclass(frozen=True)
class Permutation:
    """Represent a zero-based permutation by its image tuple.

    Parameters
    ----------
    image : tuple of int
        Image of each zero-based index. The entries must be a bijection of
        ``range(len(image))``.
    """

    image: tuple[int, ...]

    def __post_init__(self) -> None:
        image = tuple(self.image)
        if any(not isinstance(item, int) or isinstance(item, bool) for item in image):
            raise ValueError(f"Permutation.image entries must be integers, got {image}")
        if sorted(image) != list(range(len(image))):
            raise ValueError(
                "Permutation.image must be a zero-based bijection of "
                f"range({len(image)}), got {image}"
            )
        object.__setattr__(self, "image", image)

    def __len__(self) -> int:
        """Return the permutation size.

        Returns
        -------
        int
            Number of permuted indices.
        """

        return len(self.image)

    @classmethod
    def identity(cls, size: int) -> "Permutation":
        """Return the identity permutation.

        Parameters
        ----------
        size : int
            Number of indices fixed by the permutation.

        Returns
        -------
        Permutation
            Identity permutation on ``range(size)``.
        """

        if size < 0:
            raise ValueError(f"Permutation size must be non-negative, got {size}")
        return cls(tuple(range(size)))

    def apply_index(self, index: int) -> int:
        """Apply the permutation to one index.

        Parameters
        ----------
        index : int
            Zero-based source index.

        Returns
        -------
        int
            Image of `index`.
        """

        return self.image[index]

    def apply_tuple(self, indices: tuple[int, ...]) -> tuple[int, ...]:
        """Apply the permutation elementwise to an index tuple.

        Parameters
        ----------
        indices : tuple of int
            Tuple of zero-based indices.

        Returns
        -------
        tuple of int
            Tuple with the permutation applied to every entry.
        """

        return tuple(self.apply_index(index) for index in indices)

    def inverse(self) -> "Permutation":
        """Return the inverse permutation.

        Returns
        -------
        Permutation
            Permutation that undoes this permutation.
        """

        inverse_image = [0] * len(self)
        for source, target in enumerate(self.image):
            inverse_image[target] = source
        return Permutation(tuple(inverse_image))

    def compose(self, other: "Permutation") -> "Permutation":
        """Return this permutation after another permutation.

        Parameters
        ----------
        other : Permutation
            Permutation applied first.

        Returns
        -------
        Permutation
            Composition ``self`` after `other`.
        """

        if len(self) != len(other):
            raise ValueError(f"Cannot compose permutations of sizes {len(self)} and {len(other)}")
        return Permutation(tuple(self.apply_index(other.apply_index(index)) for index in range(len(self))))

    @property
    def sign(self) -> int:
        """Return the permutation parity.

        Returns
        -------
        int
            ``1`` for even permutations and ``-1`` for odd permutations.
        """

        inversions = 0
        for left, left_image in enumerate(self.image):
            for right_image in self.image[left + 1 :]:
                if left_image > right_image:
                    inversions += 1
        return -1 if inversions % 2 else 1


def all_permutations(size: int) -> tuple[Permutation, ...]:
    """Return every permutation of ``range(size)`` in lexicographic order.

    Parameters
    ----------
    size : int
        Permutation size.

    Returns
    -------
    tuple of Permutation
        Full symmetric group in deterministic order.
    """

    _validate_size(size)
    return tuple(Permutation(tuple(image)) for image in permutations(range(size)))


def adjacent_transpositions(size: int) -> tuple[Permutation, ...]:
    """Return adjacent transpositions of ``range(size)``.

    Parameters
    ----------
    size : int
        Permutation size.

    Returns
    -------
    tuple of Permutation
        Adjacent swaps ``(0 1), (1 2), ...`` in deterministic order.
    """

    _validate_size(size)
    base = list(range(size))
    images: list[tuple[int, ...]] = []
    for idx in range(size - 1):
        image = base.copy()
        image[idx], image[idx + 1] = image[idx + 1], image[idx]
        images.append(tuple(image))
    return _unique_permutations(images)


def reversal_permutation(size: int) -> Permutation:
    """Return the reversal permutation on ``range(size)``."""

    _validate_size(size)
    return Permutation(tuple(reversed(range(size))))


def _validate_size(size: int) -> None:
    if size < 0:
        raise ValueError(f"Permutation size must be nonnegative, got {size}")


def _unique_permutations(images: list[tuple[int, ...]] | tuple[tuple[int, ...], ...]) -> tuple[Permutation, ...]:
    seen: set[tuple[int, ...]] = set()
    unique: list[Permutation] = []
    for image in images:
        if image not in seen:
            seen.add(image)
            unique.append(Permutation(image))
    return tuple(unique)


__all__ = [
    "Permutation",
    "adjacent_transpositions",
    "all_permutations",
    "reversal_permutation",
]
