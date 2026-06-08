"""Permutation helpers for particle-label actions."""

from __future__ import annotations

import math
import random
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


def count_nonidentity_permutations(n_particles: int) -> int:
    """Return ``n! - 1``, the number of non-identity permutations of `n_particles`.

    Parameters
    ----------
    n_particles : int
        Number of particle labels.

    Returns
    -------
    int
        ``factorial(n_particles) - 1`` (``0`` for ``n_particles <= 1``).
    """

    _validate_size(n_particles)
    return math.factorial(n_particles) - 1


def select_nonidentity_permutations(
    *,
    n_particles: int,
    fraction: float,
    max_count: int,
    seed: int | None = None,
    step: int | None = None,
    max_enumerable: int = 100_000,
) -> list[Permutation]:
    """Select a deterministic subset of distinct non-identity permutations.

    The count is fixed first, then exactly that many distinct permutations are
    sampled; sampling never determines the count::

        n_available = n! - 1
        n_to_test = min(max_count, ceil(fraction * n_available), n_available)

    The ``(seed, step)`` pair seeds a local RNG, so the subset is reproducible:
    the same ``seed`` and ``step`` yield the same subset, while a different
    ``step`` yields a different subset when one exists. The identity is never
    selected.

    Parameters
    ----------
    n_particles : int
        Number of particle labels.
    fraction : float
        Fraction of available non-identity permutations to target, in ``[0, 1]``.
    max_count : int
        Hard cap on the number of permutations (``>= 0``).
    seed : int or None, optional
        Base seed controlling subset selection.
    step : int or None, optional
        Training step, mixed into the seed so checks vary across steps.
    max_enumerable : int, optional
        Above this many available permutations, sample by rejection instead of
        enumerating all ``n!`` permutations.

    Returns
    -------
    list of Permutation
        Distinct non-identity permutations; length is the deterministic count.
    """

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    if max_count < 0:
        raise ValueError(f"max_count must be nonnegative, got {max_count}")

    n_available = count_nonidentity_permutations(n_particles)
    if n_available <= 0:
        return []
    n_to_test = min(max_count, math.ceil(fraction * n_available), n_available)
    if n_to_test <= 0:
        return []

    rng = random.Random((0 if seed is None else int(seed)) * 1_000_003 + (0 if step is None else int(step)))
    identity = tuple(range(n_particles))

    if n_available <= max_enumerable:
        candidates = [image for image in permutations(range(n_particles)) if image != identity]
        rng.shuffle(candidates)
        chosen = candidates[:n_to_test]
        return [Permutation(image) for image in chosen]

    seen: set[tuple[int, ...]] = set()
    chosen_perms: list[Permutation] = []
    while len(chosen_perms) < n_to_test:
        image = list(range(n_particles))
        rng.shuffle(image)
        candidate = tuple(image)
        if candidate == identity or candidate in seen:
            continue
        seen.add(candidate)
        chosen_perms.append(Permutation(candidate))
    return chosen_perms


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
    "count_nonidentity_permutations",
    "reversal_permutation",
    "select_nonidentity_permutations",
]
