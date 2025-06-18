import warnings
from typing import Protocol

import numpy as np
from jaxtyping import Float


class Selector(Protocol):
    def __call__(
        self, X: Float[np.ndarray, "N D"], n: int
    ) -> Float[np.ndarray, "n D"]: ...


class RandomSelector(Selector):
    def __init__(self, seed: int = 0):
        self.seed = seed

    def __call__(
        self, X: Float[np.ndarray, "N D"], n: int
    ) -> Float[np.ndarray, "n D"]:
        A = len(X)
        if n > A:
            warnings.warn(
                f"n ({n}) is greater than the number of points ({A}). "
                f"Using all points as inducing points.",
                stacklevel=2,
            )
            return X
        rand = np.random.RandomState(self.seed)
        return X[rand.choice(A, n, replace=False)]


def GridSelector(
    X: Float[np.ndarray, "N D"],
    n: int,
) -> Float[np.ndarray, "n D"]:
    # find the number of points to sample in each dimension, d
    _, D = X.shape
    d = int(np.ceil(n ** (1 / D)))

    # create a grid of points in the unit hypercube
    unit_grid = np.linspace(0, 1, d)
    grid = np.meshgrid(*[unit_grid] * D)
    grid = np.stack(grid, axis=-1)

    # re-scale the grid to the limits of the data
    lo, hi = np.min(X, axis=0), np.max(X, axis=0)
    ranges = hi - lo
    grid = lo + ranges * grid

    return grid
