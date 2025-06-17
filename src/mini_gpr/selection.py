import warnings
from typing import Protocol

import numpy as np
from jaxtyping import Float


class Selector(Protocol):
    def __call__(
        self, X: Float[np.ndarray, "A D"], n: int
    ) -> Float[np.ndarray, "n D"]: ...


class RandomSelector(Selector):
    def __init__(self, seed: int = 0):
        self.seed = seed

    def __call__(
        self, X: Float[np.ndarray, "A D"], n: int
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
