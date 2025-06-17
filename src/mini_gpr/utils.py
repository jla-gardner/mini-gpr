# ruff: noqa: F722, F821

from abc import ABC, abstractmethod
from functools import wraps
from inspect import signature

import numpy as np
from jaxtyping import Float


def ensure_2d(*var_names):
    """
    Utility decorator to ensure that the input variables are 2D arrays.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for name in var_names:
                X = bound.arguments[name]
                if len(X.shape) == 1:
                    bound.arguments[name] = X[:, None]

            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


class Model(ABC):
    @abstractmethod
    def fit(
        self, X: Float[np.ndarray, "A D"], y: Float[np.ndarray, "A"]
    ): ...

    @abstractmethod
    def predict(
        self, X: Float[np.ndarray, "B D"]
    ) -> Float[np.ndarray, "B"]: ...

    def __call__(
        self, X: Float[np.ndarray, "B D"]
    ) -> Float[np.ndarray, "B"]:
        return self.predict(X)

    def uncertainty(
        self, X: Float[np.ndarray, "B D"]
    ) -> Float[np.ndarray, "B"]:
        return np.nan * np.ones(X.shape[0])
