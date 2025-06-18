# ruff: noqa: F722, F821

from functools import wraps
from inspect import signature
from typing import Protocol

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


class Model(Protocol):
    def fit(
        self, X: Float[np.ndarray, "N D"], y: Float[np.ndarray, "N"]
    ): ...

    def predict(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]: ...


class UncertaintyModel(Model, Protocol):
    def uncertainty(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]: ...


class ModelStack(Model):
    # no type hint here to allow sklearn models to be passed
    def __init__(self, *models):
        self.models = models

    def fit(self, X: Float[np.ndarray, "N D"], y: Float[np.ndarray, "N"]):
        for model in self.models:
            model.fit(X, y)
            y = y - model.predict(X)

    def predict(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]:
        return np.sum([model.predict(T) for model in self.models], axis=0)

    def __getitem__(self, key: int) -> Model:
        return self.models[key]
