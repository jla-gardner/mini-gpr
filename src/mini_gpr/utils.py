# ruff: noqa: F722, F821

from functools import wraps
from inspect import signature

import numpy as np


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


def ensure_1d(*var_names):
    """
    Utility decorator to ensure that the input variables are 1D arrays.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            for name in var_names:
                X = bound.arguments[name]
                if len(X.shape) == 2:
                    bound.arguments[name] = X.flatten()
            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


def get_rng(rng: np.random.RandomState | int | None = None):
    if rng is None:
        rng = np.random.RandomState()
    if isinstance(rng, int):
        rng = np.random.RandomState(rng)
    return rng
