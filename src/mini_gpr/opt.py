from typing import Protocol

import numpy as np
from jaxtyping import Float

from mini_gpr.models import Model


class Objective(Protocol):
    """
    An objective function takes a model as input, and returns a scalar value,
    such that a lower value is a "better" model.
    """

    def __call__(self, model: Model) -> float: ...


def maximise_log_likelihood(model: Model):
    """Maximise the log likelihood of the model."""
    return -model.log_likelihood


def validation_set_mse(
    X: Float[np.ndarray, "N D"],
    y: Float[np.ndarray, "N"],
) -> Objective:
    """Minimise the mean squared error of the model on the validation set."""

    def func(model: Model):
        yy = model.predict(X)
        return np.mean((y - yy) ** 2).item()

    return func


def validation_set_log_likelihood(
    X: Float[np.ndarray, "N D"],
    y: Float[np.ndarray, "N"],
) -> Objective:
    """Minimise the log likelihood of the model on the validation set."""

    def func(model: Model):
        yy = model.predict(X)
        std = model.predictive_uncertainty(X)
        # normal distribution likelihoods
        ls = (
            1
            / (std * np.sqrt(2 * np.pi))
            * np.exp(-((y - yy) ** 2) / (2 * std**2))
        )  # (N,)

        # take logs and sum
        return np.sum(np.log(ls)).item()

    return func


class Convertor:
    def __init__(self, params: dict[str, float | list[float]]):
        self.og_params = params

    def to_list(self, params: dict[str, float | list[float]]) -> list[float]:
        l = []
        for v in params.values():
            if isinstance(v, list):
                l.extend(v)
            else:
                l.append(v)
        return l

    def to_dict(self, params: list[float]) -> dict[str, float | list[float]]:
        d = {}
        left = 0
        for k, v in self.og_params.items():
            if isinstance(v, list):
                right = left + len(v)
                d[k] = params[left:right]
            else:
                right = left + 1
                d[k] = params[left]
            left = right
        return d


def optimise_model(
    m: Model,
    objective: Objective,
    X: Float[np.ndarray, "N D"],
    y: Float[np.ndarray, "N"],
    *,
    optimise_noise: bool = False,
    max_iterations: int = 100,
):
    """
    Optimise the model (kernel hyperparameters and noise)
    to minimise the objective function.

    Parameters
    ----------
    m
        the model to optimise.
    objective
        the objective function to minimise.
    X
        the training data.
    y
        the training targets.
    optimise_noise
        whether to optimise the noise.
    max_iterations
        the maximum number of iterations.
    """

    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError(
            "scipy is required to optimise the model. "
            "Please install it with `pip install scipy`."
        ) from None

    convertor = Convertor(m.kernel.params)

    # stuff parameters into a list in the order:
    # kernel params, sparse points, noise

    def params_to_model(params: list[float]) -> Model:
        noise = params.pop() if optimise_noise else m.noise

        param_dict = convertor.to_dict(params)
        new_kernel = m.kernel.with_new(param_dict)
        return m.with_new(new_kernel, noise)

    def _objective(params: list[float]):
        try:
            new_model = params_to_model(list(params))
            new_model.fit(X, y)
            return objective(new_model)
        except Exception:
            return 1e33

    starting_params = convertor.to_list(m.kernel.params)
    if optimise_noise:
        starting_params.append(m.noise)

    # TODO: cache models so no need to refit each time
    best_params = minimize(
        _objective,
        starting_params,
        options={"maxiter": max_iterations},
    ).x
    m = params_to_model(list(best_params))
    m.fit(X, y)
    return m
