from typing import Protocol

import numpy as np
from jaxtyping import Float

from mini_gpr.models import Model


class Objective(Protocol):
    def __call__(self, model: Model) -> float: ...


def maximise_log_likelihood(model: Model):
    return -model.log_likelihood


class Convertor:
    def __init__(self, params: dict[str, np.ndarray]):
        self.og_params = params

    def to_list(self, params: dict[str, np.ndarray]) -> list[float]:
        l = []
        for v in params.values():
            l.extend(v.reshape(-1).tolist())
        return l

    def to_dict(self, params: list[float]) -> dict[str, np.ndarray]:
        d = {}
        left = 0
        for k, v in self.og_params.items():
            right = left + len(v.reshape(-1))
            d[k] = np.array(params[left:right])
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
    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError(
            "scipy is required to optimise the model. "
            "Please install it with `pip install scipy`."
        ) from None

    convertor = Convertor(m.kernel.params)

    def params_to_model(params: list[float]) -> Model:
        noise = params.pop() if optimise_noise else m.noise
        param_dict = convertor.to_dict(params)
        new_kernel = m.kernel.with_new(param_dict)
        return m.with_new(new_kernel, noise)

    def _objective(params: list[float]):
        try:
            new_model = params_to_model(params)
            new_model.fit(X, y)
            return objective(new_model)
        except Exception:
            return 1e10

    starting_params = convertor.to_list(m.kernel.params)
    if optimise_noise:
        starting_params.append(m.noise)

    # TODO: cache models so no need to refit each time
    best_params = minimize(
        _objective,
        starting_params,
        options={"maxiter": max_iterations},
    ).x
    m = params_to_model(best_params)
    m.fit(X, y)
    return m
