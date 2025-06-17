# ruff: noqa: F722, F821

import numpy as np
from jaxtyping import Float

from .kernels import Kernel
from .solvers import LinearSolver, vanilla
from .utils import Model, ensure_2d


class GPR(Model):
    def __init__(
        self,
        kernel: Kernel,
        noise: float = 1e-8,
        solver: LinearSolver = vanilla,
    ):
        self.kernel = kernel
        self.noise = noise
        self.solver = solver

    @ensure_2d("X")
    def fit(self, X: Float[np.ndarray, "A D"], y: Float[np.ndarray, "A"]):
        self.X = X
        K = self.kernel(X, X) + self.noise * np.eye(len(X))
        self.c = self.solver(K, y)

    @ensure_2d("X")
    def predict(
        self, X: Float[np.ndarray, "B D"]
    ) -> Float[np.ndarray, "B"]:
        K = self.kernel(self.X, X)  # (A, B)
        return np.einsum("ab,a->b", K, self.c)  # (B)
