# ruff: noqa: F722, F821

import numpy as np
from jaxtyping import Float

from .kernels import Kernel
from .solvers import LinearSolver, vanilla
from .utils import UncertaintyModel, ensure_2d


class GPR(UncertaintyModel):
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
    def fit(self, X: Float[np.ndarray, "N D"], y: Float[np.ndarray, "N"]):
        self.X = X
        self.K_XX = self.kernel(X, X) + self.noise * np.eye(len(X))
        self.c = self.solver(self.K_XX, y)

    @ensure_2d("T")
    def predict(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]:
        K_XT = self.kernel(self.X, T)  # (A, B)
        return np.einsum("ab,a->b", K_XT, self.c)  # (B)

    @ensure_2d("T")
    def uncertainty(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]:
        K_XT = self.kernel(self.X, T)  # (A, B)
        K_TT_diag = np.diag(self.kernel(T, T))  # (B,)
        v = self.solver(self.K_XX, K_XT)  # (A, B)
        var = K_TT_diag - np.einsum("ab,ab->b", K_XT, v)
        var = np.maximum(var, 0.0)  # Numerical stability
        return var
