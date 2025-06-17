import numpy as np
from jaxtyping import Float

from .kernels import Kernel
from .selection import RandomSelector, Selector
from .solvers import LinearSolver, vanilla
from .utils import Model, ensure_2d


class SparseGPR(Model):
    def __init__(
        self,
        kernel: Kernel,
        noise: float = 1e-8,
        n_sparse: int = 10,
        strategy: Selector | None = None,
        solver: LinearSolver = vanilla,
    ):
        self.kernel = kernel
        self.noise = noise
        self.n_sparse = n_sparse
        self.strategy = strategy or RandomSelector(seed=42)
        self.solver = solver

    @ensure_2d("X")
    def fit(self, X: Float[np.ndarray, "A D"], y: Float[np.ndarray, "A"]):
        # choose m sparse points
        M = self.strategy(X, self.n_sparse)

        # compute kernel matrices
        K_MX = self.kernel(M, X)
        K_MM = self.kernel(M, M)
        Σ = np.eye(len(X)) / self.noise

        # compute full kernel matrix
        K = K_MM + K_MX @ Σ @ K_MX.T

        # compute full right-hand side
        y_prime = K_MX @ Σ @ y

        # store sparse points and coefficients
        self.points = M
        self.c = self.solver(K, y_prime)

    @ensure_2d("X")
    def predict(
        self, X: Float[np.ndarray, "B D"]
    ) -> Float[np.ndarray, "B"]:
        K = self.kernel(self.points, X)
        return np.einsum("mb,m->b", K, self.c)
