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
    def fit(self, X: Float[np.ndarray, "N D"], y: Float[np.ndarray, "N"]):
        # choose m sparse points
        M = self.strategy(X, self.n_sparse)

        # compute kernel matrices
        K_MX = self.kernel(M, X)  # (M, N)
        K_MM = self.kernel(M, M)  # (M, M)
        diag_K_XX = np.diag(self.kernel(X, X))  # (N,)

        # compute low-rank approximation diagonal
        K_MM_inv = np.linalg.inv(K_MM)
        Q_X_diag = np.einsum("mn,mk,nk->n", K_MX, K_MM_inv, K_MX)

        # diagonal correction term (Λ)
        Λ = diag_K_XX - Q_X_diag + self.noise  # (N,)

        # build approximate covariance in inducing space
        A = K_MX @ (K_MX / Λ).T + K_MM  # (M, M)

        # compute right-hand side
        y_prime = K_MX @ (y / Λ)

        # store for prediction
        self.points = M
        self.K_MM_inv = K_MM_inv
        self.A = A
        self.c = self.solver(A, y_prime)
        self.Λ = Λ

    @ensure_2d("T")
    def predict(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]:
        K_MT = self.kernel(self.points, T)  # (M, T)
        mean = np.einsum("mt,m->t", K_MT, self.c)
        return mean

    @ensure_2d("T")
    def uncertainty(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]:
        K_MT = self.kernel(self.points, T)  # (M, T)
        K_TT_diag = np.diag(self.kernel(T, T))  # (T,)

        v = self.solver(self.A, K_MT)  # (M, T)
        Q_T_diag = np.einsum("mt,mt->t", K_MT, v)

        var = K_TT_diag - Q_T_diag + self.noise
        var = np.maximum(var, 0.0)
        return var
