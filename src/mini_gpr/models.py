# ruff: noqa: F722, F821

import numpy as np
from jaxtyping import Float

from .kernels import Kernel
from .selection import RandomSelector, Selector
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


class SoR(UncertaintyModel):
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

        # store necessary components for prediction
        self.M = M
        self.y = y

        # cache
        self.K_MX = K_MX
        self.inv_matrix = self.solver(
            K_MX @ K_MX.T + self.noise * K_MM,
            np.eye(len(M)),
        )
        self.K_MM = K_MM

    @ensure_2d("T")
    def predict(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]:
        K_TM = self.kernel(T, self.M)  # (T, M)
        temp = self.inv_matrix @ (self.K_MX @ self.y)  # (M,)

        return K_TM @ temp  # (T,)

    @ensure_2d("T")
    def uncertainty(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]:
        # Compute required kernel matrices
        K_TM = self.kernel(T, self.M)
        K_TT_diag = np.diag(self.kernel(T, T))

        var = K_TT_diag.copy()

        K_MM_inv_K_MT = self.solver(self.K_MM, K_TM.T)
        var -= np.einsum("tm,mt->t", K_TM, K_MM_inv_K_MT)

        temp = K_TM @ self.inv_matrix @ K_TM.T
        var += self.noise * np.diag(temp)

        # Ensure numerical stability
        var = np.maximum(var, 0.0)
        return var


class FITC(UncertaintyModel):
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
        K_MX = self.kernel(M, X)  # K_mn: M x N
        K_MM = self.kernel(M, M)  # K_mm: M x M
        K_XX_diag = np.diag(self.kernel(X, X))  # diagonal of K_nn: N

        # compute Lambda matrix (diagonal)
        # Lambda = diag[K_nn - K_nm K_mm^(-1) K_mn]
        K_MM_inv_K_MX = self.solver(K_MM, K_MX)  # K_mm^(-1) K_mn
        Lambda_diag = K_XX_diag - np.einsum(
            "mn,mn->n", K_MX, K_MM_inv_K_MX
        )

        # ensure numerical stability
        Lambda_diag = np.maximum(Lambda_diag, 1e-12)

        # compute (Lambda + sigma_epsilon^2 I)^(-1)
        Lambda_noise_inv_diag = 1.0 / (Lambda_diag + self.noise)

        # compute the main inverted matrix for predictions
        # (K_mm + K_mn(Lambda + sigma_epsilon^2 I)^(-1) K_nm)^(-1)
        temp_matrix = (
            K_MX * Lambda_noise_inv_diag[None, :] @ K_MX.T
        )  # K_mn (Lambda + noise I)^(-1) K_nm
        self.A_inv = self.solver(K_MM + temp_matrix, np.eye(len(M)))

        # store necessary components for prediction
        self.M = M
        self.y = y
        self.K_MX = K_MX
        self.K_MM = K_MM
        self.Lambda_noise_inv_diag = Lambda_noise_inv_diag
        self.K_MM_inv = self.solver(K_MM, np.eye(len(M)))

    @ensure_2d("T")
    def predict(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]:
        # FITC predictive mean:
        # f̄_*,FITC = K_tm (K_mm + K_mn(Λ + σ_ε² I)^(-1) K_nm)^(-1) K_mn (Λ + σ_ε² I)^(-1) y

        K_TM = self.kernel(T, self.M)  # K_tm: T x M

        # K_mn (Lambda + sigma_epsilon^2 I)^(-1) y
        temp_y = self.K_MX @ (self.Lambda_noise_inv_diag * self.y)  # M

        # final prediction
        return K_TM @ (self.A_inv @ temp_y)  # T

    @ensure_2d("T")
    def uncertainty(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]:
        # FITC predictive covariance:
        # cov(f_*,FITC) = K(T, T) - K_tm K_mm^(-1) K_mt + K_tm (K_mm + K_mn(Λ + σ_ε² I)^(-1) K_nm)^(-1) K_mt

        K_TM = self.kernel(T, self.M)  # K_tm: T x M
        K_TT_diag = np.diag(self.kernel(T, T))  # diagonal of K(T,T): T

        var = K_TT_diag.copy()

        # subtract K_tm K_mm^(-1) K_mt
        temp1 = K_TM @ self.K_MM_inv @ K_TM.T  # T x T
        var -= np.diag(temp1)

        # add K_tm (K_mm + K_mn(Λ + σ_ε² I)^(-1) K_nm)^(-1) K_mt
        temp2 = K_TM @ self.A_inv @ K_TM.T  # T x T
        var += np.diag(temp2)

        # ensure numerical stability
        var = np.maximum(var, 0.0)
        return var
