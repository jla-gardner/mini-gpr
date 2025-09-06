# ruff: noqa: F722, F821

from abc import ABC, abstractmethod

import numpy as np
from jaxtyping import Float

from mini_gpr.kernels import Kernel
from mini_gpr.selection import RandomSelector, Selector
from mini_gpr.solvers import LinearSolver, vanilla
from mini_gpr.utils import ensure_2d


class Model(ABC):
    def __init__(self, kernel: Kernel, noise: float):
        self.kernel = kernel
        self.noise = noise

    @abstractmethod
    def fit(self, X: Float[np.ndarray, "N D"], y: Float[np.ndarray, "N"]): ...

    @abstractmethod
    def predict(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]: ...

    @abstractmethod
    def uncertainty(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]: ...

    @property
    @abstractmethod
    def log_likelihood(self) -> float: ...

    @abstractmethod
    def with_new(self, kernel: Kernel, noise: float) -> "Model": ...

    @ensure_2d("locations")
    def sample_prior(
        self,
        locations: Float[np.ndarray, "N D"],
        n_samples: int = 1,
        *,
        rng: np.random.RandomState | None = None,
        jitter: float = 1e-8,
    ) -> Float[np.ndarray, "N n"]:
        N = locations.shape[0]
        if rng is None:
            rng = np.random.RandomState()
        K = self.kernel(locations, locations) + np.eye(N) * jitter
        L = np.linalg.cholesky(K)
        Z = rng.randn(N, n_samples)
        return (L @ Z).T


class GPR(Model):
    def __init__(
        self,
        kernel: Kernel,
        noise: float = 1e-8,
        solver: LinearSolver = vanilla,
    ):
        super().__init__(kernel, noise)
        self.solver = solver

    @ensure_2d("X")
    def fit(self, X: Float[np.ndarray, "N D"], y: Float[np.ndarray, "N"]):
        self.X = X
        self.K_XX = self.kernel(X, X) + self.noise * np.eye(len(X))
        self.c = self.solver(self.K_XX, y)
        self.y = y

    @ensure_2d("T")
    def predict(self, T: Float[np.ndarray, "T D"]) -> Float[np.ndarray, "T"]:
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

    @property
    def log_likelihood(self) -> float:
        n = len(self.y)

        # quadratic term: y^T (K+σ²I)^(-1) y
        quad = np.dot(self.y, self.c)

        # log determinant of covariance matrix
        sign, logdet = np.linalg.slogdet(self.K_XX)
        if sign <= 0:
            raise np.linalg.LinAlgError(
                "Kernel matrix is not positive definite. "
                "Try gradually increasing the noise."
            )

        return -0.5 * quad - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)

    def with_new(self, kernel: Kernel, noise: float) -> "GPR":
        return GPR(kernel, noise, self.solver)

    def __repr__(self):
        return f"GPR(kernel={self.kernel}, noise={self.noise})"


class SoR(Model):
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
    def predict(self, T: Float[np.ndarray, "T D"]) -> Float[np.ndarray, "T"]:
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

    @property
    def log_likelihood(self) -> float:
        """
        For SoR, we approximate K_XX ≈ Q := K_XM K_MM^{-1} K_MX and use
        Σ := Q + σ² I.

        Using Woodbury:
          Σ^{-1} = σ^{-2} I - K_XM (σ² K_MM + K_MX K_XM)^{-1} K_MX σ^{-2}

        Using the matrix determinant lemma:
          |Σ| = (σ²)^(n-m) * |σ² K_MM + K_MX K_XM| / |K_MM|
        """
        n = len(self.y)
        m = len(self.M)
        sigma2 = float(self.noise)

        if sigma2 <= 0.0:
            raise ValueError(
                "Noise variance must be positive for the likelihood."
            )

        # --- Quadratic term: y^T Σ^{-1} y ---
        # t = K_MX y
        t = self.K_MX @ self.y  # (m,)
        # u = (K_MX K_MX^T + σ² K_MM)^{-1} (K_MX y)
        u = self.inv_matrix @ t  # (m,)
        # y^T Σ^{-1} y = (1/σ²) [ y^T y - t^T u ]
        quad = (self.y @ self.y - t @ u) / sigma2

        # --- log|Σ| via determinant lemma ---
        # B = σ² K_MM + K_MX K_XM  (same matrix inverted in 'inv_matrix')
        B = self.K_MX @ self.K_MX.T + sigma2 * self.K_MM

        sign_B, logdet_B = np.linalg.slogdet(B)
        sign_K, logdet_K = np.linalg.slogdet(self.K_MM)

        if sign_B <= 0 or sign_K <= 0:
            raise np.linalg.LinAlgError(
                "Encountered non–PD matrix in SoR likelihood (check kernel/noise)."
            )

        logdet_Sigma = (n - m) * np.log(sigma2) - logdet_K + logdet_B

        return -0.5 * (quad + logdet_Sigma + n * np.log(2 * np.pi))

    def with_new(self, kernel: Kernel, noise: float) -> "SoR":
        return SoR(
            kernel,
            noise,
            n_sparse=self.n_sparse,
            strategy=self.strategy,
            solver=self.solver,
        )


# class FITC(UncertaintyModel):
#     def __init__(
#         self,
#         kernel: Kernel,
#         noise: float = 1e-8,
#         n_sparse: int = 10,
#         strategy: Selector | None = None,
#         solver: LinearSolver = vanilla,
#     ):
#         self.kernel = kernel
#         self.noise = noise
#         self.n_sparse = n_sparse
#         self.strategy = strategy or RandomSelector(seed=42)
#         self.solver = solver

#     @ensure_2d("X")
#     def fit(self, X: Float[np.ndarray, "A D"], y: Float[np.ndarray, "A"]):
#         # choose m sparse points
#         M = self.strategy(X, self.n_sparse)

#         # compute kernel matrices
#         K_MX = self.kernel(M, X)  # K_mn: M x N
#         K_MM = self.kernel(M, M)  # K_mm: M x M
#         K_XX_diag = np.diag(self.kernel(X, X))  # diagonal of K_nn: N

#         # compute Lambda matrix (diagonal)
#         # Lambda = diag[K_nn - K_nm K_mm^(-1) K_mn]
#         K_MM_inv_K_MX = self.solver(K_MM, K_MX)  # K_mm^(-1) K_mn
#         Lambda_diag = K_XX_diag - np.einsum(
#             "mn,mn->n", K_MX, K_MM_inv_K_MX
#         )

#         # ensure numerical stability
#         Lambda_diag = np.maximum(Lambda_diag, 1e-12)

#         # compute (Lambda + sigma_epsilon^2 I)^(-1)
#         Lambda_noise_inv_diag = 1.0 / (Lambda_diag + self.noise)

#         # compute the main inverted matrix for predictions
#         # (K_mm + K_mn(Lambda + sigma_epsilon^2 I)^(-1) K_nm)^(-1)
#         temp_matrix = (
#             K_MX * Lambda_noise_inv_diag[None, :] @ K_MX.T
#         )  # K_mn (Lambda + noise I)^(-1) K_nm
#         self.A_inv = self.solver(K_MM + temp_matrix, np.eye(len(M)))

#         # store necessary components for prediction
#         self.M = M
#         self.y = y
#         self.K_MX = K_MX
#         self.K_MM = K_MM
#         self.Lambda_noise_inv_diag = Lambda_noise_inv_diag
#         self.K_MM_inv = self.solver(K_MM, np.eye(len(M)))

#     @ensure_2d("T")
#     def predict(
#         self, T: Float[np.ndarray, "T D"]
#     ) -> Float[np.ndarray, "T"]:
#         # FITC predictive mean:
#         # f̄_*,FITC = K_tm (K_mm + K_mn(Λ + σ_ε² I)^(-1) K_nm)^(-1) K_mn (Λ + σ_ε² I)^(-1) y

#         K_TM = self.kernel(T, self.M)  # K_tm: T x M

#         # K_mn (Lambda + sigma_epsilon^2 I)^(-1) y
#         temp_y = self.K_MX @ (self.Lambda_noise_inv_diag * self.y)  # M

#         # final prediction
#         return K_TM @ (self.A_inv @ temp_y)  # T

#     @ensure_2d("T")
#     def uncertainty(
#         self, T: Float[np.ndarray, "T D"]
#     ) -> Float[np.ndarray, "T"]:
#         # FITC predictive covariance:
#         # cov(f_*,FITC) = K(T, T) - K_tm K_mm^(-1) K_mt + K_tm (K_mm + K_mn(Λ + σ_ε² I)^(-1) K_nm)^(-1) K_mt

#         K_TM = self.kernel(T, self.M)  # K_tm: T x M
#         K_TT_diag = np.diag(self.kernel(T, T))  # diagonal of K(T,T): T

#         var = K_TT_diag.copy()

#         # subtract K_tm K_mm^(-1) K_mt
#         temp1 = K_TM @ self.K_MM_inv @ K_TM.T  # T x T
#         var -= np.diag(temp1)

#         # add K_tm (K_mm + K_mn(Λ + σ_ε² I)^(-1) K_nm)^(-1) K_mt
#         temp2 = K_TM @ self.A_inv @ K_TM.T  # T x T
#         var += np.diag(temp2)

#         # ensure numerical stability
#         var = np.maximum(var, 0.0)
#         return var
