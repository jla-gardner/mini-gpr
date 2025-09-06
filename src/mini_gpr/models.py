# ruff: noqa: F722, F821

from abc import ABC, abstractmethod

import numpy as np
from jaxtyping import Float

from mini_gpr.kernels import Kernel
from mini_gpr.solvers import LinearSolver, vanilla
from mini_gpr.utils import ensure_2d


class Model(ABC):
    def __init__(
        self,
        kernel: Kernel,
        noise: float = 1e-8,
        solver: LinearSolver = vanilla,
    ):
        self.kernel = kernel
        self.noise = noise
        self.solver = solver

    @abstractmethod
    def fit(self, X: Float[np.ndarray, "N D"], y: Float[np.ndarray, "N"]): ...

    @abstractmethod
    def predict(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]: ...

    @abstractmethod
    def latent_uncertainty(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]: ...

    def predictive_uncertainty(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]:
        return (self.latent_uncertainty(T) ** 2 + self.noise) ** 0.5

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

    def __repr__(self):
        return f"{self.__class__.__name__}(kernel={self.kernel}, noise={self.noise:.2e})"


class GPR(Model):
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
    def latent_uncertainty(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]:
        K_XT = self.kernel(self.X, T)  # (A, B)
        K_TT_diag = np.diag(self.kernel(T, T))  # (B,)
        v = self.solver(self.K_XX, K_XT)  # (A, B)
        var = K_TT_diag - np.einsum("ab,ab->b", K_XT, v)
        var = np.maximum(var, 0.0)  # Numerical stability
        return var**0.5

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


class SparseModel(Model):
    def __init__(
        self,
        kernel: Kernel,
        sparse_points: Float[np.ndarray, "M D"],
        noise: float = 1e-8,
        solver: LinearSolver = vanilla,
    ):
        super().__init__(kernel, noise, solver)
        self.M = sparse_points

    def with_new(self, kernel: Kernel, noise: float) -> "SparseModel":
        return self.__class__(
            kernel,
            sparse_points=self.M,
            noise=noise,
            solver=self.solver,
        )


class SoR(SparseModel):
    @ensure_2d("X")
    def fit(self, X: Float[np.ndarray, "A D"], y: Float[np.ndarray, "A"]):
        # compute kernel matrices
        K_MX = self.kernel(self.M, X)
        K_MM = self.kernel(self.M, self.M)

        # store necessary components for prediction
        self.y = y
        self.K_MX = K_MX
        self.inv_matrix = self.solver(
            K_MX @ K_MX.T + self.noise * K_MM,
            np.eye(len(self.M)),
        )
        self.K_MM = K_MM

    @ensure_2d("T")
    def predict(self, T: Float[np.ndarray, "T D"]) -> Float[np.ndarray, "T"]:
        K_TM = self.kernel(T, self.M)  # (T, M)
        temp = self.inv_matrix @ (self.K_MX @ self.y)  # (M,)

        return K_TM @ temp  # (T,)

    @ensure_2d("T")
    def latent_uncertainty(
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
        return var**0.5

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
