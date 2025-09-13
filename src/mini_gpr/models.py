# ruff: noqa: F722, F821

from abc import ABC, abstractmethod

import numpy as np
from jaxtyping import Float

from mini_gpr.kernels import Kernel
from mini_gpr.solvers import LinearSolver, vanilla
from mini_gpr.utils import ensure_1d, ensure_2d, get_rng


class Model(ABC):
    def __init__(
        self,
        kernel: Kernel,
        noise: float = 1e-8,
        solver: LinearSolver = vanilla,
    ):
        self.kernel = kernel
        self.noise = abs(noise)
        self.solver = solver

    @abstractmethod
    def fit(self, X: Float[np.ndarray, "N D"], y: Float[np.ndarray, "N"]):
        """
        Fit the model to the data.

        Parameters
        ----------
        X
            the data points.
        y
            the target values.
        """

    @abstractmethod
    def predict(self, T: Float[np.ndarray, "T D"]) -> Float[np.ndarray, "T"]:
        """
        Get the predictive mean of the function at the given locations.

        Parameters
        ----------
        T
            the data points/locations at which to make predictions
        """

    @abstractmethod
    def latent_uncertainty(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]:
        """
        Get the latent uncertainty of the function at the given locations.

        Parameters
        ----------
        T
            the data points/locations at which to get the latent uncertainty
        """

    def predictive_uncertainty(
        self, T: Float[np.ndarray, "T D"]
    ) -> Float[np.ndarray, "T"]:
        r"""
        Get the predictive uncertainty of the function at the given locations.

        Parameters
        ----------
        T
            the data points/locations at which to get the predictive uncertainty
        """
        return (self.latent_uncertainty(T) ** 2 + self.noise**2) ** 0.5

    @property
    @abstractmethod
    def log_likelihood(self) -> float:
        """
        Get the log marginal likelihood of the model conditioned on the
        training data.
        """

    @abstractmethod
    def with_new(self, kernel: Kernel, noise: float) -> "Model": ...

    @ensure_2d("locations")
    def sample_prior(
        self,
        locations: Float[np.ndarray, "N D"],
        n_samples: int = 1,
        *,
        rng: np.random.RandomState | int | None = None,
        jitter: float = 1e-8,
    ) -> Float[np.ndarray, "n N"]:
        """
        Generate samples from the model's prior distribution.

        Parameters
        ----------
        locations
            the data points/locations at which to generate samples
        n_samples
            the number of samples to generate
        rng
            the random number generator to use
        jitter
            a small value to add to the diagonal of the kernel matrix to ensure
            numerical stability
        """
        N = locations.shape[0]
        rng = get_rng(rng)
        K = self.kernel(locations, locations) + np.eye(N) * jitter
        L = np.linalg.cholesky(K)
        Z = rng.randn(N, n_samples)
        return L @ Z

    @abstractmethod
    def sample_posterior(
        self,
        locations: Float[np.ndarray, "N D"],
        n_samples: int = 1,
        *,
        rng: np.random.RandomState | int | None = None,
        jitter: float = 1e-6,
    ) -> Float[np.ndarray, "n N"]:
        """
        Generate samples from the model's posterior distribution.

        Parameters
        ----------
        locations
            the data points/locations at which to generate samples
        n_samples
            the number of samples to generate
        rng
            the random number generator to use
        jitter
            a small value to add to the diagonal of the kernel matrix to ensure
            numerical stability
        """

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(kernel={self.kernel}, noise={self.noise:.2e})"


class GPR(Model):
    """
    Full-rank Gaussian Process Regression model.

    Parameters
    ----------
    kernel: Kernel
        defines the covariance between data points.
    noise: float
        the aleatoric noise assumed to be present in the data.
    solver: LinearSolver
        solver of linear systems of the form `A @ x = y`.

    Example
    -------
    >>> from mini_gpr.kernels import RBF
    >>> from mini_gpr.models import GPR
    >>> model = GPR(kernel=RBF(), noise=1e-3)
    >>> model.fit(X, y)
    >>> predictions = model.predict(T) # (T,)
    >>> uncertainties = model.predictive_uncertainty(T) # (T,)
    """

    @ensure_1d("y")
    @ensure_2d("X")
    def fit(self, X: Float[np.ndarray, "N D"], y: Float[np.ndarray, "N"]):
        self.X = X
        self.K_XX = self.kernel(X, X) + self.noise**2 * np.eye(len(X))
        self.c = self.solver(self.K_XX, y).flatten()
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

        return (-0.5 * quad - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)).item()

    def with_new(self, kernel: Kernel, noise: float) -> "GPR":
        return GPR(kernel, noise, self.solver)

    @ensure_2d("locations")
    def sample_posterior(
        self,
        locations: Float[np.ndarray, "N D"],
        n_samples: int = 1,
        *,
        rng: np.random.RandomState | int | None = None,
        jitter: float = 1e-6,
    ) -> Float[np.ndarray, "n N"]:
        rng = get_rng(rng)

        N = locations.shape[0]

        mu = self.predict(locations)

        K_XT = self.kernel(self.X, locations)
        v = self.solver(self.K_XX, K_XT)
        K_TT = self.kernel(locations, locations)
        cov = K_TT - K_XT.T @ v

        # Force symmetry & numerical stability
        cov = 0.5 * (cov + cov.T)
        L = np.linalg.cholesky(cov + np.eye(N) * jitter)

        Z = rng.randn(N, n_samples)
        return (L @ Z) + mu[:, None]


class SoR(Model):
    """
    Subset of Regressors low-rank Gaussian Process Regression approximation.

    Parameters
    ----------
    kernel
        defines the covariance between data points.
    sparse_points
        the inducing points.
    noise
        the aleatoric noise assumed to be present in the data.
    solver
        solver of linear systems of the form `A @ x = y`.

    Example
    -------
    >>> from mini_gpr.kernels import RBF
    >>> from mini_gpr.models import SoR
    >>> model = SoR(
    ...    kernel=RBF(), sparse_points=np.random.rand(10, 2), noise=1e-3
    ... )
    >>> model.fit(X, y)
    >>> predictions = model.predict(T) # (T,)
    >>> uncertainties = model.latent_uncertainty(T) # (T,)
    """

    def __init__(
        self,
        kernel: Kernel,
        sparse_points: Float[np.ndarray, "M D"],
        noise: float = 1e-8,
        solver: LinearSolver = vanilla,
    ):
        super().__init__(kernel, noise, solver)
        self.M = sparse_points

    def with_new(self, kernel: Kernel, noise: float) -> "SoR":
        return self.__class__(
            kernel,
            sparse_points=self.M,
            noise=noise,
            solver=self.solver,
        )

    @ensure_2d("X")
    def fit(self, X: Float[np.ndarray, "A D"], y: Float[np.ndarray, "A"]):
        # compute kernel matrices
        K_MX = self.kernel(self.M, X)
        K_MM = self.kernel(self.M, self.M)

        # store necessary components for prediction
        self.y = y
        self.K_MX = K_MX
        self.inv_matrix = self.solver(
            K_MX @ K_MX.T + self.noise**2 * K_MM,
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
        var += self.noise**2 * np.diag(temp)

        # Ensure numerical stability
        var = np.maximum(var, 0.0)
        return var**0.5

    @property
    def log_likelihood(self) -> float:
        n = len(self.y)
        m = len(self.M)
        sigma2 = float(self.noise**2)

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
                "Kernel matrix is not positive definite. "
                "Try gradually increasing the noise."
            )

        logdet_Sigma = (n - m) * np.log(sigma2) - logdet_K + logdet_B

        return (-0.5 * (quad + logdet_Sigma + n * np.log(2 * np.pi))).item()

    @ensure_2d("locations")
    def sample_posterior(
        self,
        locations: Float[np.ndarray, "N D"],
        n_samples: int = 1,
        *,
        rng: np.random.RandomState | int | None = None,
        jitter: float = 1e-6,
    ) -> Float[np.ndarray, "N n"]:
        rng = get_rng(rng)

        # Posterior mean
        mu = self.predict(locations)  # (N,)

        # --- Compute posterior covariance ---
        K_TM = self.kernel(locations, self.M)  # (N, M)
        K_TT = self.kernel(locations, locations)  # (N, N)

        # First subtract Nyström term: K_TM K_MM^{-1} K_MT
        K_MM_inv_K_MT = self.solver(self.K_MM, K_TM.T)  # (M, N)
        cov = K_TT - K_TM @ K_MM_inv_K_MT  # (N, N)

        # Add correction term from the noise-adjusted projection
        # temp = K_TM @ inv_matrix @ K_TM.T
        cov += self.noise**2 * (K_TM @ self.inv_matrix @ K_TM.T)

        # Symmetrize for stability
        cov = 0.5 * (cov + cov.T)

        # --- Cholesky with jitter ---
        L = np.linalg.cholesky(cov + np.eye(cov.shape[0]) * jitter)

        # --- Draw samples ---
        N = cov.shape[0]
        Z = rng.randn(N, n_samples)  # (N, n_samples)
        return (L @ Z) + mu[:, None]
