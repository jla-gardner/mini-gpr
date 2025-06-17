# ruff: noqa: F722, F821

from typing import Callable, Literal
import warnings
from abc import ABC, abstractmethod
from functools import wraps
from inspect import signature

import numpy as np
from jaxtyping import Float


Matrix_AD = Float[np.ndarray, "a d"]
Matrix_BD = Float[np.ndarray, "b d"]
Matrix_AB = Float[np.ndarray, "a b"]
Tensor_ABD = Float[np.ndarray, "a b d"]
Vector_A = Float[np.ndarray, "a"]  # a scalar


def ensure_2d(*var_names):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for name in var_names:
                X = bound.arguments[name]
                if len(X.shape) == 1:
                    bound.arguments[name] = X[:, None]

            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


@ensure_2d("A", "B")
def pairwise_residual(A: Matrix_AD, B: Matrix_BD) -> Tensor_ABD:
    """
    Computes the pairwise residual between two matrices.

    A and B are matrices of shape (a, d) and (b, d) respectively.
    The output is a matrix of shape (a, b, d) where the (i, j, k)th
    entry is A[i, k] - B[j, k].

    Taking the norm of the output along the last axis gives the
    pairwise distance between A and B.
    """

    return A[:, None, :] - B[None, :, :]


@ensure_2d("A", "B")
def pairwise_distance(A: Matrix_AD, B: Matrix_BD, ord: float = 2) -> Matrix_AB:
    """Computes the pairwise distance between two matrices."""

    return np.linalg.norm(pairwise_residual(A, B), axis=2, ord=ord)


class Kernel(ABC):
    """
    A base class for kernels.

    Kernels are callable objects that compute the kernel matrix between two
    matrices.
    """

    @abstractmethod
    def __call__(self, A: Matrix_AD, B: Matrix_BD) -> Matrix_AB:
        """
        Compute the kernel matrix between two matrices.
        """

    def __add__(self, other: "Kernel") -> "Kernel":
        """
        Create a new kernel object that acts on pairs of matrices to return
        the sum of the two wrapped kernels.
        """
        return SumKernel(self, other)

    def __mul__(self, other: "Kernel") -> "Kernel":
        """
        Create a new kernel object that acts on pairs of matrices to return
        the product of the two wrapped kernels.
        """
        return ProductKernel(self, other)


class SumKernel(Kernel):
    def __init__(self, *kernels: Kernel):
        self.kernels = kernels

    def __add__(self, other: Kernel) -> "SumKernel":
        return SumKernel(*self.kernels, other)

    @ensure_2d("A", "B")
    def __call__(self, A: Matrix_AD, B: Matrix_BD) -> Matrix_AB:
        return np.sum([k(A, B) for k in self.kernels], axis=0)


class ProductKernel(Kernel):
    def __init__(self, *kernels: Kernel):
        self.kernels = kernels

    def __mul__(self, other: Kernel) -> "ProductKernel":
        return ProductKernel(*self.kernels, other)

    @ensure_2d("A", "B")
    def __call__(self, A: Matrix_AD, B: Matrix_BD) -> Matrix_AB:
        return np.prod([k(A, B) for k in self.kernels], axis=0)


class RBF(Kernel):
    def __init__(self, sigma: float = 1):
        self.sigma = sigma

    @ensure_2d("A", "B")
    def __call__(self, A: Matrix_AD, B: Matrix_BD) -> Matrix_AB:
        return rbf(A, B, self.sigma)


class Linear(Kernel):
    @ensure_2d("A", "B")
    def __call__(self, A: Matrix_AD, B: Matrix_BD) -> Matrix_AB:
        """get the pairwise dot product between two matrices"""
        return A @ B.T


class ScalarMapping:
    def __init__(self, wrapped_kernel, mapping):
        self.wrapped_kernel = wrapped_kernel
        self.mapping = mapping

    @ensure_2d("A", "B")
    def __call__(self, A, B):
        return self.mapping(self.wrapped_kernel(A, B))


class Power(ScalarMapping):
    def __init__(self, wrapped_kernel, p=2):
        super().__init__(wrapped_kernel, lambda x: x**p)


def rbf(A: Matrix_AD, B: Matrix_BD, sigma: float = 1) -> Matrix_AB:
    """Computes the RBF kernel between two matrices."""

    distance = pairwise_distance(A, B, ord=2)
    return np.exp(-(distance**2) / (2 * sigma**2))


class ConstantMean:
    def __init__(self, mean: float = 0.0):
        self.mean = mean

    def fit(self, X: Matrix_AD, y: Vector_A):
        self.mean = np.mean(y)

    def __call__(self, X):
        return np.ones(len(X)) * self.mean


def solve(A: Matrix_AB, y: Vector_A) -> Vector_A:
    return np.linalg.solve(A, y)


def lstsq(A: Matrix_AB, y: Vector_A) -> Vector_A:
    return np.linalg.lstsq(A, y, rcond=None)[0]


def conj_grad(
    A: Matrix_AB,
    b: Vector_A,
    x: Vector_A | None = None,
    eps: float = 1e-9,
) -> Vector_A:
    """
    A function to solve the linear system `Ax = b` with the
    [conjugate gradient method](http://en.wikipedia.org/wiki/Conjugate_gradient_method).

    Parameters
    ----------
    A : matrix
        A real symmetric positive definite matrix.

    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution.
    """
    if x is None:
        _x = np.zeros(len(b))
    else:
        _x = x

    r = b - np.dot(A, _x)
    p = r
    rsold = np.dot(np.transpose(r), r)

    for _ in range(len(b)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        _x = _x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        if np.sqrt(rsnew) < eps:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return _x


SolverName = Literal["solve", "lstsq", "conj_grad"]
SolverFn = Callable[[Matrix_AB, Vector_A], Vector_A]
SOLVERS: dict[SolverName, SolverFn] = {
    "solve": solve,
    "lstsq": lstsq,
    "conj_grad": conj_grad,
}


class Solver:
    def __init__(self, solver: SolverName = "solve"):
        self.solver = SOLVERS[solver]

    def solve(self, A: Matrix_AB, y: Vector_A) -> Vector_A:
        return self.solver(A, y)


class FullGPR(Solver):
    def __init__(
        self,
        kernel: Kernel,
        noise: float = 1e-8,
        mean: ConstantMean | None = None,
        solver: SolverName = "solve",
    ):
        super().__init__(solver)
        self.kernel = kernel
        self.noise = noise
        if mean is None:
            mean = ConstantMean()
        self.mean = mean

    @ensure_2d("X")
    def fit(self, X: Matrix_AD, y: Vector_A):
        self.mean.fit(X, y)
        y = y - self.mean(X)

        self.X = X
        K = self.kernel(X, X) + self.noise * np.eye(len(X))
        self.c = self.solve(K, y)

    @ensure_2d("X")
    def predict(self, X: Matrix_AD) -> Vector_A:
        K = self.kernel(self.X, X)
        return self.mean(X) + K.T @ self.c


class SelectionStrategy(ABC):
    @abstractmethod
    def __call__(
        self, X: Matrix_AD, n_inducing: int
    ) -> Float[np.ndarray, "n_inducing d"]:
        """
        Select `n_inducing` points from `X` to use as inducing points.
        """


class RandomStrategy(SelectionStrategy):
    def __init__(self, seed=0):
        self.seed = seed

    def __call__(
        self, X: Matrix_AD, n_inducing: int
    ) -> Float[np.ndarray, "n_inducing d"]:
        rand = np.random.RandomState(self.seed)
        N = len(X)
        if n_inducing > N:
            warnings.warn(
                f"n_inducing ({n_inducing}) is greater than the number of "
                f"points ({N}). Using all points as inducing points."
            )
            return X

        return X[rand.choice(len(X), n_inducing, replace=False)]


class SparseGPR(Solver):
    def __init__(
        self,
        kernel,
        noise=1e-8,
        mean=None,
        n_inducing=10,
        strategy=None,
        solver: SolverName = "solve",
    ):
        super().__init__(solver)
        self.kernel = kernel
        self.noise = noise
        self.n_inducing = n_inducing
        self.mean = ConstantMean() if mean is None else mean
        self.strategy = RandomStrategy() if strategy is None else strategy

    @ensure_2d("X")
    def fit(self, X, y):
        self.mean.fit(X, y)
        y = y - self.mean(X)

        M = self.strategy(X, self.n_inducing)
        K_MX = self.kernel(M, X)
        K_MM = self.kernel(M, M)
        Σ = np.eye(len(X)) / self.noise

        K = K_MM + K_MX @ Σ @ K_MX.T
        y_prime = K_MX @ Σ @ y

        self.M = M
        self.c = self.solve(K, y_prime)

    @ensure_2d("X")
    def predict(self, X):
        K = self.kernel(self.M, X)
        return self.mean(X) + K.T @ self.c
