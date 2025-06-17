# ruff: noqa: F722, F821

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from jaxtyping import Float

from .utils import ensure_2d


class Kernel(Protocol):
    def __call__(
        self,
        A: Float[np.ndarray, "A D"],
        B: Float[np.ndarray, "B D"],
    ) -> Float[np.ndarray, "A B"]: ...


class SumKernel(Kernel):
    def __init__(self, *kernels: Kernel):
        self.kernels = kernels

    @ensure_2d("A", "B")
    def __call__(
        self, A: Float[np.ndarray, "A D"], B: Float[np.ndarray, "B D"]
    ) -> Float[np.ndarray, "A B"]:
        return np.sum([kernel(A, B) for kernel in self.kernels], axis=0)


class ProductKernel(Kernel):
    def __init__(self, *kernels: Kernel):
        self.kernels = kernels

    @ensure_2d("A", "B")
    def __call__(
        self, A: Float[np.ndarray, "A D"], B: Float[np.ndarray, "B D"]
    ) -> Float[np.ndarray, "A B"]:
        return np.prod([kernel(A, B) for kernel in self.kernels], axis=0)


@ensure_2d("A", "B")
def cdist(
    A: Float[np.ndarray, "A D"], B: Float[np.ndarray, "B D"]
) -> Float[np.ndarray, "A B"]:
    residuals = A[:, None, :] - B[None, :, :]  # (A, B, D)
    return np.sqrt(np.sum(residuals**2, axis=2))  # (A, B)


@dataclass
class RBF(Kernel):
    sigma: float = 1.0

    @ensure_2d("A", "B")
    def __call__(
        self, A: Float[np.ndarray, "A D"], B: Float[np.ndarray, "B D"]
    ) -> Float[np.ndarray, "A B"]:
        return np.exp(-(cdist(A, B) ** 2) / (2 * self.sigma**2))


@ensure_2d("A", "B")
def DotProduct(
    A: Float[np.ndarray, "A D"], B: Float[np.ndarray, "B D"]
) -> Float[np.ndarray, "A B"]:
    return np.einsum("ad,bd->ab", A, B)
