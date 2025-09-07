# ruff: noqa: F722, F821

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from jaxtyping import Float

from mini_gpr.utils import ensure_2d


# TODO: diag method


class Kernel(ABC):
    def __init__(self, params: dict[str, float | list[float]]):
        self.params = params

    @abstractmethod
    def __call__(
        self,
        A: Float[np.ndarray, "N D"],
        B: Float[np.ndarray, "T D"],
    ) -> Float[np.ndarray, "A B"]: ...

    def with_new(self, params: dict[str, float | list[float]]) -> "Kernel":
        copy = deepcopy(self)
        copy.params = params
        return copy

    def __repr__(self):
        name = self.__class__.__name__
        params = []
        for k, v in self.params.items():
            if isinstance(v, list):
                vv = "[" + ", ".join(f"{x:.2e}" for x in v) + "]"
            else:
                vv = f"{v:.2e}"
            params.append(f"{k}={vv}")
        return f"{name}({', '.join(params)})"

    def __add__(self, other: "Kernel") -> "SumKernel":
        kernels: list[Kernel] = []
        for thing in [self, other]:
            if isinstance(thing, SumKernel):
                kernels.extend(thing.kernels)
            else:
                kernels.append(thing)
        return SumKernel(*kernels)

    def __mul__(self, other: "Kernel") -> "ProductKernel":
        kernels: list[Kernel] = []
        for thing in [self, other]:
            if isinstance(thing, ProductKernel):
                kernels.extend(thing.kernels)
            else:
                kernels.append(thing)
        return ProductKernel(*kernels)

    def __pow__(self, other: float) -> "PowerKernel":
        return PowerKernel(power=other, kernel=self)


class MultiKernel(Kernel):
    def __init__(self, *kernels: Kernel):
        self.kernels = list(kernels)
        params = {}
        for i, kernel in enumerate(self.kernels):
            updated_keys = {f"{i}-{k}": p for k, p in kernel.params.items()}
            params.update(updated_keys)
        super().__init__(params)

    def with_new(self, params: dict[str, float | list[float]]) -> "MultiKernel":
        new_kernels = []
        for i, kernel in enumerate(self.kernels):
            actual_params = {k: params[f"{i}-{k}"] for k in kernel.params}
            new_kernels.append(kernel.with_new(actual_params))

        return self.__class__(*new_kernels)

    def __repr__(self):
        name = self.__class__.__name__
        kernel_reps = [str(k) for k in self.kernels]
        return f"{name}({', '.join(kernel_reps)})"


class SumKernel(MultiKernel):
    @ensure_2d("A", "B")
    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        values = [kernel(A, B) for kernel in self.kernels]
        return np.sum(values, axis=0)


class ProductKernel(MultiKernel):
    @ensure_2d("A", "B")
    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        values = [kernel(A, B) for kernel in self.kernels]
        return np.prod(values, axis=0)


class PowerKernel(Kernel):
    def __init__(self, power: float, kernel: Kernel):
        super().__init__(kernel.params)
        self.kernel = kernel
        self.power = power

    @ensure_2d("A", "B")
    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return self.kernel(A, B) ** self.power

    def with_new(self, params) -> "PowerKernel":
        kernel = self.kernel.with_new(params)
        return PowerKernel(power=self.power, kernel=kernel)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(power={self.power:.2e}, kernel={repr(self.kernel)})"


class RBF(Kernel):
    def __init__(
        self,
        sigma: float | list[float] = 1.0,
        scale: float = 1.0,
    ):
        super().__init__(params={"sigma": sigma, "scale": scale})

    @ensure_2d("A", "B")
    def __call__(self, A, B):
        sigma, scale = self.params["sigma"], self.params["scale"]
        assert isinstance(scale, float | int)
        sigma = np.abs(sigma)

        norm_A = A / sigma
        norm_B = B / sigma
        k = (norm_A[:, None, :] - norm_B[None, :, :]) ** 2
        return np.exp(-k.sum(axis=2) / 2) * scale**2


class DotProduct(Kernel):
    def __init__(self, scale: float = 1.0):
        super().__init__(params={"scale": scale})

    @ensure_2d("A", "B")
    def __call__(self, A, B):
        scale = self.params["scale"]
        assert isinstance(scale, float | int)
        return np.einsum("ad,bd->ab", A, B) * scale**2


class Constant(Kernel):
    def __init__(self, value: float = 1.0):
        super().__init__(params={"value": value})

    def __call__(self, A, B):
        value = self.params["value"]
        assert isinstance(value, float | int)
        return np.ones((A.shape[0], B.shape[0])) * value**2


class Linear(Kernel):
    def __init__(self, m: float | list[float] = 0, scale: float = 1.0):
        super().__init__(params={"m": m, "scale": scale})

    @ensure_2d("A", "B")
    def __call__(self, A, B):
        m, scale = self.params["m"], self.params["scale"]
        assert isinstance(scale, float | int)

        return np.einsum("ad,bd->ab", A - m, B - m) * scale**2


class Periodic(Kernel):
    def __init__(
        self,
        sigma: float = 1.0,
        period: float | list[float] = 1.0,
        lengthscale: float | list[float] = 1.0,
    ):
        super().__init__(
            params={
                "sigma": sigma,
                "period": period,
                "lengthscale": lengthscale,
            }
        )

    @ensure_2d("A", "B")
    def __call__(self, A, B):
        sigma = self.params["sigma"]
        assert isinstance(sigma, float | int)
        period = self.params["period"]
        lengthscale = self.params["lengthscale"]

        # all shapes are (N, M, D)
        diff = A[:, None, :] - B[None, :, :]
        sin_terms = np.sin(np.pi * np.abs(diff) / period) ** 2
        exp_terms = -2 * sin_terms / np.power(lengthscale, 2)

        # shape is (N, M)
        exp_term = np.sum(exp_terms, axis=2)

        return (sigma**2) * np.exp(exp_term)
