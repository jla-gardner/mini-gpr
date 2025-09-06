# ruff: noqa: F722, F821

from abc import ABC, abstractmethod

import numpy as np
from jaxtyping import Float

from mini_gpr.utils import ensure_2d


class Kernel(ABC):
    @abstractmethod
    def __call__(
        self,
        A: Float[np.ndarray, "N D"],
        B: Float[np.ndarray, "T D"],
    ) -> Float[np.ndarray, "A B"]: ...

    @property
    @abstractmethod
    def params(self) -> dict[str, Float[np.ndarray, "P"]]: ...

    @abstractmethod
    def with_new(
        self, params: dict[str, Float[np.ndarray, "P"]]
    ) -> "Kernel": ...

    def __repr__(self):
        name = self.__class__.__name__
        params = [f"{k}={v}" for k, v in self.params.items()]
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


class RBF(Kernel):
    def __init__(
        self,
        sigma: float | Float[np.ndarray, "D"] = 1.0,
        scale: float = 1.0,
    ):
        self.sigma = np.array(sigma)
        self.scale = scale

    @ensure_2d("A", "B")
    def __call__(self, A, B):
        norm_A = A / self.sigma
        norm_B = B / self.sigma
        k = (norm_A[:, None, :] - norm_B[None, :, :]) ** 2
        return np.exp(-k.sum(axis=2) / 2) * self.scale

    @property
    def params(self):
        return {"scale": np.array(self.scale), "sigma": np.array(self.sigma)}

    def with_new(self, params) -> "RBF":
        return RBF(sigma=params["sigma"], scale=params["scale"].item())


class DotProduct(Kernel):
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, A, B):
        return np.einsum("ad,bd->ab", A, B) * self.scale

    @property
    def params(self):
        return {"scale": np.array(self.scale)}

    def with_new(self, params) -> "DotProduct":
        return DotProduct(scale=params["scale"].item())


class Constant(Kernel):
    def __init__(self, value: float = 1.0):
        self.value = value

    def __call__(self, A, B):
        return np.ones((A.shape[0], B.shape[0])) * self.value

    @property
    def params(self):
        return {"value": np.array(self.value)}

    def with_new(self, params) -> "Constant":
        return Constant(value=params["value"].item())


class Linear(Kernel):
    def __init__(self, m: float | Float[np.ndarray, "D"], scale: float = 1.0):
        self.m = np.array(m)
        self.scale = scale

    def __call__(self, A, B):
        return np.einsum("ad,bd->ab", A - self.m, B - self.m) * self.scale

    @property
    def params(self):
        return {"m": np.array(self.m), "scale": np.array(self.scale)}

    def with_new(self, params) -> "Linear":
        return Linear(m=params["m"], scale=params["scale"].item())


class MultiKernel(Kernel):
    def __init__(self, *kernels: Kernel):
        self.kernels = kernels

    @property
    def params(self) -> dict[str, Float[np.ndarray, "P"]]:
        params: dict[str, Float[np.ndarray, "P"]] = {}
        for i, kernel in enumerate(self.kernels):
            updated_keys = {f"{i}-{k}": p for k, p in kernel.params.items()}
            params.update(updated_keys)
        return params

    def with_new(
        self, params: dict[str, Float[np.ndarray, "P"]]
    ) -> "MultiKernel":
        kernels = []
        for i, kernel in enumerate(self.kernels):
            old_keys = {f"{i}-{k}" for k in kernel.params}
            new_kernel_params = {
                k.removeprefix(f"{i}-"): v
                for k, v in params.items()
                if k in old_keys
            }
            kernels.append(kernel.with_new(new_kernel_params))
        return self.__class__(*kernels)

    def __repr__(self):
        name = self.__class__.__name__
        kernel_reps = [str(k) for k in self.kernels]
        return f"{name}({', '.join(kernel_reps)})"


class SumKernel(MultiKernel):
    @ensure_2d("A", "B")
    def __call__(
        self,
        A: Float[np.ndarray, "N D"],
        B: Float[np.ndarray, "T D"],
    ) -> Float[np.ndarray, "A B"]:
        return np.sum(
            [kernel(A, B) for kernel in self.kernels],
            axis=0,
        )


class ProductKernel(MultiKernel):
    @ensure_2d("A", "B")
    def __call__(
        self,
        A: Float[np.ndarray, "N D"],
        B: Float[np.ndarray, "T D"],
    ) -> Float[np.ndarray, "A B"]:
        return np.prod(
            [kernel(A, B) for kernel in self.kernels],
            axis=0,
        )


class PowerKernel(Kernel):
    def __init__(self, power: float, kernel: Kernel):
        self.power = power
        self.kernel = kernel

    @ensure_2d("A", "B")
    def __call__(
        self,
        A: Float[np.ndarray, "N D"],
        B: Float[np.ndarray, "T D"],
    ) -> Float[np.ndarray, "A B"]:
        return self.kernel(A, B) ** self.power

    @property
    def params(self) -> dict[str, Float[np.ndarray, "P"]]:
        params = {f"wrapped-{k}": v for k, v in self.kernel.params.items()}
        params["power"] = np.array(self.power)
        return params

    def with_new(self, params) -> "PowerKernel":
        power = params.pop("power").item()
        kernel = self.kernel.with_new(
            {k.removeprefix("wrapped-"): v for k, v in params.items()}
        )
        return PowerKernel(power=power, kernel=kernel)
