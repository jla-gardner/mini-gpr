# ruff: noqa: F722, F821

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from jaxtyping import Float

from mini_gpr.utils import ensure_2d

# TODO: diag method


class Kernel(ABC):
    """
    Base class for all kernels.

    Parameters
    ----------
    params
        the hyper-parameters of the kernel.
    """

    def __init__(self, params: dict[str, float | list[float]]):
        self.params = params

    @abstractmethod
    def __call__(
        self,
        A: Float[np.ndarray, "N D"],
        B: Float[np.ndarray, "T D"],
    ) -> Float[np.ndarray, "A B"]:
        """
        Compute the kernel matrix between two sets of points.

        Note that all implementations should access the hyper-parameters via
        the `params` attribute.

        Parameters
        ----------
        A
            the first set of points.
        B
            the second set of points.
        """

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
        """
        Add two kernels together.

        Example
        -------
        >>> k1 = RBF(sigma=1.0)
        >>> k2 = Linear()
        >>> k = k1 + k2
        >>> assert k(A, B) == k1(A, B) + k2(A, B)
        """
        kernels: list[Kernel] = []
        for thing in [self, other]:
            if isinstance(thing, SumKernel):
                kernels.extend(thing.kernels)
            else:
                kernels.append(thing)
        return SumKernel(*kernels)

    def __mul__(self, other: "Kernel") -> "ProductKernel":
        """
        Multiply two kernels together.

        Example
        -------
        >>> k1 = RBF(sigma=1.0)
        >>> k2 = Linear()
        >>> k = k1 * k2
        >>> assert k(A, B) == k1(A, B) * k2(A, B)
        """
        kernels: list[Kernel] = []
        for thing in [self, other]:
            if isinstance(thing, ProductKernel):
                kernels.extend(thing.kernels)
            else:
                kernels.append(thing)
        return ProductKernel(*kernels)

    def __pow__(self, other: float) -> "PowerKernel":
        """
        Raise this kernel to some power.

        Example
        -------
        >>> k = RBF(sigma=1.0)
        >>> k2 = k**2.0
        >>> assert k2(A, B) == k(A, B)**2.0
        """
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
    """Sum over multiple kernels."""

    @ensure_2d("A", "B")
    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        values = [kernel(A, B) for kernel in self.kernels]
        return np.sum(values, axis=0)


class ProductKernel(MultiKernel):
    """Product over multiple kernels."""

    @ensure_2d("A", "B")
    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        values = [kernel(A, B) for kernel in self.kernels]
        return np.prod(values, axis=0)


class PowerKernel(Kernel):
    """Raise a kernel to some power."""

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
    r"""
    Radial Basis Function kernel.

    .. math::

        k(x, x^\prime) = \exp\left(-\frac{1}{2\sigma^2}
        (x - x^\prime)^2\right) \cdot s^2

    where:

    - :math:`\sigma` is the ``sigma`` parameter.
    - :math:`s` is the ``scale`` parameter.

    Example
    -------
    >>> k = RBF(sigma=1.0, scale=1.0)
    >>> K = k(A, B)
    >>> assert K.shape == (A.shape[0], B.shape[0])
    """

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
    r"""
    Dot product kernel.

    .. math::

        k(x, x^\prime) = x \cdot x^\prime \cdot s^2

    where:

    - :math:`s` is the ``scale`` parameter.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__(params={"scale": scale})

    @ensure_2d("A", "B")
    def __call__(self, A, B):
        scale = self.params["scale"]
        assert isinstance(scale, float | int)
        return np.einsum("ad,bd->ab", A, B) * scale**2


class Constant(Kernel):
    r"""
    Constant kernel.

    .. math::

        k(x, x^\prime) = s^2

    where:

    - :math:`s` is the ``value`` parameter.
    """

    def __init__(self, value: float = 1.0):
        super().__init__(params={"value": value})

    def __call__(self, A, B):
        value = self.params["value"]
        assert isinstance(value, float | int)
        return np.ones((A.shape[0], B.shape[0])) * value**2


class Linear(Kernel):
    r"""
    Linear kernel.

    .. math::

        k(x, x^\prime) = (x - m) \cdot (x^\prime - m) \cdot s^2

    where:

    - :math:`m` is the ``m`` parameter.
    - :math:`s` is the ``scale`` parameter.
    """

    def __init__(self, m: float | list[float] = 0, scale: float = 1.0):
        super().__init__(params={"m": m, "scale": scale})

    @ensure_2d("A", "B")
    def __call__(self, A, B):
        m, scale = self.params["m"], self.params["scale"]
        assert isinstance(scale, float | int)

        return np.einsum("ad,bd->ab", A - m, B - m) * scale**2


class Periodic(Kernel):
    r"""
    Periodic kernel.

    .. math::

        k(x, x^\prime) = \exp\left(-\frac{1}{2\sigma^2}
        (x - x^\prime)^2\right) \cdot s^2

    where:

    - :math:`s` is the ``scale`` parameter.
    - :math:`p` is the ``period`` parameter.
    - :math:`\sigma` is the ``sigma`` parameter.
    """

    def __init__(
        self,
        scale: float = 1.0,
        period: float | list[float] = 1.0,
        sigma: float | list[float] = 1.0,
    ):
        super().__init__(
            params={"sigma": sigma, "period": period, "scale": scale}
        )

    @ensure_2d("A", "B")
    def __call__(self, A, B):
        sigma = self.params["sigma"]
        period = self.params["period"]
        scale = self.params["scale"]
        assert isinstance(scale, float | int)

        # all shapes are (N, M, D)
        diff = A[:, None, :] - B[None, :, :]
        sin_terms = np.sin(np.pi * np.abs(diff) / period) ** 2
        exp_terms = -2 * sin_terms / np.power(sigma, 2)

        # shape is (N, M)
        exp_term = np.sum(exp_terms, axis=2)

        return (scale**2) * np.exp(exp_term)
