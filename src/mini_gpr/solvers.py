# ruff: noqa: F722, F821

from typing import Protocol

import numpy as np
from jaxtyping import Float

from .utils import ensure_2d


class LinearSolver(Protocol):
    """
    Solve a linear system of the form `A @ x = y`.
    """

    def __call__(
        self,
        A: Float[np.ndarray, "A B"],
        y: Float[np.ndarray, "N"],
    ) -> Float[np.ndarray, "N"]: ...


@ensure_2d("A")
def vanilla(A, y):
    return np.linalg.solve(A, y)


@ensure_2d("A")
def least_squares(A, y):
    return np.linalg.lstsq(A, y, rcond=None)[0]


@ensure_2d("A")
def conjugate_gradient(A, y):
    EPS = 1e-9
    x = np.zeros_like(y)

    r = y - np.dot(A, x)
    p = r
    rsold = np.dot(np.transpose(r), r)

    for _ in range(len(y)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        if np.sqrt(rsnew) < EPS:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x
