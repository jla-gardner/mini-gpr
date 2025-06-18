import numpy as np
from jaxtyping import Float

from .full import GPR
from .kernels import RBF
from .sparse import SparseGPR
from .utils import UncertaintyModel

__version__ = "0.1.0"


def auto_fit_gpr(
    X: Float[np.ndarray, "N D"],
    y: Float[np.ndarray, "N"],
    lengthscale: float = 1.0,
    noise: float = 1e-2,
    sparsify: bool | int | None = None,
) -> UncertaintyModel:
    """
    Quickly create a GPR model for the given data.

    Parameters
    ----------
    X
        Data locations, cotaining A examples each of dimension D.
    y
        Data values, containing A scalar values.
    lengthscale
        The lengthscale over which you anticipate the data to vary.
    noise
        The standard deviation of the noise in the data.
    sparsify
        If ``True``, use sparse GPR with 10% of the data as inducing
        points.
        If an integer, use sparse GPR with the given number of inducing
        points.
        If ``None``, use full GPR.
    """
    kernel = RBF(sigma=lengthscale)
    if sparsify is None or sparsify is False:
        model = GPR(kernel, noise)

    else:
        n_sparse = sparsify if isinstance(sparsify, int) else len(X) // 10
        model = SparseGPR(kernel, noise, n_sparse)

    model.fit(X, y)
    return model
