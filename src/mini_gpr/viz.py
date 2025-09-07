from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float

from mini_gpr.kernels import Kernel
from mini_gpr.models import GPR, Model, SoR

try:
    from IPython.core.getipython import get_ipython

    _shell = get_ipython()
    if _shell is not None:
        _shell.run_line_magic("config", "InlineBackend.figure_format = 'svg'")
except ImportError:
    pass


def show_model_predictions(
    model: Model,
    X: Float[np.ndarray, "N D"],
    y: Float[np.ndarray, "N"],
    *,
    test_points: Float[np.ndarray, "T D"] | None = None,
    n_sigma: int = 3,
    keep_axes: bool = False,
    legend_loc: Literal["right", "top"] = "right",
    uncertainty_type: Literal["latent", "predictive"] = "predictive",
    marker_size: int | float | None = None,
):
    plt.figure(figsize=(4, 3))

    if test_points is None:
        lo, hi = np.min(X), np.max(X)
        w = hi - lo
        xx = np.linspace(lo - w * 0.15, hi + w * 0.15, 200)
    else:
        xx = test_points
    yy_mean = model.predict(xx)
    yy_std = (
        model.latent_uncertainty(xx)
        if uncertainty_type == "latent"
        else model.predictive_uncertainty(xx)
    )

    plt.plot(xx, yy_mean, c="crimson", label="Prediction", zorder=20, lw=2)
    for n in range(1, n_sigma + 1):
        plt.fill_between(
            xx,
            yy_mean - n * yy_std,
            yy_mean + n * yy_std,
            alpha=0.2,
            color="crimson",
            lw=0,
            label="Uncertainty" if n == 1 else None,
        )
    plt.plot(
        X, y, "ok", label="Data", zorder=10, lw=0, ms=marker_size, alpha=0.5
    )
    if isinstance(model, SoR):
        plt.plot(
            model.M,
            [0.0] * len(model.M),
            "^",
            color="gray",
            transform=plt.gca().get_xaxis_transform(),
            label="Sparse Points",
            clip_on=False,
        )
        plt.plot(
            model.M,
            [1.0] * len(model.M),
            "v",
            color="gray",
            transform=plt.gca().get_xaxis_transform(),
            clip_on=False,
        )
        for m in model.M:
            plt.axvline(m, color="gray", lw=1, ls="--")

    if legend_loc == "right":
        plt.legend(
            bbox_to_anchor=(1.05, 0.5),
            loc="center left",
        )
    else:
        plt.legend(
            ncol=3,
            bbox_to_anchor=(0.5, 1.05),
            loc="lower center",
        )
    if not keep_axes:
        plt.axis("off")


def sample_kernel(
    kernel: Kernel,
    x: np.ndarray | None = None,
    n_samples: int = 4,
    seed: int | None = None,
):
    if x is None:
        x = np.linspace(0, 6, 250)
    plt.figure(figsize=(3, 3))
    _model = GPR(kernel=kernel, noise=0.0)
    y = _model.sample_prior(x, n_samples, rng=np.random.RandomState(seed))
    plt.plot(x, y)
    for side in "top", "right":
        plt.gca().spines[side].set_visible(False)
    for side in "left", "bottom":
        plt.gca().spines[side].set_position(("outward", 10))
