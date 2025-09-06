from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float

from mini_gpr.models import Model

try:
    from IPython.core.getipython import get_ipython

    _shell = get_ipython()
    if _shell is not None:
        _shell.run_line_magic(
            "config", "InlineBackend.figure_format = 'svg'"
        )
except ImportError:
    pass


def show_model_predictions(
    model: Model,
    X: Float[np.ndarray, "N D"],
    y: Float[np.ndarray, "N"],
    *,
    n_sigma: int = 3,
    keep_axes: bool = False,
    legend_loc: Literal["right", "top"] = "right",
):
    lo, hi = np.min(X), np.max(X)
    w = hi - lo
    xx = np.linspace(lo - w * 0.1, hi + w * 0.1, 100)
    yy_mean = model.predict(xx)
    yy_std = model.uncertainty(xx)

    plt.plot(X, y, "ok", label="Data", zorder=20)
    plt.plot(xx, yy_mean, c="crimson", label="Prediction", zorder=10)
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
