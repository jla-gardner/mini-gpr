from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from make_a_gif import gif

from mini_gpr.kernels import RBF
from mini_gpr.models import GPR
from mini_gpr.opt import maximise_log_likelihood, optimise_model
from mini_gpr.tutorials import sample_toy_1d_system
from mini_gpr.viz import show_model_predictions

X, y = sample_toy_1d_system(n=100, seed=41)


def plot(n: int):
    n = max(1, n)
    n = min(n, 25)
    _X, _y = X[:n], y[:n]
    model = optimise_model(
        GPR(RBF(sigma=0.3), noise=0.1),
        maximise_log_likelihood,
        _X,
        _y,
        optimise_noise=True,
        max_iterations=1000,
    )
    show_model_predictions(model, _X, _y, test_points=np.linspace(-1, 11, 200))
    plt.text(-1, 4, f"{n} points", fontsize=12, ha="left", va="top")
    plt.ylim(-2, 4)


gif(
    range(-1, 28),
    plot,
    fps=1.5,
    save_to=Path(__file__).parent / "1d-gpr.gif",
    savefig_kwargs={"dpi": 200},
)
