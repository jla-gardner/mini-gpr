import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (3, 2)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

try:
    from IPython.core.getipython import get_ipython

    _shell = get_ipython()
    if _shell is not None:
        _shell.run_line_magic("config", "InlineBackend.figure_format = 'svg'")
except ImportError:
    pass


def sample_toy_1d_system(n=100, seed=42):
    def true_function(x):
        return np.sin(x) + (x / 7) ** 2

    rng = np.random.RandomState(seed)
    x = rng.rand(n) * 10
    y = true_function(x) + rng.randn(n) * 0.2
    return x, y


def get_grid(xlim, ylim, N=100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(xlim[0], xlim[1], N)
    y = np.linspace(ylim[0], ylim[1], N)
    mesh_x, mesh_y = np.meshgrid(x, y)
    return x, y, np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)
