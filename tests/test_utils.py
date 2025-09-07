import numpy as np

from mini_gpr.utils import ensure_2d, get_rng


def test_ensure_2d():
    @ensure_2d("x", "y")
    def func(x, y, z):
        return x, y, z

    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    z = np.array([7, 8, 9])
    x, y, z = func(x, y, z)
    # x and y are now 2D, z remains 1D
    assert x.shape == (3, 1)
    assert y.shape == (3, 1)
    assert z.shape == (3,)  # remains unchanged

    x = np.array([[1, 2, 3]])
    y = np.array([[4, 5, 6]])
    z = np.array([[7, 8, 9]])
    x, y, z = func(x, y, z)
    # all are already 2D: no changes
    assert x.shape == (1, 3)
    assert y.shape == (1, 3)
    assert z.shape == (1, 3)


def test_get_rng():
    base_rng = np.random.RandomState(42)
    rng = get_rng(base_rng)
    # should be the same object
    assert rng is base_rng

    new_rng = get_rng(42)

    # should behave identically:
    assert rng.randn() == new_rng.randn()
