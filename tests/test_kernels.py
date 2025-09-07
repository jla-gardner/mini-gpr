import numpy as np
import pytest
from numpy.testing import assert_allclose

from mini_gpr.kernels import (
    RBF,
    Constant,
    DotProduct,
    Kernel,
    Linear,
    Periodic,
    PowerKernel,
    ProductKernel,
    SumKernel,
)

kernels_to_test = [
    RBF(),
    DotProduct(),
    Constant(),
    Linear(),
    Periodic(),
    RBF() + Constant(),
    Linear() * Periodic(),
    DotProduct() ** 2,
]


@pytest.mark.parametrize(
    "k",
    kernels_to_test,
    ids=lambda x: x.__class__.__name__,
)
def test_kernel(k):
    A = np.random.randn(10, 2)
    B = np.random.randn(5, 2)

    K = k(A, B)
    assert K.shape == (10, 5)
    assert np.all(np.isfinite(K))

    params = k.params
    assert isinstance(params, dict)
    assert all(isinstance(v, float | int | list) for v in params.values())

    new_k = k.with_new(params)
    assert new_k.params == params
    assert new_k != k
    new_K = new_k(A, B)
    assert_allclose(K, new_K)


def test_kernel_abstract():
    """Test that Kernel is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        Kernel({"param": 1.0})  # type: ignore


def test_kernel_repr():
    """Test kernel string representation."""
    rbf = RBF(sigma=1.0, scale=2.0)
    expected = "RBF(sigma=1.00e+00, scale=2.00e+00)"
    assert repr(rbf) == expected


def test_kernel_repr_with_list_params():
    """Test kernel repr with list parameters."""
    rbf = RBF(sigma=[1.0, 2.0], scale=1.0)
    expected = "RBF(sigma=[1.00e+00, 2.00e+00], scale=1.00e+00)"
    assert repr(rbf) == expected


def test_with_new():
    """Test creating new kernel with different parameters."""
    rbf = RBF(sigma=1.0, scale=2.0)
    new_rbf = rbf.with_new({"sigma": 3.0, "scale": 4.0})

    assert new_rbf.params["sigma"] == 3.0
    assert new_rbf.params["scale"] == 4.0
    assert rbf.params["sigma"] == 1.0  # Original unchanged
    assert rbf.params["scale"] == 2.0


def test_kernel_addition():
    """Test kernel addition creates SumKernel."""
    k1 = RBF(sigma=1.0)
    k2 = Linear()
    sum_kernel = k1 + k2

    assert isinstance(sum_kernel, SumKernel)
    assert len(sum_kernel.kernels) == 2
    assert len(sum_kernel.params) == sum(len(k.params) for k in (k1, k2))

    A = np.random.randn(10, 2)
    B = np.random.randn(5, 2)
    K1 = k1(A, B)
    K2 = k2(A, B)
    K = sum_kernel(A, B)
    assert_allclose(K, K1 + K2)
    assert K.shape == (10, 5)
    assert np.all(np.isfinite(K))


def test_kernel_multiplication():
    """Test kernel multiplication creates ProductKernel."""
    k1 = RBF(sigma=1.0)
    k2 = Linear()
    prod_kernel = k1 * k2

    assert isinstance(prod_kernel, ProductKernel)
    assert len(prod_kernel.kernels) == 2
    assert len(prod_kernel.params) == sum(len(k.params) for k in (k1, k2))

    A = np.random.randn(10, 2)
    B = np.random.randn(5, 2)
    K1 = k1(A, B)
    K2 = k2(A, B)
    K = prod_kernel(A, B)
    assert_allclose(K, K1 * K2)
    assert K.shape == (10, 5)
    assert np.all(np.isfinite(K))


def test_kernel_power():
    """Test kernel power creates PowerKernel."""
    k = RBF(sigma=1.0)
    power_kernel = k**2

    assert isinstance(power_kernel, PowerKernel)
    assert power_kernel.power == 2
    assert power_kernel.kernel is k


def test_rbf_basic():
    """Test basic RBF kernel functionality."""
    rbf = RBF(sigma=1.0, scale=1.0)
    X = np.array([[0.0], [1.0]])
    Y = np.array([[0.0], [1.0]])

    K = rbf(X, Y)
    expected = np.array([[1.0, np.exp(-0.5)], [np.exp(-0.5), 1.0]])
    assert_allclose(K, expected)


def test_rbf_1d_input():
    """Test RBF with 1D input (should be converted to 2D)."""
    rbf = RBF(sigma=1.0, scale=1.0)
    X = np.array([0.0, 1.0])
    Y = np.array([0.0, 1.0])

    K = rbf(X, Y)
    expected = np.array([[1.0, np.exp(-0.5)], [np.exp(-0.5), 1.0]])
    assert_allclose(K, expected)


def test_rbf_scale():
    """Test RBF kernel with different scale."""
    rbf = RBF(sigma=1.0, scale=2.0)
    X = np.array([[0.0], [1.0]])
    Y = np.array([[0.0], [1.0]])

    K = rbf(X, Y)
    expected = np.array([[4.0, 4.0 * np.exp(-0.5)], [4.0 * np.exp(-0.5), 4.0]])
    assert_allclose(K, expected)


def test_rbf_sigma_list():
    """Test RBF kernel with list of sigma values."""
    rbf = RBF(sigma=[1.0, 2.0], scale=1.0)
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    Y = np.array([[0.0, 0.0], [1.0, 1.0]])

    K = rbf(X, Y)
    # Should be 1.0 for identical points
    assert_allclose(np.diag(K), 1.0)


def test_rbf_symmetry():
    """Test RBF kernel is symmetric."""
    rbf = RBF(sigma=1.0, scale=1.0)
    X = np.random.randn(5, 2)
    Y = np.random.randn(5, 2)

    K_XY = rbf(X, Y)
    K_YX = rbf(Y, X)
    assert_allclose(K_XY, K_YX.T)


def test_rbf_positive_definite():
    """Test RBF kernel produces positive definite matrices."""
    rbf = RBF(sigma=1.0, scale=1.0)
    X = np.random.randn(10, 3)
    K = rbf(X, X)

    # Check eigenvalues are positive
    eigenvals = np.linalg.eigvals(K)
    assert np.all(eigenvals > -1e-10)  # Allow small numerical errors


def test_dot_product_basic():
    """Test basic dot product kernel functionality."""
    dp = DotProduct(scale=1.0)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Y = np.array([[1.0, 0.0], [0.0, 1.0]])

    K = dp(X, Y)
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert_allclose(K, expected)


def test_dot_product_scale():
    """Test dot product kernel with different scale."""
    dp = DotProduct(scale=2.0)
    X = np.array([[1.0, 2.0]])
    Y = np.array([[1.0, 0.0]])

    K = dp(X, Y)
    expected = np.array([[4.0]])  # 1.0 * 4.0
    assert_allclose(K, expected)


def test_dot_product_1d_input():
    """Test dot product with 1D input."""
    dp = DotProduct(scale=1.0)
    X = np.array([1.0, 2.0])
    Y = np.array([1.0, 0.0])

    K = dp(X, Y)
    # After ensure_2d, X becomes [[1.0, 2.0]] and Y becomes [[1.0, 0.0]]
    # So we get a 2x2 matrix: [[1.0*1.0, 1.0*0.0], [2.0*1.0, 2.0*0.0]]
    expected = np.array([[1.0, 0.0], [2.0, 0.0]])
    assert_allclose(K, expected)


def test_constant_basic():
    """Test basic constant kernel functionality."""
    const = Constant(value=2.0)
    X = np.array([[1.0], [2.0]])
    Y = np.array([[3.0], [4.0]])

    K = const(X, Y)
    expected = np.array([[4.0, 4.0], [4.0, 4.0]])  # value^2
    assert_allclose(K, expected)


def test_constant_1d_input():
    """Test constant kernel with 1D input."""
    const = Constant(value=3.0)
    X = np.array([1.0, 2.0])
    Y = np.array([3.0, 4.0])

    K = const(X, Y)
    expected = np.array([[9.0, 9.0], [9.0, 9.0]])
    assert_allclose(K, expected)


def test_linear_basic():
    """Test basic linear kernel functionality."""
    linear = Linear(m=0.0, scale=1.0)
    X = np.array([[1.0], [2.0]])
    Y = np.array([[1.0], [2.0]])

    K = linear(X, Y)
    expected = np.array([[1.0, 2.0], [2.0, 4.0]])
    assert_allclose(K, expected)


def test_linear_with_offset():
    """Test linear kernel with non-zero offset."""
    linear = Linear(m=1.0, scale=1.0)
    X = np.array([[1.0], [2.0]])
    Y = np.array([[1.0], [2.0]])

    K = linear(X, Y)
    expected = np.array([[0.0, 0.0], [0.0, 1.0]])
    assert_allclose(K, expected)


def test_linear_scale():
    """Test linear kernel with different scale."""
    linear = Linear(m=0.0, scale=2.0)
    X = np.array([[1.0], [2.0]])
    Y = np.array([[1.0], [2.0]])

    K = linear(X, Y)
    expected = np.array([[4.0, 8.0], [8.0, 16.0]])
    assert_allclose(K, expected)


def test_linear_1d_input():
    """Test linear kernel with 1D input."""
    linear = Linear(m=0.0, scale=1.0)
    X = np.array([1.0, 2.0])
    Y = np.array([1.0, 2.0])

    K = linear(X, Y)
    expected = np.array([[1.0, 2.0], [2.0, 4.0]])
    assert_allclose(K, expected)


def test_periodic_basic():
    """Test basic periodic kernel functionality."""
    periodic = Periodic(sigma=1.0, period=1.0, lengthscale=1.0)
    X = np.array([[0.0], [0.5], [1.0]])
    Y = np.array([[0.0], [0.5], [1.0]])

    K = periodic(X, Y)
    # Should be 1.0 on diagonal
    assert_allclose(np.diag(K), 1.0)


def test_periodic_periodicity():
    """Test periodic kernel respects periodicity."""
    periodic = Periodic(sigma=1.0, period=1.0, lengthscale=1.0)
    X = np.array([[0.0], [1.0]])
    Y = np.array([[0.0], [1.0]])

    K = periodic(X, Y)
    # Points separated by one period should be identical
    assert_allclose(K[0, 1], K[1, 0])
    assert_allclose(K[0, 0], K[1, 1])


def test_periodic_1d_input():
    """Test periodic kernel with 1D input."""
    periodic = Periodic(sigma=1.0, period=1.0, lengthscale=1.0)
    X = np.array([0.0, 1.0])
    Y = np.array([0.0, 1.0])

    K = periodic(X, Y)
    assert_allclose(np.diag(K), 1.0)


def test_periodic_different_periods():
    """Test periodic kernel with different periods for different dimensions."""
    periodic = Periodic(sigma=1.0, period=[1.0, 2.0], lengthscale=1.0)
    X = np.array([[0.0, 0.0], [1.0, 2.0]])
    Y = np.array([[0.0, 0.0], [1.0, 2.0]])

    K = periodic(X, Y)
    assert_allclose(np.diag(K), 1.0)


def test_sum_kernel_basic():
    """Test basic sum kernel functionality."""
    rbf = RBF(sigma=1.0, scale=1.0)
    const = Constant(value=1.0)
    sum_kernel = SumKernel(rbf, const)

    X = np.array([[0.0], [1.0]])
    Y = np.array([[0.0], [1.0]])

    K = sum_kernel(X, Y)
    K_rbf = rbf(X, Y)
    K_const = const(X, Y)
    expected = K_rbf + K_const

    assert_allclose(K, expected)


def test_sum_kernel_nested():
    """Test sum kernel with nested sum kernels."""
    rbf1 = RBF(sigma=1.0)
    rbf2 = RBF(sigma=2.0)
    const = Constant(value=1.0)

    sum_kernel = rbf1 + rbf2 + const

    X = np.array([[0.0], [1.0]])
    Y = np.array([[0.0], [1.0]])

    K = sum_kernel(X, Y)
    K_expected = rbf1(X, Y) + rbf2(X, Y) + const(X, Y)

    assert_allclose(K, K_expected)


def test_sum_kernel_repr():
    """Test sum kernel string representation."""
    rbf = RBF(sigma=1.0)
    const = Constant(value=1.0)
    sum_kernel = SumKernel(rbf, const)

    repr_str = repr(sum_kernel)
    assert "SumKernel" in repr_str
    assert "RBF" in repr_str
    assert "Constant" in repr_str


def test_sum_kernel_with_new():
    """Test sum kernel with_new method."""
    rbf = RBF(sigma=1.0)
    const = Constant(value=1.0)
    sum_kernel = SumKernel(rbf, const)

    new_params: dict[str, float | list[float]] = {
        "0-sigma": 2.0,
        "0-scale": 1.0,
        "1-value": 2.0,
    }
    new_sum = sum_kernel.with_new(new_params)

    assert new_sum.kernels[0].params["sigma"] == 2.0
    assert new_sum.kernels[1].params["value"] == 2.0


def test_product_kernel_basic():
    """Test basic product kernel functionality."""
    rbf = RBF(sigma=1.0, scale=1.0)
    const = Constant(value=2.0)
    prod_kernel = ProductKernel(rbf, const)

    X = np.array([[0.0], [1.0]])
    Y = np.array([[0.0], [1.0]])

    K = prod_kernel(X, Y)
    K_rbf = rbf(X, Y)
    K_const = const(X, Y)
    expected = K_rbf * K_const

    assert_allclose(K, expected)


def test_product_kernel_nested():
    """Test product kernel with nested product kernels."""
    rbf1 = RBF(sigma=1.0)
    rbf2 = RBF(sigma=2.0)
    const = Constant(value=1.0)

    prod_kernel = rbf1 * rbf2 * const

    X = np.array([[0.0], [1.0]])
    Y = np.array([[0.0], [1.0]])

    K = prod_kernel(X, Y)
    K_expected = rbf1(X, Y) * rbf2(X, Y) * const(X, Y)

    assert_allclose(K, K_expected)


def test_product_kernel_repr():
    """Test product kernel string representation."""
    rbf = RBF(sigma=1.0)
    const = Constant(value=1.0)
    prod_kernel = ProductKernel(rbf, const)

    repr_str = repr(prod_kernel)
    assert "ProductKernel" in repr_str
    assert "RBF" in repr_str
    assert "Constant" in repr_str


def test_power_kernel_basic():
    """Test basic power kernel functionality."""
    rbf = RBF(sigma=1.0, scale=1.0)
    power_kernel = PowerKernel(power=2.0, kernel=rbf)

    X = np.array([[0.0], [1.0]])
    Y = np.array([[0.0], [1.0]])

    K = power_kernel(X, Y)
    K_rbf = rbf(X, Y)
    expected = K_rbf**2.0

    assert_allclose(K, expected)


def test_power_kernel_with_new():
    """Test power kernel with_new method."""
    rbf = RBF(sigma=1.0, scale=1.0)
    power_kernel = PowerKernel(power=2.0, kernel=rbf)

    new_params: dict[str, float | list[float]] = {"sigma": 2.0, "scale": 1.0}
    new_power = power_kernel.with_new(new_params)

    assert new_power.power == 2.0
    assert new_power.kernel.params["sigma"] == 2.0


def test_power_kernel_repr():
    """Test power kernel string representation."""
    rbf = RBF(sigma=1.0, scale=1.0)
    power_kernel = PowerKernel(power=2.0, kernel=rbf)

    repr_str = repr(power_kernel)
    assert "PowerKernel" in repr_str
    assert "power=2.00e+00" in repr_str


def test_complex_kernel_expression():
    """Test complex kernel expressions."""
    rbf = RBF(sigma=1.0, scale=1.0)
    const = Constant(value=0.1)
    linear = Linear(m=0.0, scale=1.0)

    # (RBF + Constant) * Linear
    complex_kernel = (rbf + const) * linear

    X = np.array([[0.0], [1.0]])
    Y = np.array([[0.0], [1.0]])

    K = complex_kernel(X, Y)
    K_rbf_const = rbf(X, Y) + const(X, Y)
    K_linear = linear(X, Y)
    expected = K_rbf_const * K_linear

    assert_allclose(K, expected)


def test_kernel_power_operations():
    """Test kernel power operations."""
    rbf = RBF(sigma=1.0, scale=1.0)
    power_kernel = rbf**2.0

    X = np.array([[0.0], [1.0]])
    Y = np.array([[0.0], [1.0]])

    K = power_kernel(X, Y)
    K_rbf = rbf(X, Y)
    expected = K_rbf**2.0

    assert_allclose(K, expected)


def test_kernel_arithmetic_precedence():
    """Test kernel arithmetic precedence."""
    rbf = RBF(sigma=1.0)
    const = Constant(value=1.0)
    linear = Linear(m=0.0)

    # Test: rbf + const * linear should be rbf + (const * linear)
    result1 = rbf + const * linear
    result2 = rbf + (const * linear)

    X = np.array([[0.0], [1.0]])
    Y = np.array([[0.0], [1.0]])

    K1 = result1(X, Y)
    K2 = result2(X, Y)

    assert_allclose(K1, K2)


def test_empty_arrays():
    """Test kernels with empty arrays."""
    rbf = RBF(sigma=1.0, scale=1.0)
    X = np.empty((0, 1))
    Y = np.empty((0, 1))

    K = rbf(X, Y)
    assert K.shape == (0, 0)


def test_single_point():
    """Test kernels with single point."""
    rbf = RBF(sigma=1.0, scale=1.0)
    X = np.array([[1.0]])
    Y = np.array([[2.0]])

    K = rbf(X, Y)
    assert K.shape == (1, 1)
    assert K[0, 0] > 0


def test_different_shapes():
    """Test kernels with different input shapes."""
    rbf = RBF(sigma=1.0, scale=1.0)
    X = np.array([[1.0], [2.0]])  # 2x1
    Y = np.array([[3.0], [4.0], [5.0]])  # 3x1

    K = rbf(X, Y)
    assert K.shape == (2, 3)


def test_high_dimensional_input():
    """Test kernels with high-dimensional input."""
    rbf = RBF(sigma=1.0, scale=1.0)
    X = np.random.randn(5, 10)  # 5 points, 10 dimensions
    Y = np.random.randn(3, 10)  # 3 points, 10 dimensions

    K = rbf(X, Y)
    assert K.shape == (5, 3)
    assert np.all(np.isfinite(K))
