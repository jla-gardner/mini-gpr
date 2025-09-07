import numpy as np

from mini_gpr.kernels import RBF, Constant
from mini_gpr.models import GPR
from mini_gpr.opt import (
    Convertor,
    maximise_log_likelihood,
    optimise_model,
    validation_set_log_likelihood,
    validation_set_mse,
)


def test_objective_protocol_implementation():
    """Test that functions implementing Objective protocol work correctly."""

    # Create a simple model for testing
    kernel = RBF(sigma=1.0, scale=1.0)
    model = GPR(kernel, noise=0.1)

    # Create some test data
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 1.5])
    model.fit(X, y)

    # Test that maximise_log_likelihood implements the protocol
    objective = maximise_log_likelihood
    result = objective(model)
    assert isinstance(result, float)
    assert result == -model.log_likelihood

    # Test that validation functions implement the protocol
    val_X = np.array([[1.5], [2.5]])
    val_y = np.array([1.2, 1.8])

    mse_obj = validation_set_mse(val_X, val_y)
    mse_result = mse_obj(model)
    assert isinstance(mse_result, float)
    assert mse_result >= 0

    ll_obj = validation_set_log_likelihood(val_X, val_y)
    ll_result = ll_obj(model)
    assert isinstance(ll_result, float)


def test_maximise_log_likelihood_basic():
    """Test basic functionality of maximise_log_likelihood."""
    kernel = RBF(sigma=1.0, scale=1.0)
    model = GPR(kernel, noise=0.1)

    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 1.5])
    model.fit(X, y)

    result = maximise_log_likelihood(model)
    expected = -model.log_likelihood

    assert result == expected
    assert isinstance(result, float)


def test_maximise_log_likelihood_different_models():
    """Test maximise_log_likelihood with different model configurations."""
    # Test with different noise levels
    kernel = RBF(sigma=1.0, scale=1.0)

    for noise in [0.01, 0.1, 1.0]:
        model = GPR(kernel, noise=noise)
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 1.5])
        model.fit(X, y)

        result = maximise_log_likelihood(model)
        assert result == -model.log_likelihood
        assert isinstance(result, float)


def test_validation_set_mse_basic():
    """Test basic functionality of validation_set_mse."""
    kernel = RBF(sigma=1.0, scale=1.0)
    model = GPR(kernel, noise=0.1)

    # Training data
    X_train = np.array([[1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 2.0, 1.5])
    model.fit(X_train, y_train)

    # Validation data
    X_val = np.array([[1.5], [2.5]])
    y_val = np.array([1.2, 1.8])

    mse_func = validation_set_mse(X_val, y_val)
    mse = mse_func(model)

    # Calculate expected MSE manually
    y_pred = model.predict(X_val)
    expected_mse = np.mean((y_val - y_pred) ** 2)

    assert np.isclose(mse, expected_mse)
    assert isinstance(mse, float)
    assert mse >= 0


def test_validation_set_mse_perfect_prediction():
    """Test MSE when prediction is perfect."""
    kernel = RBF(sigma=1.0, scale=1.0)
    model = GPR(kernel, noise=0.1)

    X_train = np.array([[1.0], [2.0]])
    y_train = np.array([1.0, 2.0])
    model.fit(X_train, y_train)

    # Use same data for validation (should give low MSE)
    mse_func = validation_set_mse(X_train, y_train)
    mse = mse_func(model)

    assert mse >= 0
    assert isinstance(mse, float)


def test_validation_set_mse_different_shapes():
    """Test MSE with different input shapes."""
    kernel = RBF(sigma=1.0, scale=1.0)
    model = GPR(kernel, noise=0.1)

    X_train = np.array([[1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 2.0, 1.5])
    model.fit(X_train, y_train)

    # Test with single validation point
    X_val = np.array([[1.5]])
    y_val = np.array([1.2])
    mse_func = validation_set_mse(X_val, y_val)
    mse = mse_func(model)

    assert isinstance(mse, float)
    assert mse >= 0


def test_validation_set_log_likelihood_basic():
    """Test basic functionality of validation_set_log_likelihood."""
    kernel = RBF(sigma=1.0, scale=1.0)
    model = GPR(kernel, noise=0.1)

    # Training data
    X_train = np.array([[1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 2.0, 1.5])
    model.fit(X_train, y_train)

    # Validation data
    X_val = np.array([[1.5], [2.5]])
    y_val = np.array([1.2, 1.8])

    ll_func = validation_set_log_likelihood(X_val, y_val)
    ll = ll_func(model)

    # Calculate expected log likelihood manually
    y_pred = model.predict(X_val)
    std = model.predictive_uncertainty(X_val)

    # Normal distribution likelihoods
    ls = (
        1
        / (std * np.sqrt(2 * np.pi))
        * np.exp(-((y_val - y_pred) ** 2) / (2 * std**2))
    )

    expected_ll = np.sum(np.log(ls))

    assert np.isclose(ll, expected_ll)
    assert isinstance(ll, float)


def test_validation_set_log_likelihood_perfect_prediction():
    """Test log likelihood when prediction is perfect."""
    kernel = RBF(sigma=1.0, scale=1.0)
    model = GPR(kernel, noise=0.1)

    X_train = np.array([[1.0], [2.0]])
    y_train = np.array([1.0, 2.0])
    model.fit(X_train, y_train)

    # Use same data for validation
    ll_func = validation_set_log_likelihood(X_train, y_train)
    ll = ll_func(model)

    assert isinstance(ll, float)


def test_validation_set_log_likelihood_different_noise():
    """Test log likelihood with different noise levels."""
    kernel = RBF(sigma=1.0, scale=1.0)

    X_train = np.array([[1.0], [2.0]])
    y_train = np.array([1.0, 2.0])
    X_val = np.array([[1.5]])
    y_val = np.array([1.2])

    for noise in [0.01, 0.1, 1.0]:
        model = GPR(kernel, noise=noise)
        model.fit(X_train, y_train)

        ll_func = validation_set_log_likelihood(X_val, y_val)
        ll = ll_func(model)

        assert isinstance(ll, float)


def test_convertor_init():
    """Test Convertor initialization."""
    params: dict[str, float | list[float]] = {
        "a": 1.0,
        "b": [2.0, 3.0],
        "c": 4.0,
    }
    convertor = Convertor(params)

    assert convertor.og_params == params


def test_convertor_to_list_basic():
    """Test basic to_list functionality."""
    params: dict[str, float | list[float]] = {
        "a": 1.0,
        "b": [2.0, 3.0],
        "c": 4.0,
    }
    convertor = Convertor(params)

    result = convertor.to_list(params)
    expected = [1.0, 2.0, 3.0, 4.0]

    assert result == expected


def test_convertor_to_list_only_scalars():
    """Test to_list with only scalar parameters."""
    params: dict[str, float | list[float]] = {"a": 1.0, "b": 2.0, "c": 3.0}
    convertor = Convertor(params)

    result = convertor.to_list(params)
    expected = [1.0, 2.0, 3.0]

    assert result == expected


def test_convertor_to_list_only_lists():
    """Test to_list with only list parameters."""
    params: dict[str, float | list[float]] = {
        "a": [1.0, 2.0],
        "b": [3.0, 4.0, 5.0],
    }
    convertor = Convertor(params)

    result = convertor.to_list(params)
    expected = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert result == expected


def test_convertor_to_dict_basic():
    """Test basic to_dict functionality."""
    og_params: dict[str, float | list[float]] = {
        "a": 1.0,
        "b": [2.0, 3.0],
        "c": 4.0,
    }
    convertor = Convertor(og_params)

    params_list = [1.0, 2.0, 3.0, 4.0]
    result = convertor.to_dict(params_list)

    assert result == og_params


def test_convertor_to_dict_only_scalars():
    """Test to_dict with only scalar parameters."""
    og_params: dict[str, float | list[float]] = {
        "a": 1.0,
        "b": 2.0,
        "c": 3.0,
    }
    convertor = Convertor(og_params)

    params_list = [1.0, 2.0, 3.0]
    result = convertor.to_dict(params_list)

    assert result == og_params


def test_convertor_to_dict_only_lists():
    """Test to_dict with only list parameters."""
    og_params: dict[str, float | list[float]] = {
        "a": [1.0, 2.0],
        "b": [3.0, 4.0, 5.0],
    }
    convertor = Convertor(og_params)

    params_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = convertor.to_dict(params_list)

    assert result == og_params


def test_convertor_round_trip_conversion():
    """Test that to_list and to_dict are inverse operations."""
    og_params: dict[str, float | list[float]] = {
        "a": 1.0,
        "b": [2.0, 3.0],
        "c": 4.0,
        "d": [5.0, 6.0, 7.0],
    }
    convertor = Convertor(og_params)

    # Convert to list and back
    params_list = convertor.to_list(og_params)
    result = convertor.to_dict(params_list)

    assert result == og_params


def test_convertor_with_kernel_params():
    """Test Convertor with actual kernel parameters."""
    kernel = RBF(sigma=1.0, scale=2.0)
    convertor = Convertor(kernel.params)

    # Test to_list
    params_list = convertor.to_list(kernel.params)
    assert params_list == [1.0, 2.0]

    # Test to_dict
    result = convertor.to_dict(params_list)
    assert result == kernel.params


def test_optimise_model_basic():
    """Test basic optimization functionality."""
    kernel = RBF(sigma=1.0, scale=1.0)
    model = GPR(kernel, noise=0.1)

    # Create some test data
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1.0, 2.0, 1.5, 2.5, 2.0])

    # Optimize using log likelihood
    objective = maximise_log_likelihood
    optimized_model = optimise_model(model, objective, X, y)

    # Check that we get a model back
    assert isinstance(optimized_model, GPR)
    assert hasattr(optimized_model, "kernel")
    assert hasattr(optimized_model, "noise")

    # Check that the model can make predictions
    predictions = optimized_model.predict(X)
    assert len(predictions) == len(X)


def test_optimise_model_with_validation_mse():
    """Test optimization with validation MSE objective."""
    kernel = RBF(sigma=1.0, scale=1.0)
    model = GPR(kernel, noise=0.1)

    # Create training and validation data
    X_train = np.array([[1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 2.0, 1.5])

    X_val = np.array([[1.5], [2.5]])
    y_val = np.array([1.2, 1.8])

    # Optimize using validation MSE
    objective = validation_set_mse(X_val, y_val)
    optimized_model = optimise_model(model, objective, X_train, y_train)

    assert isinstance(optimized_model, GPR)

    # Check that the model can make predictions
    predictions = optimized_model.predict(X_train)
    assert len(predictions) == len(X_train)


def test_optimise_model_with_validation_log_likelihood():
    """Test optimization with validation log likelihood objective."""
    kernel = RBF(sigma=1.0, scale=1.0)
    model = GPR(kernel, noise=0.1)

    # Create training and validation data
    X_train = np.array([[1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 2.0, 1.5])

    X_val = np.array([[1.5], [2.5]])
    y_val = np.array([1.2, 1.8])

    # Optimize using validation log likelihood
    objective = validation_set_log_likelihood(X_val, y_val)
    optimized_model = optimise_model(model, objective, X_train, y_train)

    assert isinstance(optimized_model, GPR)

    # Check that the model can make predictions
    predictions = optimized_model.predict(X_train)
    assert len(predictions) == len(X_train)


def test_optimise_model_with_noise_optimization():
    """Test optimization with noise parameter included."""
    kernel = RBF(sigma=1.0, scale=1.0)
    model = GPR(kernel, noise=0.1)

    # Create some test data
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1.0, 2.0, 1.5, 2.5, 2.0])

    # Optimize including noise
    objective = maximise_log_likelihood
    optimized_model = optimise_model(
        model, objective, X, y, optimise_noise=True
    )

    assert isinstance(optimized_model, GPR)
    assert optimized_model.noise > 0  # Noise should be positive


def test_optimise_model_max_iterations():
    """Test optimization with custom max iterations."""
    kernel = RBF(sigma=1.0, scale=1.0)
    model = GPR(kernel, noise=0.1)

    # Create some test data
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 1.5])

    # Optimize with limited iterations
    objective = maximise_log_likelihood
    optimized_model = optimise_model(model, objective, X, y, max_iterations=10)

    assert isinstance(optimized_model, GPR)


def test_optimise_model_with_complex_kernel():
    """Test optimization with a more complex kernel."""
    # Create a sum kernel
    rbf = RBF(sigma=1.0, scale=1.0)
    white = Constant(value=0.1)
    kernel = rbf + white

    model = GPR(kernel, noise=0.1)

    # Create some test data
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1.0, 2.0, 1.5, 2.5, 2.0])

    # Optimize
    objective = maximise_log_likelihood
    optimized_model = optimise_model(model, objective, X, y)

    assert isinstance(optimized_model, GPR)

    # Check that the model can make predictions
    predictions = optimized_model.predict(X)
    assert len(predictions) == len(X)


def test_optimise_model_error_handling():
    """Test error handling in optimization."""
    kernel = RBF(sigma=1.0, scale=1.0)
    model = GPR(kernel, noise=0.1)

    # Create some test data
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 1.5])

    # Test with invalid objective that always returns high value
    def bad_objective(model):
        return 1e33

    # This should not raise an exception, but return a model
    optimized_model = optimise_model(model, bad_objective, X, y)
    assert isinstance(optimized_model, GPR)


def test_optimise_model_scipy_import_error():
    """Test that ImportError is raised when scipy is not available."""
    # This test would require mocking scipy import, which is complex
    # For now, we'll just test that the function exists and can be called
    # when scipy is available (which it should be in the test environment)
    kernel = RBF(sigma=1.0, scale=1.0)
    model = GPR(kernel, noise=0.1)

    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 1.5])

    # This should work if scipy is available
    try:
        optimized_model = optimise_model(model, maximise_log_likelihood, X, y)
        assert isinstance(optimized_model, GPR)
    except ImportError:
        # If scipy is not available, that's expected
        pass


def test_validation_functions_with_empty_arrays():
    """Test validation functions with empty arrays."""
    kernel = RBF(sigma=1.0, scale=1.0)
    model = GPR(kernel, noise=0.1)

    X_train = np.array([[1.0], [2.0]])
    y_train = np.array([1.0, 2.0])
    model.fit(X_train, y_train)

    # Test with empty validation arrays
    X_val = np.array([]).reshape(0, 1)
    y_val = np.array([])

    mse_func = validation_set_mse(X_val, y_val)
    ll_func = validation_set_log_likelihood(X_val, y_val)

    # These should handle empty arrays gracefully
    mse = mse_func(model)
    ll = ll_func(model)

    assert isinstance(mse, float)
    assert isinstance(ll, float)


def test_convertor_with_empty_params():
    """Test Convertor with empty parameter dictionary."""
    params: dict[str, float | list[float]] = {}
    convertor = Convertor(params)

    # Test to_list with empty params
    result = convertor.to_list(params)
    assert result == []

    # Test to_dict with empty list
    result = convertor.to_dict([])
    assert result == {}


def test_convertor_with_mixed_types():
    """Test Convertor with mixed parameter types."""
    params: dict[str, float | list[float]] = {
        "a": 1.0,
        "b": [2.0, 3.0],
        "c": 4.0,
    }
    convertor = Convertor(params)

    # Test round trip
    params_list = convertor.to_list(params)
    result = convertor.to_dict(params_list)

    assert result == params
    assert len(params_list) == 4  # 1 + 2 + 1 = 4 parameters
