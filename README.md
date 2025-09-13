<div align="center">

# `mini-gpr`

[![Tests](https://github.com/jla-gardner/mini-gpr/actions/workflows/tests.yaml/badge.svg?branch=main)](https://github.com/jla-gardner/mini-gpr/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/jla-gardner/mini-gpr/branch/main/graph/badge.svg)](https://codecov.io/gh/jla-gardner/mini-gpr)
[![PyPI](https://img.shields.io/pypi/v/mini-gpr)](https://pypi.org/project/mini-gpr/)

<img src="docs/_static/1d-gpr.gif" alt="1D GPR" width="400">

</div>

A minimal reference implementation of Gaussian Process Regression in pure NumPy.

## Installation

```bash
pip install mini-gpr
```

## Documentation

📚 **[View the full documentation](https://jla-gardner.github.io/mini-gpr/)**

The documentation includes tutorials, API reference, and examples.

## Features

- Full and sparse GPR model implementations
- Common kernel functions (RBF, Linear, Periodic, etc.)
- Automated hyperparameter optimization
- Strong typing with jaxtyping
- Pure NumPy implementation (no external dependencies beyond NumPy, Matplotlib)