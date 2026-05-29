"""Shared fixtures and numerical-gradient utilities for the entire test suite."""

import pytest
import numpy as np
from autograd.eager import Tensor, Engine


# ---------------------------------------------------------------------------
# Numerical gradient helper (central finite differences)
# ---------------------------------------------------------------------------

def numerical_grad(f, x_tensor, eps=1e-5):
    """Compute numerical gradient of scalar-valued f w.r.t. x_tensor.data.

    Works for tensors of any shape.  Returns an ndarray the same shape as
    x_tensor.data.
    """
    grad = np.zeros_like(x_tensor.data)
    it = np.nditer(x_tensor.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = float(x_tensor.data[idx])

        x_tensor.data[idx] = old + eps
        fxp = float(f().data.sum())

        x_tensor.data[idx] = old - eps
        fxm = float(f().data.sum())

        grad[idx] = (fxp - fxm) / (2 * eps)
        x_tensor.data[idx] = old
        it.iternext()
    return grad


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return Engine()


@pytest.fixture
def rng():
    """Seeded RNG for reproducibility."""
    return np.random.default_rng(42)
