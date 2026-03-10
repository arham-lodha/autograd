"""Tests for Tensor construction, properties, and representation."""

import numpy as np
import pytest
from autograd import Tensor
from autograd.ops import Operation


class TestTensorCreation:
    """Verify that tensors can be built from every supported input type."""

    def test_from_float(self):
        t = Tensor(3.0)
        assert t.data.shape == (1,)
        np.testing.assert_allclose(t.data, [3.0])

    def test_from_int(self):
        t = Tensor(7)
        np.testing.assert_allclose(t.data, [7])

    def test_from_list(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)

    def test_from_ndarray(self):
        arr = np.array([[1, 2], [3, 4]], dtype=float)
        t = Tensor(arr)
        np.testing.assert_array_equal(t.data, arr)

    def test_default(self):
        t = Tensor()
        np.testing.assert_allclose(t.data, [1.0])

    def test_grad_initialized_to_zero(self):
        t = Tensor(np.ones((2, 3)))
        np.testing.assert_array_equal(t.grad, np.zeros((2, 3)))

    def test_operation_default(self):
        t = Tensor(1.0)
        assert t.operation == Operation.CONSTANT

    def test_prev_default_empty(self):
        t = Tensor(1.0)
        assert t.prev == []

    def test_retain_flag(self):
        t = Tensor(1.0, retain=True)
        assert t.retain is True


class TestTensorProperties:
    def test_shape(self):
        t = Tensor(np.zeros((3, 4, 5)))
        assert t.shape == (3, 4, 5)

    def test_transpose_property(self):
        t = Tensor(np.arange(6.0).reshape(2, 3))
        np.testing.assert_array_equal(t.T.data, t.data.T)

    def test_repr(self):
        t = Tensor(1.0)
        r = repr(t)
        assert "Tensor" in r
        assert "CONSTANT" in r
