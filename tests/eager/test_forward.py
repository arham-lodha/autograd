"""Tests for the forward pass (data correctness) of every operation."""

import numpy as np
import pytest
from autograd.eager import Tensor


class TestArithmeticForward:
    """Element-wise arithmetic: +, -, *, /, **, neg."""

    def test_add_tensor_tensor(self):
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([3.0, 4.0]))
        np.testing.assert_allclose((a + b).data, [4.0, 6.0])

    def test_add_tensor_scalar(self):
        a = Tensor(np.array([1.0, 2.0]))
        np.testing.assert_allclose((a + 10).data, [11.0, 12.0])

    def test_radd(self):
        a = Tensor(np.array([1.0]))
        np.testing.assert_allclose((5 + a).data, [6.0])

    def test_sub_tensor_tensor(self):
        a = Tensor(np.array([5.0, 3.0]))
        b = Tensor(np.array([1.0, 1.0]))
        np.testing.assert_allclose((a - b).data, [4.0, 2.0])

    def test_rsub(self):
        a = Tensor(np.array([2.0]))
        np.testing.assert_allclose((10 - a).data, [8.0])

    def test_mul_tensor_tensor(self):
        a = Tensor(np.array([2.0, 3.0]))
        b = Tensor(np.array([4.0, 5.0]))
        np.testing.assert_allclose((a * b).data, [8.0, 15.0])

    def test_mul_tensor_scalar(self):
        a = Tensor(np.array([2.0, 3.0]))
        np.testing.assert_allclose((a * 3).data, [6.0, 9.0])

    def test_rmul(self):
        a = Tensor(np.array([2.0]))
        np.testing.assert_allclose((3 * a).data, [6.0])

    def test_neg(self):
        a = Tensor(np.array([1.0, -2.0]))
        np.testing.assert_allclose((-a).data, [-1.0, 2.0])

    def test_pow_scalar_exponent(self):
        a = Tensor(np.array([2.0, 3.0]))
        np.testing.assert_allclose((a ** 3).data, [8.0, 27.0])

    def test_pow_tensor_exponent(self):
        a = Tensor(np.array([2.0, 3.0]))
        b = Tensor(np.array([3.0, 2.0]))
        np.testing.assert_allclose((a ** b).data, [8.0, 9.0])

    def test_div_tensor_scalar(self):
        a = Tensor(np.array([6.0, 9.0]))
        np.testing.assert_allclose((a / 3).data, [2.0, 3.0])

    def test_div_tensor_tensor(self):
        a = Tensor(np.array([6.0]))
        b = Tensor(np.array([3.0]))
        np.testing.assert_allclose((a / b).data, [2.0])

    def test_rdiv(self):
        a = Tensor(np.array([2.0]))
        np.testing.assert_allclose((6 / a).data, [3.0])


class TestUnaryForward:
    def test_log(self):
        a = Tensor(np.array([1.0, np.e, np.e ** 2]))
        np.testing.assert_allclose((a.log()).data, [0.0, 1.0, 2.0], atol=1e-7)

    def test_log_negative_raises(self):
        a = Tensor(np.array([-1.0]))
        with pytest.raises(ValueError, match="non-positive"):
            a.log()

    def test_transpose_2d(self):
        a = Tensor(np.arange(6.0).reshape(2, 3))
        np.testing.assert_array_equal(a.transpose().data, a.data.T)


class TestReductionForward:
    def test_sum_all(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_allclose(a.sum().data, [10.0])

    def test_sum_axis0(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_allclose(a.sum(axis=0).data, [4.0, 6.0])

    def test_sum_axis1_keepdims(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        s = a.sum(axis=1, keepdims=True)
        assert s.shape == (2, 1)
        np.testing.assert_allclose(s.data, [[3.0], [7.0]])


class TestShapeForward:
    def test_reshape(self):
        a = Tensor(np.arange(6.0))
        b = a.reshape((2, 3))
        assert b.shape == (2, 3)

    def test_expand_dims(self):
        a = Tensor(np.arange(3.0))
        b = a.expand_dims(0)
        assert b.shape == (1, 3)

    def test_squeeze(self):
        a = Tensor(np.arange(3.0).reshape(1, 3))
        b = a.squeeze(axis=0)
        assert b.shape == (3,)

    def test_broadcast_to(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]))
        b = a.broadcast_to((4, 3))
        assert b.shape == (4, 3)
        np.testing.assert_array_equal(b.data[0], b.data[3])

    def test_swap_axis(self):
        a = Tensor(np.arange(24.0).reshape(2, 3, 4))
        b = a.swap_axis(0, 2)
        assert b.shape == (4, 3, 2)


class TestMatmulForward:
    def test_matmul_2d(self):
        a = Tensor(np.ones((2, 3)))
        b = Tensor(np.ones((3, 4)))
        c = a @ b
        np.testing.assert_allclose(c.data, np.full((2, 4), 3.0))

    def test_matmul_ndarray_rhs(self):
        a = Tensor(np.ones((2, 3)))
        W = np.ones((3, 4))
        c = a @ W
        np.testing.assert_allclose(c.data, np.full((2, 4), 3.0))

    def test_matmul_batched(self):
        a = Tensor(np.ones((5, 2, 3)))
        b = Tensor(np.ones((5, 3, 4)))
        c = a @ b
        assert c.shape == (5, 2, 4)
