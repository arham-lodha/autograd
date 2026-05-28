"""Gradient correctness tests — every op's backward is checked against
numerical finite differences.  This is the most important test file."""

import numpy as np
import pytest
from autograd.eager import Tensor, Engine
from tests.eager.conftest import numerical_grad

TOL = 1e-5  # tolerance for numerical vs analytical gradient comparison


# ===========================================================================
# Helpers
# ===========================================================================

def _check_grad(engine, loss_fn, *tensors, tol=TOL):
    """Run backward analytically, then compare each tensor's .grad to the
    numerical gradient of loss_fn w.r.t. that tensor."""
    # Compute numerical gradients first (these don't touch .grad)
    num_grads = []
    for t in tensors:
        num_grads.append(numerical_grad(loss_fn, t))

    # Zero grads, build fresh graph, run analytical backward
    for t in tensors:
        t.grad = np.zeros_like(t.data)

    loss = loss_fn()
    eng = Engine()  # fresh engine, no stale cache
    eng.backward(loss)

    for t, expected in zip(tensors, num_grads):
        np.testing.assert_allclose(
            t.grad, expected, atol=tol, rtol=tol,
            err_msg=f"Gradient mismatch for tensor with shape {t.shape}",
        )


# ===========================================================================
# Scalar / element-wise ops
# ===========================================================================

class TestAddBackward:
    def test_add_two_tensors(self, engine):
        a = Tensor(np.array([1.0, 2.0, 3.0]))
        b = Tensor(np.array([4.0, 5.0, 6.0]))
        _check_grad(engine, lambda: (a + b).sum(), a, b)

    def test_add_scalar(self, engine):
        a = Tensor(np.array([1.0, 2.0]))
        _check_grad(engine, lambda: (a + 5.0).sum(), a)

    def test_add_chain(self, engine):
        a = Tensor(np.array([1.0]))
        b = Tensor(np.array([2.0]))
        c = Tensor(np.array([3.0]))
        _check_grad(engine, lambda: (a + b + c).sum(), a, b, c)

    def test_add_with_retain(self, engine):
        a = Tensor(np.array([1.0, 2.0]), retain=True)
        b = Tensor(np.array([3.0, 4.0]))
        _check_grad(engine, lambda: (a + b).sum(), a, b)


class TestSubBackward:
    def test_sub_tensors(self, engine):
        a = Tensor(np.array([5.0, 3.0]))
        b = Tensor(np.array([1.0, 2.0]))
        _check_grad(engine, lambda: (a - b).sum(), a, b)

    def test_rsub(self, engine):
        a = Tensor(np.array([2.0, 3.0]))
        _check_grad(engine, lambda: (10.0 - a).sum(), a)


class TestMulBackward:
    def test_mul_two_tensors(self, engine):
        a = Tensor(np.array([2.0, 3.0]))
        b = Tensor(np.array([4.0, 5.0]))
        _check_grad(engine, lambda: (a * b).sum(), a, b)

    def test_mul_scalar(self, engine):
        a = Tensor(np.array([2.0, 3.0]))
        _check_grad(engine, lambda: (a * 7.0).sum(), a)

    def test_mul_chain(self, engine):
        a = Tensor(np.array([2.0]))
        b = Tensor(np.array([3.0]))
        c = Tensor(np.array([4.0]))
        _check_grad(engine, lambda: (a * b * c).sum(), a, b, c)

    def test_mul_with_retain(self, engine):
        a = Tensor(np.array([2.0, 3.0]), retain=True)
        b = Tensor(np.array([4.0, 5.0]))
        _check_grad(engine, lambda: (a * b).sum(), a, b)


class TestPowBackward:
    def test_pow_scalar_exponent(self, engine):
        a = Tensor(np.array([2.0, 3.0]))
        _check_grad(engine, lambda: (a ** 3.0).sum(), a)

    def test_pow_tensor_exponent(self, engine):
        a = Tensor(np.array([2.0, 3.0]))
        b = Tensor(np.array([3.0, 2.0]))
        _check_grad(engine, lambda: (a ** b).sum(), a, b)

    def test_pow_fractional(self, engine):
        a = Tensor(np.array([4.0, 9.0]))
        _check_grad(engine, lambda: (a ** 0.5).sum(), a)

    def test_pow_negative_exponent(self, engine):
        a = Tensor(np.array([2.0, 4.0]))
        _check_grad(engine, lambda: (a ** -1.0).sum(), a)


class TestDivBackward:
    def test_div_tensor_scalar(self, engine):
        a = Tensor(np.array([6.0, 9.0]))
        _check_grad(engine, lambda: (a / 3.0).sum(), a)

    def test_div_tensor_tensor(self, engine):
        a = Tensor(np.array([6.0, 8.0]))
        b = Tensor(np.array([2.0, 4.0]))
        _check_grad(engine, lambda: (a / b).sum(), a, b)

    def test_rdiv(self, engine):
        a = Tensor(np.array([2.0, 4.0]))
        _check_grad(engine, lambda: (6.0 / a).sum(), a)


class TestNegBackward:
    def test_neg(self, engine):
        a = Tensor(np.array([1.0, -2.0, 3.0]))
        _check_grad(engine, lambda: (-a).sum(), a)


class TestLogBackward:
    def test_log(self, engine):
        a = Tensor(np.array([1.0, 2.0, 3.0]))
        _check_grad(engine, lambda: a.log().sum(), a)


# ===========================================================================
# Matrix / higher-dim ops
# ===========================================================================

class TestTransposeBackward:
    def test_transpose_2d(self, engine):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        _check_grad(engine, lambda: a.transpose().sum(), a)


class TestMatmulBackward:
    def test_matmul_2d(self, engine):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
        _check_grad(engine, lambda: (a @ b).sum(), a, b)

    def test_matmul_nonsquare(self, engine):
        a = Tensor(np.random.randn(3, 4))
        b = Tensor(np.random.randn(4, 5))
        _check_grad(engine, lambda: (a @ b).sum(), a, b)

    def test_matmul_ndarray_rhs(self, engine):
        a = Tensor(np.random.randn(3, 4))
        W = np.random.randn(4, 2)
        _check_grad(engine, lambda: (a @ W).sum(), a)

    def test_rmatmul_tensor_lhs(self, engine):
        a = Tensor(np.random.randn(2, 3))
        b = Tensor(np.random.randn(3, 4))
        _check_grad(engine, lambda: (a @ b).sum(), a, b)

    def test_matmul_batched(self, engine):
        a = Tensor(np.random.randn(2, 3, 4))
        b = Tensor(np.random.randn(2, 4, 5))
        _check_grad(engine, lambda: (a @ b).sum(), a, b)


# ===========================================================================
# Reductions & shape ops
# ===========================================================================

class TestSumBackward:
    def test_sum_all(self, engine):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        _check_grad(engine, lambda: a.sum(), a)

    def test_sum_axis0(self, engine):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        _check_grad(engine, lambda: a.sum(axis=0).sum(), a)

    def test_sum_axis1_keepdims(self, engine):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        _check_grad(engine, lambda: a.sum(axis=1, keepdims=True).sum(), a)


class TestBroadcastBackward:
    def test_broadcast_same_ndim(self, engine):
        """Broadcast (1,3) -> (4,3): no dimension reduction needed in _unbroadcast."""
        a = Tensor(np.array([[1.0, 2.0, 3.0]]))  # shape (1,3)
        _check_grad(engine, lambda: a.broadcast_to((4, 3)).sum(), a)

    def test_broadcast_col_vector(self, engine):
        a = Tensor(np.array([[1.0], [2.0]]))  # shape (2,1)
        _check_grad(engine, lambda: a.broadcast_to((2, 5)).sum(), a)

    def test_broadcast_adds_dimension(self, engine):
        """Broadcast (3,) -> (4,3): triggers the while-loop bug in _unbroadcast."""
        import signal

        def handler(signum, frame):
            raise TimeoutError("_unbroadcast infinite loop detected")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(3)
        try:
            a = Tensor(np.array([1.0, 2.0, 3.0]))
            _check_grad(engine, lambda: a.broadcast_to((4, 3)).sum(), a)
        finally:
            signal.alarm(0)


class TestReshapeBackward:
    def test_reshape(self, engine):
        a = Tensor(np.arange(1.0, 7.0))
        _check_grad(engine, lambda: (a.reshape((2, 3)) * 2).sum(), a)


class TestExpandDimsBackward:
    def test_expand_dims(self, engine):
        a = Tensor(np.array([1.0, 2.0, 3.0]))
        _check_grad(engine, lambda: (a.expand_dims(0) * 2).sum(), a)


class TestSqueezeBackward:
    def test_squeeze(self, engine):
        a = Tensor(np.array([[1.0, 2.0, 3.0]]))
        _check_grad(engine, lambda: (a.squeeze(axis=0) * 2).sum(), a)


class TestSwapAxisBackward:
    def test_swap_axis_small(self, engine):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        _check_grad(engine, lambda: (a.swap_axis(0, 1) * 2).sum(), a)
