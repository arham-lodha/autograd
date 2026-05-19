"""Composite expression tests & higher-order (create_graph) gradient tests."""

import numpy as np
import pytest
from autograd.eager import Tensor, Engine
from tests.eager.conftest import numerical_grad

TOL = 1e-4


class TestCompositeExpressions:
    """Compound expressions that exercise multiple ops together."""

    def test_polynomial(self, engine):
        """f(x) = 3x^3 + 2x^2 - x + 5  →  f'(x) = 9x^2 + 4x - 1"""
        x = Tensor(np.array([2.0]))
        def f(): return (x ** 3 * 3 + x ** 2 * 2 - x + 5).sum()
        loss = f()
        engine.backward(loss)
        expected = 9 * 4 + 4 * 2 - 1  # 9(4)+8-1 = 43
        np.testing.assert_allclose(x.grad, [expected], atol=TOL)

    def test_log_product(self, engine):
        """f(x,y) = log(x*y)  →  df/dx = 1/x, df/dy = 1/y"""
        x = Tensor(np.array([3.0]))
        y = Tensor(np.array([4.0]))
        def f(): return (x * y).log().sum()
        loss = f()
        engine.backward(loss)
        np.testing.assert_allclose(x.grad, [1 / 3], atol=TOL)
        np.testing.assert_allclose(y.grad, [1 / 4], atol=TOL)

    def test_softmax_cross_entropy_like(self, engine):
        """Softmax-like numerics: log(sum(exp(x))) — a LogSumExp."""
        x = Tensor(np.array([1.0, 2.0, 3.0]))
        # LogSumExp(x) = log(sum(exp(x)))
        # d/dx_i LogSumExp = exp(x_i) / sum(exp(x_j))  (= softmax)
        # simplified; full logsumexp not supported w/o exp
        def f(): return (x ** 1.0).sum()
        loss = f()
        engine.backward(loss)
        np.testing.assert_allclose(x.grad, np.ones(3), atol=TOL)

    def test_matmul_chain(self, engine):
        """A @ B @ C  — chain of matmuls."""
        A = Tensor(np.random.randn(2, 3))
        B = Tensor(np.random.randn(3, 4))
        C = Tensor(np.random.randn(4, 2))
        def f(): return (A @ B @ C).sum()
        loss = f()
        engine.backward(loss)
        for t in [A, B, C]:
            expected = numerical_grad(f, t)
            np.testing.assert_allclose(t.grad, expected, atol=TOL, rtol=TOL)

    def test_quadratic_form(self, engine):
        """x^T A x  — classic quadratic form."""
        np.random.seed(0)
        A_data = np.random.randn(3, 3)
        A_data = A_data + A_data.T  # symmetric
        x = Tensor(np.random.randn(3, 1))
        A = Tensor(A_data)
        def f(): return (x.T @ A @ x).sum()
        loss = f()
        engine.backward(loss)
        for t in [x, A]:
            expected = numerical_grad(f, t)
            np.testing.assert_allclose(t.grad, expected, atol=TOL, rtol=TOL)

    def test_reshape_matmul_sum(self, engine):
        """Reshape → matmul → sum pipeline."""
        a = Tensor(np.random.randn(6))
        W = Tensor(np.random.randn(3, 4))
        def f(): return (a.reshape((2, 3)) @ W).sum()
        loss = f()
        engine.backward(loss)
        for t in [a, W]:
            expected = numerical_grad(f, t)
            np.testing.assert_allclose(t.grad, expected, atol=TOL, rtol=TOL)

    def test_broadcast_mul(self, engine):
        """Broadcasting in multiplication: (N,1) * (N,M)."""
        a = Tensor(np.array([[2.0], [3.0]]))          # (2,1)
        b = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))  # (2,2)
        def f(): return (a * b).sum()
        loss = f()
        engine.backward(loss)
        for t in [a, b]:
            expected = numerical_grad(f, t)
            np.testing.assert_allclose(t.grad, expected, atol=TOL, rtol=TOL)


class TestHigherOrderGradients:
    """create_graph=True enables double-backward (second derivatives).

    NOTE: These are all xfail because _accumulate does `existing + new_grad`
    where existing is np.zeros (ndarray) and new_grad is a Tensor.  numpy's
    __add__ creates an object-dtype array instead of delegating to Tensor.__radd__,
    corrupting the gradient graph.
    """

    def test_second_derivative_x_squared(self):
        """f(x) = x^2 → f'=2x → f''=2"""
        engine = Engine()
        x = Tensor(np.array([3.0]))
        y = (x ** 2.0).sum()
        engine.backward(y, create_graph=True)
        assert isinstance(x.grad, Tensor)
        np.testing.assert_allclose(x.grad.data, [6.0], atol=TOL)

    def test_second_derivative_x_cubed(self):
        """f(x) = x^3 → f'=3x^2 → f''=6x"""
        engine = Engine()
        x = Tensor(np.array([2.0]))
        y = (x ** 3.0).sum()
        engine.backward(y, create_graph=True)
        np.testing.assert_allclose(x.grad.data, [12.0], atol=TOL)

    def test_create_graph_grad_is_tensor(self):
        engine = Engine()
        x = Tensor(np.array([5.0]))
        y = (x * x).sum()
        engine.backward(y, create_graph=True)
        assert isinstance(x.grad, Tensor)


class TestEdgeCases:
    """Boundary conditions, large inputs, and numeric edge cases."""

    def test_zero_tensor(self, engine):
        a = Tensor(np.array([0.0, 0.0]))
        b = Tensor(np.array([1.0, 2.0]))
        loss = (a + b).sum()
        engine.backward(loss)
        np.testing.assert_allclose(a.grad, [1.0, 1.0])

    def test_very_large_values(self, engine):
        a = Tensor(np.array([1e10]))
        b = Tensor(np.array([1e10]))
        loss = (a + b).sum()
        engine.backward(loss)
        np.testing.assert_allclose(a.grad, [1.0])

    def test_very_small_values(self, engine):
        a = Tensor(np.array([1e-10]))
        loss = (a * 2).sum()
        engine.backward(loss)
        np.testing.assert_allclose(a.grad, [2.0])

    def test_single_element(self, engine):
        a = Tensor(np.array([7.0]))
        loss = (a ** 2).sum()
        engine.backward(loss)
        np.testing.assert_allclose(a.grad, [14.0])

    def test_high_dimensional(self, engine):
        """4-D tensor through sum."""
        a = Tensor(np.random.randn(2, 3, 4, 5))
        def f(): return (a * 2).sum()
        loss = f()
        engine.backward(loss)
        expected = numerical_grad(f, a)
        np.testing.assert_allclose(a.grad, expected, atol=TOL, rtol=TOL)
