"""Tests for Executor forward pass — one test per operation."""

import numpy as np
import pytest

from autograd.symbol import Symbol
from autograd.compiler import Compiler
from autograd.Executor import Executor
from autograd.ops import Operation
from tests.conftest import make_var, make_const


def _fwd(sym, feed=None):
    topo = Compiler().compile(sym, skip_optimization=True)
    return Executor().forward(topo, feed or {})


class TestArithmetic:
    def test_add(self):
        x, y = make_var(None), make_var(None)
        x.operation = Operation.VARIABLE; y.operation = Operation.VARIABLE
        a, b = np.array([1.0, 2.0]), np.array([3.0, 4.0])
        x2 = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        y2 = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        r = _fwd(x2 + y2, {x2: a, y2: b})
        np.testing.assert_allclose(r, [4.0, 6.0])

    def test_multiply(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        y = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a, b = rng.standard_normal((4,)), rng.standard_normal((4,))
        np.testing.assert_allclose(_fwd(x * y, {x: a, y: b}), a * b)

    def test_power(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = np.abs(rng.standard_normal((3,))) + 0.1
        np.testing.assert_allclose(_fwd(x ** make_const(2.0), {x: a}), a ** 2)

    def test_neg(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = rng.standard_normal((5,))
        np.testing.assert_allclose(_fwd(-x, {x: a}), -a)

    def test_sub(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        y = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a, b = rng.standard_normal((3,)), rng.standard_normal((3,))
        np.testing.assert_allclose(_fwd(x - y, {x: a, y: b}), a - b)

    def test_matmul(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        y = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a, b = rng.standard_normal((3, 4)), rng.standard_normal((4, 2))
        np.testing.assert_allclose(_fwd(x @ y, {x: a, y: b}), a @ b)


class TestUnary:
    def test_exp(self):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(_fwd(x.exp(), {x: a}), np.exp(a))

    def test_log(self):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = np.array([1.0, np.e, np.e ** 2])
        np.testing.assert_allclose(_fwd(x.log(), {x: a}), np.log(a))

    def test_sqrt(self):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = np.array([1.0, 4.0, 9.0])
        np.testing.assert_allclose(_fwd(x.sqrt(), {x: a}), np.sqrt(a))

    def test_abs(self):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = np.array([-3.0, 0.0, 2.0])
        np.testing.assert_allclose(_fwd(x.abs(), {x: a}), np.abs(a))

    def test_relu(self):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = np.array([-2.0, 0.0, 3.0])
        np.testing.assert_allclose(_fwd(x.relu(), {x: a}), np.maximum(0, a))

    def test_transpose(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = rng.standard_normal((3, 4))
        np.testing.assert_allclose(_fwd(x.transpose(), {x: a}), a.T)


class TestReductions:
    def test_sum_all(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = rng.standard_normal((3, 4))
        np.testing.assert_allclose(_fwd(x.sum(), {x: a}), np.sum(a))

    def test_sum_axis0(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = rng.standard_normal((3, 4))
        np.testing.assert_allclose(_fwd(x.sum(axis=0), {x: a}), np.sum(a, axis=0))

    def test_sum_keepdims(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = rng.standard_normal((3, 4))
        np.testing.assert_allclose(_fwd(x.sum(axis=1, keepdims=True), {x: a}),
                                   np.sum(a, axis=1, keepdims=True))

    def test_mean_all(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = rng.standard_normal((3, 4))
        np.testing.assert_allclose(_fwd(x.mean(), {x: a}), np.mean(a), atol=1e-10)

    def test_mean_axis(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = rng.standard_normal((4, 5))
        np.testing.assert_allclose(_fwd(x.mean(axis=1), {x: a}), np.mean(a, axis=1), atol=1e-10)

    def test_variance(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = rng.standard_normal((4, 5))
        np.testing.assert_allclose(_fwd(x.variance(), {x: a}), np.var(a), atol=1e-10)


class TestShapeOps:
    def test_reshape(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = rng.standard_normal((6,))
        np.testing.assert_allclose(_fwd(x.reshape((2, 3)), {x: a}), a.reshape(2, 3))

    def test_expand_dims(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = rng.standard_normal((4,))
        np.testing.assert_allclose(_fwd(x.expand_dims(0), {x: a}), np.expand_dims(a, 0))

    def test_squeeze(self):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = np.ones((1, 3, 1))
        np.testing.assert_allclose(_fwd(x.squeeze(), {x: a}), a.squeeze())

    def test_broadcast_to(self):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_allclose(_fwd(x.broadcast_to((3, 4)), {x: a}),
                                   np.broadcast_to(a, (3, 4)))

    def test_swap_axis(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = rng.standard_normal((2, 3, 4))
        np.testing.assert_allclose(_fwd(x.swap_axis(0, 2), {x: a}), np.swapaxes(a, 0, 2))


class TestSoftmax:
    def test_softmax_sums_to_one(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = rng.standard_normal((4, 5))
        result = _fwd(x.softmax(axis=1), {x: a})
        np.testing.assert_allclose(result.sum(axis=1), np.ones(4), atol=1e-6)

    def test_softmax_numerically_stable(self):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = np.array([[1000.0, 1001.0, 1002.0]])
        result = _fwd(x.softmax(axis=1), {x: a})
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result.sum(axis=1), np.ones(1), atol=1e-6)


class TestComparisons:
    def test_greater_than_elementwise(self):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        y = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 2.0, 2.0])
        result = _fwd(x > y, {x: a, y: b})
        np.testing.assert_array_equal(result, [False, False, True])
        assert result.shape == (3,)

    def test_less_than_elementwise(self):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        y = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 2.0, 2.0])
        result = _fwd(x < y, {x: a, y: b})
        np.testing.assert_array_equal(result, [True, False, False])

    def test_greater_than_scalar_broadcast(self):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a = np.array([-1.0, 0.0, 1.0])
        result = _fwd(x > make_const(0.0), {x: a})
        np.testing.assert_array_equal(result, [False, False, True])


class TestCaching:
    def test_variable_not_in_feed_raises(self):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=True)
        topo = Compiler().compile(x + x, skip_optimization=True)
        with pytest.raises(RuntimeError):
            Executor().forward(topo, {})

    def test_placeholder_not_in_feed_raises(self):
        x = Symbol(operation=Operation.PLACEHOLDER)
        topo = Compiler().compile(x, skip_optimization=True)
        with pytest.raises(RuntimeError):
            Executor().forward(topo, {})

    def test_multiple_forward_calls_independent(self, rng):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=False)
        a1 = rng.standard_normal((3,))
        a2 = rng.standard_normal((3,))
        topo = Compiler().compile(x.exp(), skip_optimization=True)
        e = Executor()
        r1 = e.forward(topo, {x: a1})
        r2 = e.forward(topo, {x: a2})
        np.testing.assert_allclose(r1, np.exp(a1))
        np.testing.assert_allclose(r2, np.exp(a2))
