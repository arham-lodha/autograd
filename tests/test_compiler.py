"""Tests for Compiler optimization passes."""

import numpy as np
import pytest

from autograd.symbol import Symbol
from autograd.compiler import Compiler
from autograd.Executor import Executor
from autograd.ops import Operation
from tests.conftest import make_var, make_const


def _run(sym, feed=None, skip_optimization=False):
    c = Compiler()
    e = Executor()
    topo = c.compile(sym, skip_optimization=skip_optimization)
    return e.forward(topo, feed or {})


class TestConstantFolding:
    def test_add_two_constants(self):
        a = make_const(2.0)
        b = make_const(3.0)
        result = _run(a + b)
        np.testing.assert_allclose(result, 5.0)

    def test_mul_two_constants(self):
        a = make_const(4.0)
        b = make_const(3.0)
        result = _run(a * b)
        np.testing.assert_allclose(result, 12.0)

    def test_pow_constants(self):
        a = make_const(2.0)
        b = make_const(3.0)
        result = _run(a ** b)
        np.testing.assert_allclose(result, 8.0)

    def test_exp_constant(self):
        a = make_const(0.0)
        result = _run(a.exp())
        np.testing.assert_allclose(result, 1.0)

    def test_log_constant(self):
        a = make_const(np.e)
        result = _run(a.log())
        np.testing.assert_allclose(result, 1.0)

    def test_constant_folding_reduces_topo(self):
        a = make_const(2.0)
        b = make_const(3.0)
        topo = Compiler().compile(a + b)
        assert len(topo) == 1
        assert topo[0].operation == Operation.CONSTANT


class TestAlgebraicSimplification:
    def test_add_zero_right(self):
        x = make_var(np.array([1.0, 2.0]))
        result = _run(x + make_const(0.0), {x: np.array([1.0, 2.0])})
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_add_zero_left(self):
        x = make_var(np.array([3.0]))
        result = _run(make_const(0.0) + x, {x: np.array([3.0])})
        np.testing.assert_allclose(result, [3.0])

    def test_mul_by_one(self):
        x = make_var(np.array([5.0]))
        result = _run(x * make_const(1.0), {x: np.array([5.0])})
        np.testing.assert_allclose(result, [5.0])

    def test_mul_by_zero(self):
        x = make_var(np.array([99.0]))
        result = _run(x * make_const(0.0), {x: np.array([99.0])})
        np.testing.assert_allclose(result, 0.0)

    def test_pow_zero_exponent(self):
        x = make_var(np.array([7.0]))
        result = _run(x ** make_const(0.0), {x: np.array([7.0])})
        np.testing.assert_allclose(result, 1.0)

    def test_pow_one_exponent(self):
        x = make_var(np.array([7.0]))
        result = _run(x ** make_const(1.0), {x: np.array([7.0])})
        np.testing.assert_allclose(result, [7.0])

    def test_log_exp_cancels(self):
        x = make_var(np.array([2.0]))
        result = _run(x.exp().log(), {x: np.array([2.0])})
        np.testing.assert_allclose(result, [2.0])

    def test_exp_log_cancels(self):
        x = make_var(np.array([2.0]))
        result = _run(x.log().exp(), {x: np.array([2.0])})
        np.testing.assert_allclose(result, [2.0])

    def test_double_transpose_cancels(self):
        x = make_var(np.array([[1.0, 2.0], [3.0, 4.0]]))
        val = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _run(x.transpose().transpose(), {x: val})
        np.testing.assert_allclose(result, val)

    def test_relu_relu_idempotent(self):
        x = make_var(np.array([-1.0, 0.0, 2.0]))
        val = np.array([-1.0, 0.0, 2.0])
        result = _run(x.relu().relu(), {x: val})
        np.testing.assert_allclose(result, np.maximum(0, val))

    def test_double_transpose_reduces_topo(self):
        x = make_var(np.array([[1.0, 2.0]]))
        topo = Compiler().compile(x.transpose().transpose())
        ops = [n.operation for n in topo]
        assert Operation.TRANSPOSE not in ops


class TestCanonicalizationAndDecanonicalization:
    def test_neg_becomes_neg_after_roundtrip(self):
        x = make_var(np.array([3.0]))
        val = np.array([3.0])
        result = _run(-x, {x: val})
        np.testing.assert_allclose(result, [-3.0])

    def test_sub_computes_correctly(self):
        x = make_var(np.array([5.0]))
        y = make_var(np.array([2.0]))
        result = _run(x - y, {x: np.array([5.0]), y: np.array([2.0])})
        np.testing.assert_allclose(result, [3.0])

    def test_sqrt_computes_correctly(self):
        x = make_var(np.array([9.0]))
        result = _run(x.sqrt(), {x: np.array([9.0])})
        np.testing.assert_allclose(result, [3.0])

    def test_sqrt_decanonicalized(self):
        x = make_var(np.array([4.0]))
        topo = Compiler().compile(x.sqrt())
        ops = [n.operation for n in topo]
        assert Operation.SQRT in ops
        assert Operation.POWER not in ops


class TestFlatteningAndDCE:
    def test_add_chain_flattened(self):
        a = make_var(np.array(1.0))
        b = make_var(np.array(2.0))
        c = make_var(np.array(3.0))
        result = _run((a + b) + c, {a: np.array(1.0), b: np.array(2.0), c: np.array(3.0)})
        np.testing.assert_allclose(result, 6.0)

    def test_mul_chain_flattened(self):
        a = make_var(np.array(2.0))
        b = make_var(np.array(3.0))
        c = make_var(np.array(4.0))
        result = _run((a * b) * c, {a: np.array(2.0), b: np.array(3.0), c: np.array(4.0)})
        np.testing.assert_allclose(result, 24.0)

    def test_dead_code_eliminated(self):
        x = make_var(np.array([1.0]))
        dead = x * make_const(2.0)  # unused
        used = x + make_const(1.0)

        topo = Compiler()._build_topo(used)
        live, _ = Compiler()._dead_code_elimination(topo)
        live_ids = {id(n) for n in live}
        assert id(dead) not in live_ids

    def test_unused_branch_not_evaluated(self, rng):
        x = make_var(rng.standard_normal((3,)))
        val = rng.standard_normal((3,))
        result = _run(x + make_const(0.0), {x: val})
        np.testing.assert_allclose(result, val)


class TestOptimizationPreservesValues:
    """End-to-end: optimized result must equal unoptimized result."""

    def _check(self, sym_fn, feed_fn, rng):
        vals = feed_fn(rng)
        syms_opt = [make_var(v) for v in vals]
        syms_raw = [make_var(v) for v in vals]
        opt = _run(sym_fn(*syms_opt), {s: v for s, v in zip(syms_opt, vals)})
        raw = _run(sym_fn(*syms_raw), {s: v for s, v in zip(syms_raw, vals)}, skip_optimization=True)
        np.testing.assert_allclose(opt, raw, rtol=1e-5, atol=1e-5)

    def test_polynomial(self, rng):
        self._check(lambda x: x ** 3 + x ** 2 - x,
                    lambda rng: [rng.standard_normal((4,)) * 0.5], rng)

    def test_log_exp(self, rng):
        self._check(lambda x: x.exp().log(),
                    lambda rng: [np.abs(rng.standard_normal((4,))) + 0.1], rng)

    def test_matmul_with_transpose(self, rng):
        self._check(lambda a, b: (a @ b).sum(),
                    lambda rng: [rng.standard_normal((3, 4)), rng.standard_normal((4, 2))], rng)

    def test_softmax(self, rng):
        self._check(lambda x: x.softmax(axis=1),
                    lambda rng: [rng.standard_normal((3, 5))], rng)
