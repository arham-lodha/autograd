"""
Investigate whether skip_optimization=True is necessary when compiling gradient graphs.

Each test compiles the same gradient topo twice — once with skip_optimization=True
(current behaviour) and once without — and asserts both produce values that match
finite differences.  If skip_optimization=False ever diverges, it flags a real bug.
"""

import numpy as np
import pytest

from autograd.symbol import Symbol
from autograd.compiler import Compiler
from autograd.Executor import Executor
from autograd.ops import Operation


def _grad_both_ways(graph_fn, *arrays, eps=1e-4, atol=1e-3, rtol=1e-3):
    """
    Compile every gradient graph twice and return three values per input:
      (skipped_result, optimized_result, finite_diff_result)
    """
    compiler = Compiler()
    executor = Executor()
    arrays = [np.asarray(a, dtype=float) for a in arrays]

    syms = [Symbol(operation=Operation.VARIABLE, requires_grad=True) for _ in arrays]
    loss = graph_fn(*syms)
    feed = {s: a.copy() for s, a in zip(syms, arrays)}

    fwd_topo = compiler.compile(loss, backwards=True)
    executor.forward(fwd_topo, feed)

    skipped, optimized = [], []
    for sym in syms:
        assert sym.grad is not None, "No gradient — check requires_grad"

        # Current behaviour
        g_topo_skip = compiler.compile(sym.grad, skip_optimization=True)
        skipped.append(executor.forward(g_topo_skip, feed))

        # Re-run forward so node.value is fresh for the second compile
        executor.forward(fwd_topo, feed)
        g_topo_opt = compiler.compile(sym.grad, skip_optimization=False)
        optimized.append(executor.forward(g_topo_opt, feed))

    # Finite differences
    numerical = []
    for i, arr in enumerate(arrays):
        grad = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index

            def _run(delta, i=i, idx=idx):
                fresh = [Symbol(operation=Operation.VARIABLE, requires_grad=True) for _ in arrays]
                vals = [a.copy() for a in arrays]
                vals[i] = vals[i].copy()
                vals[i][idx] = float(vals[i][idx]) + delta
                t = compiler.compile(graph_fn(*fresh))
                return float(executor.forward(t, {s: v for s, v in zip(fresh, vals)}).sum())

            grad[idx] = (_run(eps) - _run(-eps)) / (2 * eps)
            it.iternext()
        numerical.append(grad)

    return skipped, optimized, numerical


def _check(graph_fn, *arrays, **kw):
    skipped, optimized, numerical = _grad_both_ways(graph_fn, *arrays, **kw)
    for i, (s, o, n) in enumerate(zip(skipped, optimized, numerical)):
        np.testing.assert_allclose(s, n, atol=kw.get("atol", 1e-3), rtol=kw.get("rtol", 1e-3),
                                   err_msg=f"[skip=True  vs FD] input {i}")
        np.testing.assert_allclose(o, n, atol=kw.get("atol", 1e-3), rtol=kw.get("rtol", 1e-3),
                                   err_msg=f"[skip=False vs FD] input {i}")
        np.testing.assert_allclose(s, o, atol=kw.get("atol", 1e-3), rtol=kw.get("rtol", 1e-3),
                                   err_msg=f"[skip=True  vs skip=False] input {i}")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestSkipOptimizationInvestigation:

    def test_add(self, rng):
        a, b = rng.standard_normal((4,)), rng.standard_normal((4,))
        _check(lambda x, y: (x + y).sum(), a, b)

    def test_multiply(self, rng):
        a, b = rng.standard_normal((4,)), rng.standard_normal((4,))
        _check(lambda x, y: (x * y).sum(), a, b)

    def test_pow(self, rng):
        a = rng.standard_normal((4,)) + 2.0
        _check(lambda x: (x ** 3).sum(), a)

    def test_exp(self, rng):
        a = rng.standard_normal((4,)) * 0.5
        _check(lambda x: x.exp().sum(), a)

    def test_log(self, rng):
        a = np.abs(rng.standard_normal((4,))) + 0.5
        _check(lambda x: x.log().sum(), a)

    def test_sqrt(self, rng):
        a = np.abs(rng.standard_normal((4,))) + 0.5
        _check(lambda x: x.sqrt().sum(), a)

    def test_sub(self, rng):
        a, b = rng.standard_normal((4,)), rng.standard_normal((4,))
        _check(lambda x, y: (x - y).sum(), a, b)

    def test_neg(self, rng):
        a = rng.standard_normal((4,))
        _check(lambda x: (-x).sum(), a)

    def test_chain(self, rng):
        # log(exp(x)) simplifies to x — tests algebraic simplification on the gradient graph
        a = rng.standard_normal((4,)) * 0.5
        _check(lambda x: x.log().exp().sum(), a)

    def test_composed(self, rng):
        a = np.abs(rng.standard_normal((4,))) + 0.5
        _check(lambda x: (x.log() * x.exp()).sum(), a)

    def test_matmul(self, rng):
        A = rng.standard_normal((3, 4))
        x = rng.standard_normal((4,))
        _check(lambda a, b: (a @ b).sum(), A, x)

    def test_sum_axis(self, rng):
        a = rng.standard_normal((3, 4))
        _check(lambda x: x.sum(axis=0).sum(), a)

    def test_broadcasting(self, rng):
        a = rng.standard_normal((3, 4))
        b = rng.standard_normal((4,))
        _check(lambda x, y: (x + y).sum(), a, b)
