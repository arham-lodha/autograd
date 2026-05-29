"""Shared fixtures and helpers for the symbolic/compiled test suite."""

import numpy as np
import pytest

from autograd.symbol import Symbol
from autograd.compiler import Compiler
from autograd.Executor import Executor
from autograd.ops import Operation


def make_var(value: np.ndarray, requires_grad: bool = True) -> Symbol:
    return Symbol(value=value, operation=Operation.VARIABLE, requires_grad=requires_grad)


def make_const(value) -> Symbol:
    return Symbol(value=value, operation=Operation.CONSTANT, requires_grad=False)


def check_grad(graph_fn, *arrays, eps: float = 1e-4, atol: float = 1e-3, rtol: float = 1e-3):
    """
    Verify analytical symbolic gradients against central finite differences.

    graph_fn(*vars) -> loss Symbol, where loss reduces to a scalar (or we call .sum() internally).
    arrays: one np.ndarray per variable.
    """
    compiler = Compiler()
    executor = Executor()
    arrays = [np.asarray(a, dtype=float) for a in arrays]

    # ---- Analytical backward ----
    syms = [Symbol(operation=Operation.VARIABLE, requires_grad=True) for _ in arrays]
    loss = graph_fn(*syms)
    feed = {s: a.copy() for s, a in zip(syms, arrays)}

    fwd_topo = compiler.compile(loss, backwards=True)
    executor.forward(fwd_topo, feed)

    analytical = []
    for sym in syms:
        if sym.grad is None:
            analytical.append(np.zeros_like(arrays[0]))
        else:
            g_topo = compiler.compile(sym.grad, skip_optimization=True)
            g_val = executor.forward(g_topo, feed)
            analytical.append(g_val)

    # ---- Numerical backward ----
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

    for i, (a, n, arr) in enumerate(zip(analytical, numerical, arrays)):
        np.testing.assert_allclose(
            a, n, atol=atol, rtol=rtol,
            err_msg=f"Gradient mismatch for input {i} shape={arr.shape}",
        )


@pytest.fixture
def compiler():
    return Compiler()


@pytest.fixture
def executor():
    return Executor()


@pytest.fixture
def rng():
    return np.random.default_rng(42)
