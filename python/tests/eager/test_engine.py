"""Tests for the Engine: topological sort, caching, zero_grad."""

import numpy as np
import pytest
from autograd.eager import Tensor, Engine


class TestBuildTopo:
    def test_single_node(self, engine):
        a = Tensor(1.0)
        topo = engine.build_topo(a)
        assert len(topo) == 1
        assert topo[0] is a

    def test_linear_chain(self, engine):
        a = Tensor(np.array([2.0]), retain=True)
        b = a * 3
        c = b + 1
        topo = engine.build_topo(c)
        # a should appear before b, b before c
        ids = [id(n) for n in topo]
        assert ids.index(id(a)) < ids.index(id(c))

    def test_diamond(self, engine):
        a = Tensor(np.array([2.0]), retain=True)
        b = a * 2
        c = a * 3
        d = b + c
        topo = engine.build_topo(d)
        ids = [id(n) for n in topo]
        # a appears before both b-related and c-related nodes
        assert ids.index(id(a)) < ids.index(id(d))
        # each node appears exactly once
        assert len(ids) == len(set(ids))


class TestCaching:
    def test_cache_hit(self, engine):
        a = Tensor(np.array([1.0]))
        b = a + 1
        t1 = engine.build_topo(b)
        t2 = engine.build_topo(b)
        assert t1 is t2  # same list object

    def test_cache_different_tensors(self, engine):
        a = Tensor(np.array([1.0]))
        b = a + 1
        c = a + 2
        engine.build_topo(b)
        engine.build_topo(c)
        assert len(engine.cache) == 2


class TestZeroGrad:
    def test_zero_grad(self, engine):
        a = Tensor(np.array([1.0, 2.0]))
        b = Tensor(np.array([3.0, 4.0]))
        c = a + b
        engine.backward(c.sum())
        # grads should be non-zero
        assert np.any(a.grad != 0)
        engine.zero_grad(c.sum())
        # after zero_grad they should all be zero
        np.testing.assert_array_equal(a.grad, np.zeros_like(a.data))
        np.testing.assert_array_equal(b.grad, np.zeros_like(b.data))


class TestBackwardAPI:
    def test_backward_sets_grads(self, engine):
        a = Tensor(np.array([3.0]))
        b = a * a  # b = a^2, db/da = 2a = 6
        loss = b.sum()
        engine.backward(loss)
        np.testing.assert_allclose(a.grad, [6.0], atol=1e-7)

    def test_backward_shape_mismatch_raises(self):
        t = Tensor(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="shape"):
            t.backward(np.array([1.0, 2.0, 3.0]))
