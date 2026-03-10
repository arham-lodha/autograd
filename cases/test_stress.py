"""Stress / performance tests for hard computations.

These tests use larger tensors and deeper computation graphs to verify
correctness under load and to surface any performance regressions.
Mark them with @pytest.mark.slow so they can be excluded in quick runs.
"""

import numpy as np
import pytest
import time
from autograd import Tensor, Engine
from tests.conftest import numerical_grad

TOL = 1e-4

slow = pytest.mark.slow


@slow
class TestLargeMatmul:
    def test_large_matmul_forward(self):
        a = Tensor(np.random.randn(256, 512))
        b = Tensor(np.random.randn(512, 256))
        c = a @ b
        expected = a.data @ b.data
        np.testing.assert_allclose(c.data, expected, atol=1e-8)

    def test_large_matmul_backward(self):
        engine = Engine()
        a = Tensor(np.random.randn(128, 256))
        b = Tensor(np.random.randn(256, 64))
        loss = (a @ b).sum()
        engine.backward(loss)
        # Analytical check: d(sum(A@B))/dA = ones @ B^T
        expected_a = np.ones((128, 64)) @ b.data.T
        np.testing.assert_allclose(a.grad, expected_a, atol=1e-8)
        expected_b = a.data.T @ np.ones((128, 64))
        np.testing.assert_allclose(b.grad, expected_b, atol=1e-8)


@slow
class TestDeepGraph:
    def test_deep_add_chain(self):
        """100-deep addition chain — tests topo sort & gradient accumulation."""
        engine = Engine()
        x = Tensor(np.array([1.0]), retain=True)
        result = x
        for _ in range(100):
            result = result + x
        loss = result.sum()
        engine.backward(loss)
        # gradient should be 101 (original + 100 adds)
        np.testing.assert_allclose(x.grad, [101.0], atol=TOL)

    def test_deep_mul_chain(self):
        """Chain of multiplications: x * 1.01 * 1.01 * ... (50 times)."""
        engine = Engine()
        x = Tensor(np.array([1.0]), retain=True)
        result = x
        for _ in range(50):
            result = result * 1.01
        loss = result.sum()
        engine.backward(loss)
        expected_grad = 1.01 ** 50
        np.testing.assert_allclose(x.grad, [expected_grad], rtol=1e-3)

    def test_wide_graph(self):
        """Many independent branches summed: sum_i (w_i * x)."""
        engine = Engine()
        x = Tensor(np.array([2.0]))
        weights = [Tensor(np.random.randn(1)) for _ in range(200)]
        branches = [w * x for w in weights]
        total = branches[0]
        for b in branches[1:]:
            total = total + b
        loss = total.sum()
        engine.backward(loss)
        expected_x_grad = sum(w.data[0] for w in weights)
        np.testing.assert_allclose(x.grad, [expected_x_grad], atol=TOL)


@slow
class TestBatchedOps:
    def test_batched_matmul_gradient(self):
        engine = Engine()
        B, M, K, N = 8, 16, 32, 16
        a = Tensor(np.random.randn(B, M, K))
        b = Tensor(np.random.randn(B, K, N))
        loss = (a @ b).sum()
        engine.backward(loss)
        # Spot-check shapes
        assert a.grad.shape == (B, M, K)
        assert b.grad.shape == (B, K, N)
        # Analytical: d(sum(A@B))/dA_batch = ones @ B^T per batch
        for i in range(B):
            exp_a = np.ones((M, N)) @ b.data[i].T
            np.testing.assert_allclose(a.grad[i], exp_a, atol=1e-8)

    def test_large_element_wise(self):
        engine = Engine()
        size = (64, 64, 64)
        a = Tensor(np.random.randn(*size))
        b = Tensor(np.random.randn(*size))
        loss = (a * b + a ** 2.0).sum()
        engine.backward(loss)
        # Analytical: d/da = b + 2a
        expected = b.data + 2 * a.data
        np.testing.assert_allclose(a.grad, expected, atol=1e-7)


@slow
class TestNeuralNetworkLike:
    """Simulate a small fully-connected network forward + backward."""

    def test_two_layer_mlp(self):
        np.random.seed(0)
        engine = Engine()

        # "Data"
        X = Tensor(np.random.randn(32, 8))
        # Layer 1
        W1 = Tensor(np.random.randn(8, 16) * 0.1)
        b1 = Tensor(np.zeros((1, 16)))
        # Layer 2
        W2 = Tensor(np.random.randn(16, 4) * 0.1)
        b2 = Tensor(np.zeros((1, 4)))

        # Forward (ReLU approximated by x * (x > 0) isn't available,
        # so use x^2 as a simple nonlinearity)
        # Use (1, N) biases so broadcast_to doesn't add dimensions
        h = (X @ W1 + b1.broadcast_to((32, 16))) ** 2.0
        out = h @ W2 + b2.broadcast_to((32, 4))
        loss = out.sum()

        engine.backward(loss)

        # Basic sanity: all grads should be non-zero and finite
        for name, t in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
            assert t.grad.shape == t.shape, f"{name} grad shape mismatch"
            assert np.all(np.isfinite(t.grad)), f"{name} grad has inf/nan"

    def test_three_layer_mlp(self):
        np.random.seed(1)
        engine = Engine()

        X = Tensor(np.random.randn(16, 10))
        W1 = Tensor(np.random.randn(10, 32) * 0.05)
        W2 = Tensor(np.random.randn(32, 32) * 0.05)
        W3 = Tensor(np.random.randn(32, 1) * 0.05)

        h1 = (X @ W1) ** 2.0
        h2 = (h1 @ W2) ** 2.0
        out = h2 @ W3
        loss = out.sum()

        engine.backward(loss)

        for name, t in [("W1", W1), ("W2", W2), ("W3", W3)]:
            assert t.grad.shape == t.shape
            assert np.all(np.isfinite(t.grad)), f"{name} grad not finite"


@slow
class TestPerformance:
    """Rough timing guards — not strict, just to catch major regressions."""

    def test_backward_completes_in_time(self):
        engine = Engine()
        a = Tensor(np.random.randn(512, 512))
        b = Tensor(np.random.randn(512, 512))
        loss = (a @ b).sum()

        t0 = time.perf_counter()
        engine.backward(loss)
        elapsed = time.perf_counter() - t0

        # Should be well under 5 seconds on any modern machine
        assert elapsed < 5.0, f"Backward took {elapsed:.2f}s — too slow"

    def test_topo_sort_large_graph(self):
        engine = Engine()
        x = Tensor(np.array([1.0]), retain=True)
        result = x
        for _ in range(500):
            result = result + x

        t0 = time.perf_counter()
        engine.build_topo(result)
        elapsed = time.perf_counter() - t0

        assert elapsed < 2.0, f"Topo sort took {elapsed:.2f}s — too slow"
