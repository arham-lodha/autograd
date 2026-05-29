"""Symbolic backward pass — every op's gradient checked against numerical finite differences."""

import numpy as np
import pytest

from tests.conftest import check_grad

class TestElementWiseBackward:
    def test_add(self, rng):
        a, b = rng.standard_normal((4,)), rng.standard_normal((4,))
        check_grad(lambda x, y: (x + y).sum(), a, b)

    def test_add_scalar(self, rng):
        a = rng.standard_normal((4,))
        check_grad(lambda x: (x + 3.0).sum(), a)

    def test_multiply(self, rng):
        a, b = rng.standard_normal((4,)), rng.standard_normal((4,))
        check_grad(lambda x, y: (x * y).sum(), a, b)

    def test_multiply_scalar(self, rng):
        a = rng.standard_normal((4,))
        check_grad(lambda x: (x * 2.0).sum(), a)

    def test_neg(self, rng):
        a = rng.standard_normal((4,))
        check_grad(lambda x: (-x).sum(), a)

    def test_sub(self, rng):
        a, b = rng.standard_normal((4,)), rng.standard_normal((4,))
        check_grad(lambda x, y: (x - y).sum(), a, b)

    def test_rsub(self, rng):
        a = rng.standard_normal((4,))
        check_grad(lambda x: (3.0 - x).sum(), a)

    def test_pow_scalar_exponent(self, rng):
        a = rng.standard_normal((4,)) + 2.0
        check_grad(lambda x: (x ** 2).sum(), a)

    def test_pow_fractional_exponent(self, rng):
        a = np.abs(rng.standard_normal((4,))) + 0.5
        check_grad(lambda x: (x ** 0.5).sum(), a)

    def test_exp(self, rng):
        a = rng.standard_normal((4,)) * 0.5
        check_grad(lambda x: x.exp().sum(), a)

    def test_log(self, rng):
        a = np.abs(rng.standard_normal((4,))) + 0.5
        check_grad(lambda x: x.log().sum(), a)

    def test_sqrt(self, rng):
        a = np.abs(rng.standard_normal((4,))) + 0.5
        check_grad(lambda x: x.sqrt().sum(), a)

    def test_relu(self, rng):
        a = rng.standard_normal((8,))
        check_grad(lambda x: x.relu().sum(), a)


class TestMatmulBackward:
    def test_matmul_square(self, rng):
        a = rng.standard_normal((3, 3))
        b = rng.standard_normal((3, 3))
        check_grad(lambda x, y: (x @ y).sum(), a, b)

    def test_matmul_nonsquare(self, rng):
        a = rng.standard_normal((3, 4))
        b = rng.standard_normal((4, 2))
        check_grad(lambda x, y: (x @ y).sum(), a, b)

    def test_matmul_grad_a_only(self, rng):
        a = rng.standard_normal((2, 3))
        b = rng.standard_normal((3, 4))
        check_grad(lambda x, y: (x @ y).sum(), a, b)


class TestTransposeBackward:
    def test_transpose(self, rng):
        a = rng.standard_normal((3, 4))
        check_grad(lambda x: x.transpose().sum(), a)


class TestReductionBackward:
    def test_sum_all(self, rng):
        a = rng.standard_normal((3, 4))
        check_grad(lambda x: x.sum(), a)

    def test_sum_axis0(self, rng):
        a = rng.standard_normal((3, 4))
        check_grad(lambda x: x.sum(axis=0).sum(), a)

    def test_sum_axis1_keepdims(self, rng):
        a = rng.standard_normal((3, 4))
        check_grad(lambda x: x.sum(axis=1, keepdims=True).sum(), a)

    def test_mean_all(self, rng):
        a = rng.standard_normal((3, 4))
        check_grad(lambda x: x.mean(), a)

    def test_mean_axis(self, rng):
        a = rng.standard_normal((4, 5))
        check_grad(lambda x: x.mean(axis=1).sum(), a)

    def test_variance(self, rng):
        a = rng.standard_normal((4, 5))
        check_grad(lambda x: x.variance(), a, atol=1e-3, rtol=1e-3)


class TestShapeBackward:
    def test_reshape(self, rng):
        a = rng.standard_normal((6,))
        check_grad(lambda x: x.reshape((2, 3)).sum(), a)

    def test_expand_dims(self, rng):
        a = rng.standard_normal((4,))
        check_grad(lambda x: x.expand_dims(0).sum(), a)

    def test_squeeze(self, rng):
        a = rng.standard_normal((1, 4, 1))
        check_grad(lambda x: x.squeeze().sum(), a)

    def test_swap_axis(self, rng):
        a = rng.standard_normal((2, 3, 4))
        check_grad(lambda x: x.swap_axis(0, 2).sum(), a)


class TestSoftmaxBackward:
    def test_softmax_axis1(self, rng):
        a = rng.standard_normal((3, 5))
        check_grad(lambda x: x.softmax(axis=1).sum(), a)

    def test_softmax_axis0(self, rng):
        a = rng.standard_normal((4, 3))
        check_grad(lambda x: x.softmax(axis=0).sum(), a)


class TestCompositeBackward:
    def test_polynomial(self, rng):
        a = rng.standard_normal((4,)) * 0.5
        check_grad(lambda x: (x ** 3 + 2.0 * x ** 2 - x).sum(), a)

    def test_log_of_exp(self, rng):
        a = rng.standard_normal((4,)) * 0.5
        check_grad(lambda x: x.exp().log().sum(), a)

    def test_chain_matmul_sum(self, rng):
        a = rng.standard_normal((3, 4))
        b = rng.standard_normal((4, 2))
        check_grad(lambda x, y: (x @ y).sum(), a, b)

    def test_quadratic_form(self, rng):
        a = rng.standard_normal((3, 3))
        v = rng.standard_normal((3, 1))
        check_grad(lambda x, w: (w.transpose() @ x @ w).sum(), a, v)

    def test_mlp_one_layer(self, rng):
        W = rng.standard_normal((4, 3)) * 0.1
        x_val = rng.standard_normal((3,))
        check_grad(lambda w, x: (w @ x).relu().sum(), W, x_val)

    def test_mean_subtraction(self, rng):
        a = rng.standard_normal((4, 5))
        check_grad(lambda x: (x - x.mean(axis=1, keepdims=True)).sum(), a)
