"""Tests for Symbol graph construction."""

import numpy as np
import pytest

from autograd.symbol import Symbol
from autograd.ops import Operation


class TestSymbolConstruction:
    def test_constant_scalar(self):
        s = Symbol(value=3.0)
        assert s.operation == Operation.CONSTANT
        np.testing.assert_array_equal(s.value, np.array(3.0))
        assert s.requires_grad is False

    def test_constant_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        s = Symbol(value=arr)
        assert s.value is not None
        assert s.value.shape == (3,)

    def test_variable_requires_grad(self):
        s = Symbol(value=np.array([1.0]), operation=Operation.VARIABLE)
        assert s.requires_grad is True

    def test_constant_no_grad(self):
        s = Symbol(value=np.array([1.0]), operation=Operation.CONSTANT)
        assert s.requires_grad is False

    def test_explicit_requires_grad_override(self):
        s = Symbol(value=np.array([1.0]), operation=Operation.CONSTANT, requires_grad=True)
        assert s.requires_grad is True

    def test_value_normalised_to_ndarray(self):
        s = Symbol(value=[1, 2, 3])
        assert isinstance(s.value, np.ndarray)

    def test_no_value_is_none(self):
        s = Symbol(operation=Operation.VARIABLE)
        assert s.value is None

    def test_retain_flag(self):
        s = Symbol(value=np.array(1.0), retain=True)
        assert s.retain is True

    def test_name_stored(self):
        s = Symbol(value=np.array(1.0), name="x")
        assert s.name == "x"

    def test_kwargs_stored(self):
        s = Symbol(operation=Operation.SUM, prev=[], axis=1, keepdims=True)
        assert s.kwargs["axis"] == 1
        assert s.kwargs["keepdims"] is True


class TestRequiresGradInference:
    def test_op_of_variable_requires_grad(self):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=True)
        y = x + x
        assert y.requires_grad is True

    def test_op_of_constants_no_grad(self):
        a = Symbol(value=np.array(2.0))
        b = Symbol(value=np.array(3.0))
        c = a + b
        assert c.requires_grad is False

    def test_mixed_grad_propagates(self):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=True)
        c = Symbol(value=np.array(5.0))
        y = x * c
        assert y.requires_grad is True

    def test_grad_attribute_initialised_none(self):
        x = Symbol(operation=Operation.VARIABLE, requires_grad=True)
        assert x.grad is None


class TestOperatorOverloads:
    def setup_method(self):
        self.x = Symbol(operation=Operation.VARIABLE, requires_grad=True)
        self.y = Symbol(operation=Operation.VARIABLE, requires_grad=True)

    def test_add(self):
        z = self.x + self.y
        assert z.operation == Operation.ADD
        assert self.x in z.prev and self.y in z.prev

    def test_radd(self):
        z = 2.0 + self.x
        assert z.operation == Operation.ADD

    def test_mul(self):
        z = self.x * self.y
        assert z.operation == Operation.MULTIPLY

    def test_rmul(self):
        z = 3.0 * self.x
        assert z.operation == Operation.MULTIPLY

    def test_pow(self):
        z = self.x ** 2
        assert z.operation == Operation.POWER

    def test_neg(self):
        z = -self.x
        assert z.operation == Operation.NEG

    def test_sub(self):
        z = self.x - self.y
        assert z.operation == Operation.SUB

    def test_rsub(self):
        z = 1.0 - self.x
        assert z.operation == Operation.SUB

    def test_truediv(self):
        z = self.x / self.y
        assert z.operation == Operation.MULTIPLY

    def test_matmul(self):
        z = self.x @ self.y
        assert z.operation == Operation.MATMUL

    def test_exp(self):
        z = self.x.exp()
        assert z.operation == Operation.EXP

    def test_log(self):
        z = self.x.log()
        assert z.operation == Operation.LOG

    def test_relu(self):
        z = self.x.relu()
        assert z.operation == Operation.RELU

    def test_sum(self):
        z = self.x.sum(axis=0)
        assert z.operation == Operation.SUM
        assert z.kwargs["axis"] == 0

    def test_mean(self):
        z = self.x.mean(axis=1, keepdims=True)
        assert z.operation == Operation.MEAN
        assert z.kwargs["keepdims"] is True

    def test_transpose(self):
        z = self.x.transpose()
        assert z.operation == Operation.TRANSPOSE

    def test_reshape(self):
        z = self.x.reshape((3, 4))
        assert z.operation == Operation.RESHAPE
        assert z.kwargs["shape"] == (3, 4)

    def test_softmax(self):
        z = self.x.softmax(axis=1)
        assert z.operation == Operation.SOFTMAX

    def test_chained_ops_build_correct_dag(self):
        z = (self.x * self.y).sum()
        assert z.operation == Operation.SUM
        assert z.prev[0].operation == Operation.MULTIPLY

    def test_gt(self):
        z = self.x > self.y
        assert z.operation == Operation.GREATER_THAN

    def test_lt(self):
        z = self.x < self.y
        assert z.operation == Operation.LESS_THAN
