from typing import Any,  List, Optional, Union

import numpy as np

from .ops import Operation

type Inputs = Union[float, int, np.ndarray, list, 'Symbol']
type NonSymbolInputs = Union[float, int, np.ndarray, list]


class Symbol:
    def __init__(
        self,
        value: Optional[NonSymbolInputs] = None,
        operation: Operation = Operation.CONSTANT,
        prev: Optional[List['Symbol']] = None,
        name: Optional[str] = None,
        retain: bool = False,
        requires_grad: Optional[bool] = None,
        **kwargs: Any
    ):
        self.operation = operation
        self.prev = prev if prev is not None else []
        self.name = name
        self.retain = retain
        self.grad: Optional['Symbol'] = None
        self.kwargs = kwargs

        if value is not None:
            self.value = value if isinstance(
                value, np.ndarray) else np.array(value)
        else:
            self.value = None
        
        if self.prev:
            self.requires_grad = any(
                prev_symbol.requires_grad for prev_symbol in self.prev)

            if requires_grad is not None:
                self.requires_grad = requires_grad
        else:
            if requires_grad is not None:
                self.requires_grad = requires_grad
            else:
                self.requires_grad = (operation == Operation.VARIABLE)

    def __repr__(self) -> str:
        name_str = f"{self.name} " if self.name else "Unnamed"
        return f"Symbol(name={name_str}, op={self.operation.name}, requires_grad={self.requires_grad})"

    def _ensure_symbol(self, other: Inputs) -> 'Symbol':
        if not isinstance(other, Symbol):
            return Symbol(value=other, requires_grad=False)
        return other

    def __add__(self, other: Inputs) -> 'Symbol':
        return Symbol(operation=Operation.ADD, prev=[self, self._ensure_symbol(other)])

    def __radd__(self, other: Inputs) -> 'Symbol':
        return Symbol(operation=Operation.ADD, prev=[self._ensure_symbol(other), self])

    def __mul__(self, other: Inputs) -> 'Symbol':
        return Symbol(operation=Operation.MULTIPLY, prev=[self, self._ensure_symbol(other)])

    def __rmul__(self, other: Inputs) -> 'Symbol':
        return Symbol(operation=Operation.MULTIPLY, prev=[self._ensure_symbol(other), self])

    def __pow__(self, other: Inputs) -> 'Symbol':

        return Symbol(operation=Operation.POWER, prev=[self, self._ensure_symbol(other)])

    def __matmul__(self, other: Inputs) -> 'Symbol':
        return Symbol(operation=Operation.MATMUL, prev=[self, self._ensure_symbol(other)])

    def __rmatmul__(self, other: Inputs) -> 'Symbol':
        return Symbol(operation=Operation.MATMUL, prev=[self._ensure_symbol(other), self])

    def __neg__(self) -> 'Symbol':
        return Symbol(operation=Operation.NEG, prev=[self])

    def __sub__(self, other: Inputs) -> 'Symbol':
        return Symbol(operation=Operation.SUB, prev=[self, self._ensure_symbol(other)])

    def __rsub__(self, other: Inputs) -> 'Symbol':
        return Symbol(operation=Operation.SUB, prev=[self._ensure_symbol(other), self])

    def __truediv__(self, other: Inputs) -> 'Symbol':

        power_node = Symbol(operation=Operation.POWER, prev=[
                            self._ensure_symbol(other), Symbol(value=-1.0)])

        return Symbol(operation=Operation.MULTIPLY, prev=[self, power_node])

    def __rtruediv__(self, other: Inputs) -> 'Symbol':
        power_node = Symbol(operation=Operation.POWER, prev=[
            self, Symbol(value=-1.0)])

        return Symbol(operation=Operation.MULTIPLY, prev=[self._ensure_symbol(other), power_node])

    def exp(self) -> 'Symbol':
        return Symbol(operation=Operation.EXP, prev=[self])

    def log(self) -> 'Symbol':
        return Symbol(operation=Operation.LOG, prev=[self])

    def relu(self) -> 'Symbol':
        return Symbol(operation=Operation.RELU, prev=[self])

    def transpose(self) -> 'Symbol':
        return Symbol(operation=Operation.TRANSPOSE, prev=[self])

    def sum(self, axis: Optional[Union[int, tuple]] = None, keepdims: bool = False) -> 'Symbol':
        return Symbol(operation=Operation.SUM, prev=[self], axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[Union[int, tuple]] = None, keepdims: bool = False) -> 'Symbol':
        return Symbol(operation=Operation.MEAN, prev=[self], axis=axis, keepdims=keepdims)

    def variance(self, axis: Optional[Union[int, tuple]] = None, keepdims: bool = False) -> 'Symbol':
        return Symbol(operation=Operation.VARIANCE, prev=[self], axis=axis, keepdims=keepdims)

    def softmax(self, axis: Optional[int] = None) -> 'Symbol':
        return Symbol(operation=Operation.SOFTMAX, prev=[self], axis=axis)

    def broadcast_to(self, shape: tuple) -> 'Symbol':
        return Symbol(operation=Operation.BROADCAST_TO, prev=[self], shape=shape)

    def reshape(self, shape: tuple) -> 'Symbol':
        return Symbol(operation=Operation.RESHAPE, prev=[self], shape=shape)

    def expand_dims(self, axis) -> 'Symbol':
        return Symbol(operation=Operation.EXPAND_DIMS, prev=[self], axis=axis)

    def squeeze(self, axis: Optional[int] = None) -> 'Symbol':
        return Symbol(operation=Operation.SQUEEZE, prev=[self], axis=axis)

    def swap_axis(self, axis1: int, axis2: int) -> 'Symbol':
        return Symbol(operation=Operation.SWAP_AXIS, prev=[self], axis1=axis1, axis2=axis2)

    def sqrt(self) -> 'Symbol':
        return Symbol(operation=Operation.SQRT, prev=[self])

    def abs(self) -> 'Symbol':
        return Symbol(operation=Operation.ABS, prev=[self])

    def broadcast_to_match(self, other: Inputs) -> 'Symbol':
        return Symbol(operation=Operation.BROADCAST_TO_MATCH, prev=[self, self._ensure_symbol(other)])

    def unbroadcast(self, other: Inputs) -> 'Symbol':
        return Symbol(operation=Operation.UNBROADCAST, prev=[self, self._ensure_symbol(other)])

    def reshape_like(self, other: Inputs) -> 'Symbol':
        return Symbol(operation=Operation.RESHAPE_LIKE, prev=[self, self._ensure_symbol(other)])

    def size(self, axis=None) -> 'Symbol':
        return Symbol(operation=Operation.SIZE, prev=[self], axis=axis)

    def __gt__(self, other: Inputs) -> 'Symbol':
        return Symbol(operation=Operation.GREATER_THAN, prev=[self, self._ensure_symbol(other)])

    def __lt__(self, other: Inputs) -> 'Symbol':
        return Symbol(operation=Operation.LESS_THAN, prev=[self, self._ensure_symbol(other)])

    def sign(self) -> 'Symbol':
        return Symbol(operation=Operation.SIGN, prev=[self]);

    def get_item(self, index: int) -> 'Symbol':
        return Symbol(operation=Operation.GET_ITEM, prev=[self], index=index);

    def scatter_like(self, other: Inputs, index: int) -> 'Symbol':
        return Symbol(operation=Operation.SCATTER_LIKE, prev=[self, self._ensure_symbol(other)], index=index)
