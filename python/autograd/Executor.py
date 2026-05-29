from typing import Dict, List, Optional

import numpy as np

from .symbol import NonSymbolInputs, Symbol
from .ops import Operation
from .compiler import _unbroadcast


class Executor:
    def __init__(self) -> None:
        self._cache: Dict[int, np.ndarray] = {}

    def _get_value(self, node: Symbol) -> np.ndarray:
        if id(node) in self._cache:
            return self._cache[id(node)]

        raise RuntimeError(
            f"Node {node} has no cached value — was forward called?",
        )

    def _eval_node(self, node: Symbol, feed_dict: Dict[int, np.ndarray]) -> np.ndarray:
        if id(node) in feed_dict:
            return feed_dict[id(node)]

        if node.operation == Operation.CONSTANT:
            if node.value is not None:
                return node.value
            else:
                raise RuntimeError(
                    f"Leaf node {node} has no value and not in feed_dict")

        if node.operation == Operation.PLACEHOLDER:
            raise RuntimeError(
                f"Placeholder {node} not found in feed_dict."
            )

        if node.operation == Operation.VARIABLE:
            raise RuntimeError(
                f"Variable {node} not found in feed_dict."
            )

        inputs: list[np.ndarray] = [self._get_value(p) for p in node.prev]
        kwargs = node.kwargs

        match node.operation:
            case Operation.ADD:
                results = inputs[0]
                for val in inputs[1:]:
                    results = results + val

                return results
            case Operation.MULTIPLY:
                results = inputs[0]
                for val in inputs[1:]:
                    results = results * val

                return results
            case Operation.POWER:
                return inputs[0] ** inputs[1]
            case Operation.NEG:
                return -inputs[0]
            case Operation.SUB:
                return inputs[0] - inputs[1]
            case Operation.SQRT:
                return np.sqrt(inputs[0])
            case Operation.ABS:
                return np.abs(inputs[0])
            case Operation.IDENTITY:
                return inputs[0]
            case Operation.EXP:
                return np.exp(inputs[0])
            case Operation.LOG:
                return np.log(inputs[0])
            case Operation.MATMUL:
                a, b = inputs
                if a.ndim == 1 and b.ndim == 1:
                    return np.outer(a, b)
                return a @ b
            case Operation.TRANSPOSE:
                return inputs[0].T
            case Operation.SUM:
                return np.sum(inputs[0], axis=kwargs.get('axis'), keepdims=kwargs.get('keepdims', False))

            case Operation.MEAN:
                return np.mean(inputs[0], axis=kwargs.get('axis'), keepdims=kwargs.get('keepdims', False))

            case Operation.VARIANCE:
                return np.var(inputs[0], axis=kwargs.get('axis'), keepdims=kwargs.get('keepdims', False))
            case Operation.RELU:
                return np.maximum(0, inputs[0])
            case Operation.SOFTMAX:
                x = inputs[0]
                axis = kwargs.get('axis')
                # Numerically stable softmax
                x_shifted = x - np.max(x, axis=axis, keepdims=True)
                exp_x = np.exp(x_shifted)
                return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
            case Operation.RESHAPE:
                return inputs[0].reshape(kwargs['shape'])

            case Operation.BROADCAST_TO:
                return np.broadcast_to(inputs[0], kwargs['shape'])

            case Operation.EXPAND_DIMS:
                return np.expand_dims(inputs[0], axis=kwargs['axis'])

            case Operation.SQUEEZE:
                return np.squeeze(inputs[0], axis=kwargs.get('axis'))

            case Operation.SWAP_AXIS:
                return np.swapaxes(inputs[0], kwargs['axis1'], kwargs['axis2'])
            case Operation.BROADCAST_TO_MATCH:
                return np.broadcast_to(inputs[0], inputs[1].shape)

            case Operation.UNBROADCAST:
                return _unbroadcast(inputs[0], inputs[1].shape)

            case Operation.RESHAPE_LIKE:
                return inputs[0].reshape(inputs[1].shape)

            case Operation.GREATER_THAN:
                return inputs[0] > inputs[1];
            case Operation.LESS_THAN:
                return inputs[0] < inputs[1];
            case Operation.SIZE:
                return np.array(np.size(inputs[0], kwargs.get('axis')))
            case Operation.SIGN:
                return np.sign(inputs[0])
            case Operation.VECTOR:
                return np.stack(inputs);
            case Operation.GET_ITEM:
                index = kwargs.get('index', 0)
                return inputs[0][index]
            case Operation.SCATTER_LIKE:
                
                # Check this TODO
                original = inputs[0]
                ref = inputs[1]
                index = kwargs.get('index', 0);

                value = np.zeros(ref.shape);
                value[index] = original

                return value


            case _:
                raise RuntimeError(f"Unknown operation: {node.operation}")

    def forward(self, topo: List[Symbol], feed_dict: Optional[Dict[Symbol, NonSymbolInputs]] = None) -> np.ndarray:
        if feed_dict is None:
            feed_dict = {}

        feed_id_map: Dict[int, np.ndarray] = {}
        for symbol, value in feed_dict.items():
            if not isinstance(value, np.ndarray):
                value = np.array(value)

            feed_id_map[id(symbol)] = value

        self._cache.clear()

        for node in topo:
            self._cache[id(node)] = self._eval_node(node, feed_id_map)

        return self._cache[id(topo[-1])]
