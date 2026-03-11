from typing import Callable, List, Optional, Union

import numpy as np

from .ops import Operation


class Tensor:
    def __init__(
        self,
        data: Union[float, np.ndarray, List[float]] = np.ones(1),
        operation: Operation = Operation.CONSTANT,
        prev: Optional[List["Tensor"]] = None,
        scalar: Union[float, np.ndarray] = 1.0,
        pow_scalar: Union[float, np.ndarray] = 1.0,
        retain=False,
    ):
        self.data: np.ndarray = data if isinstance(
            data, np.ndarray) else (np.array(data) if isinstance(data, list) else np.array([data]))

        self.operation: Operation = operation

        self.prev: List["Tensor"] = prev if prev is not None else []

        self.grad: Union["Tensor", np.ndarray] = np.zeros_like(
            self.data)  # Initialize grad to zero

        self._scalar: np.ndarray = (scalar if isinstance(
            scalar, np.ndarray) else np.ones_like(self.data) * scalar)

        self.pow_scalar: np.ndarray = (
            pow_scalar if isinstance(pow_scalar, np.ndarray) else np.ones_like(
                self.data) * pow_scalar
        )

        self.retain: bool = retain
        # (axis, keepdims) or None, used for sum backward
        self.sum_metadata: Optional[tuple[Optional[int], bool]] = None
        self.swap_axis_metadata: Optional[tuple[int, int]] = None
        self.softmax_data: Optional[tuple[int, np.ndarray]] = None

        # type: Optional[Callable[[Union['Tensor', np.ndarray']], Union['Tensor', np.ndarray]]]
        self.custom_grad_func = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        return self.transpose()

    def __add__(self, other: Union["Tensor", np.ndarray, float, int]):
        new_prev = []

        if not isinstance(other, Tensor):

            if self.operation == Operation.ADD and not self.retain:
                new_prev.extend(self.prev)
            else:
                new_prev.append(self)

            return Tensor(self.data + other, operation=Operation.ADD, prev=new_prev)

        for operand in [self, other]:
            if operand.operation == Operation.ADD and not operand.retain:
                new_prev.extend(operand.prev)
            else:
                new_prev.append(operand)

        out = Tensor(self.data + other.data,
                     operation=Operation.ADD, prev=new_prev)

        return out

    def __mul__(self, other: Union["Tensor", np.ndarray, float, int]):
        new_prev = []

        if not isinstance(other, Tensor):
            new_scalar = other

            if self.operation == Operation.MULTIPLY and not self.retain:
                new_prev.extend(self.prev)
                new_scalar *= self._scalar
            else:
                new_prev.append(self)

            return Tensor(
                self.data * other,
                operation=Operation.MULTIPLY,
                prev=new_prev,
                scalar=new_scalar,
            )

        new_prev = []
        new_scalar = np.ones_like(self.data)
        for operand in [self, other]:
            if operand.operation == Operation.MULTIPLY and not operand.retain:
                new_prev.extend(operand.prev)
                new_scalar *= operand._scalar
            else:
                new_prev.append(operand)

        out = Tensor(
            self.data * other.data,
            operation=Operation.MULTIPLY,
            prev=new_prev,
            scalar=new_scalar,
        )
        return out

    def __pow__(self, other: Union["Tensor", np.ndarray, float, int]):

        if not isinstance(other, Tensor):
            return Tensor(
                np.power(self.data, other),
                operation=Operation.POWER,
                prev=[self],
                pow_scalar=other,
            )

        out = Tensor(
            np.power(self.data, other.data), operation=Operation.POWER, prev=[self, other]
        )
        return out

    def __truediv__(self, other: Union["Tensor", np.ndarray, float, int]):
        return self * (other**-1)  # Tensor: reuse pow and mul

    def __rtruediv__(self, other: Union["Tensor", np.ndarray, float, int]):
        return self**-1 * other  # other / self

    def __sub__(self, other: Union["Tensor", np.ndarray, float, int]):
        return self + (-other)  # Tensor: reuse add and neg

    def __rsub__(self, other: Union["Tensor", np.ndarray, float, int]):
        return (-self) + other

    def __rmul__(self, other: Union["Tensor", np.ndarray, float, int]):
        return self * other

    def __radd__(self, other: Union["Tensor", np.ndarray, float, int]):
        return self + other

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad}, op={self.operation.name})"

    def __neg__(self):
        return self * -1

    # self @ other
    def __matmul__(self, other: Union["Tensor", np.ndarray]):
        if not isinstance(other, Tensor):
            return Tensor(self.data @ other,
                          operation=Operation.MATMUL, prev=[self], scalar=other)

        return Tensor(self.data @ other.data,
                      operation=Operation.MATMUL, prev=[self, other])

    # other @ self
    def __rmatmul__(self, other: Union["Tensor", np.ndarray]):
        if not isinstance(other, Tensor):
            return Tensor(other @ self.data,
                          operation=Operation.RMATMUL, prev=[self], scalar=other)

        return Tensor(other.data @ self.data,
                      operation=Operation.RMATMUL, prev=[self, other])

    def transpose(self):
        return Tensor(self.data.T, operation=Operation.TRANSPOSE, prev=[self])

    def exp(self):
        return Tensor(np.exp(self.data), operation=Operation.EXP, prev=[self])

    def log(self):
        if np.any(self.data <= 0):
            raise ValueError("Cannot take log of non-positive values")

        return Tensor(np.log(self.data), operation=Operation.LOG, prev=[self])

    def relu(self):
        return Tensor(np.maximum(0, self.data), operation=Operation.RELU, prev=[self])

    def softmax(self, axis: int = -1):
        exp_data = np.exp(
            self.data - np.max(self.data, axis=axis, keepdims=True))
        softmax_data = exp_data / np.sum(exp_data, axis=axis, keepdims=True)

        out = Tensor(softmax_data, operation=Operation.SOFTMAX, prev=[self])
        out.softmax_data = (axis, softmax_data)
        return out

    def sum(self, axis: Optional[int] = None, keepdims: bool = False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     operation=Operation.SUM, prev=[self])
        out.sum_metadata = (axis, keepdims)
        return out

    def mean(self, axis: Optional[int] = None, keepdims: bool = False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims),
                     operation=Operation.MEAN, prev=[self])
        out.sum_metadata = (axis, keepdims)
        return out

    def variance(self, axis: Optional[int] = None, keepdims: bool = False):
        mean = self.data.mean(axis=axis, keepdims=True)
        out = Tensor(((self.data - mean) ** 2).mean(axis=axis, keepdims=keepdims),
                     operation=Operation.VARIANCE, prev=[self])
        out.sum_metadata = (axis, keepdims)
        return out

    def custom_op(self, func, grad_func):
        out = Tensor(func(self.data), operation=Operation.CUSTOM, prev=[self])
        out.custom_grad_func = grad_func
        return out

    def broadcast_to(self, shape: tuple[int, ...]):
        out = Tensor(np.broadcast_to(self.data, shape).copy(),
                     operation=Operation.BROADCAST_TO, prev=[self])
        return out

    def reshape(self, shape: tuple[int, ...]):
        out = Tensor(self.data.reshape(shape),
                     operation=Operation.RESHAPE, prev=[self])

        return out

    def expand_dims(self, axis: int):
        out = Tensor(np.expand_dims(self.data, axis=axis),
                     operation=Operation.EXPAND_DIMS, prev=[self])
        return out

    def squeeze(self, axis: Optional[int] = None):
        out = Tensor(np.squeeze(self.data, axis=axis),
                     operation=Operation.SQUEEZE, prev=[self])
        # We can reuse expand_dims_metadata for squeeze since they are inverses
        return out

    def swap_axis(self, axis1: int, axis2: int):
        out = Tensor(np.swapaxes(self.data, axis1, axis2),
                     operation=Operation.SWAP_AXIS, prev=[self])
        out.swap_axis_metadata = (axis1, axis2)
        return out

    def _accumulate(self, existing: Union['Tensor', np.ndarray], new_grad: Union['Tensor', np.ndarray], create_graph: bool):

        if create_graph:
            if not isinstance(existing, Tensor):
                return new_grad

        return existing + new_grad

    def _grad_mul_scalar(self, grad: Union['Tensor', np.ndarray], create_graph: bool):
        if create_graph:
            return grad * self._scalar

        return self._scalar * grad

    def _grad_mul_tensors(
        self, grad: Union['Tensor', np.ndarray], others: List["Tensor"], create_graph: bool
    ):
        if create_graph:
            result = grad * self._scalar
            for t in others:
                result = result * t
            return result

        result = grad * self._scalar
        for t in others:
            result = result * t.data
        return result

    def _grad_power_scalar(self, grad: Union['Tensor', np.ndarray], base: "Tensor", create_graph: bool):
        if create_graph:
            return grad * (base ** (self.pow_scalar - 1.0)) * self.pow_scalar

        return (
            grad
            * (base.data ** (self.pow_scalar - 1.0))
            * self.pow_scalar
        )

    def _unbroadcast(self, grad: Union['Tensor', np.ndarray], target_shape: tuple[int, ...]):

        while len(grad.shape) > len(target_shape):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad

    def backward(self, grad: Union["Tensor", np.ndarray], create_graph=False):

        if create_graph:
            if not isinstance(grad, Tensor):
                raise ValueError(
                    "grad must be a Tensor when create_graph=True")
        else:
            if isinstance(grad, Tensor):
                grad = grad.data  # normalize

        # validate shape regardless of create_graph
        grad_shape = grad.shape  # works for both Tensor and ndarray since both have .shape
        if grad_shape != self.data.shape:
            raise ValueError(
                f"grad shape {grad_shape} does not match data shape {self.data.shape}"
            )

        match self.operation:
            case Operation.CONSTANT:
                pass
            case Operation.ADD:
                for t in self.prev:
                    t.grad = self._accumulate(
                        t.grad, self._unbroadcast(grad, t.shape), create_graph)

            case Operation.MULTIPLY:
                if len(self.prev) == 1:
                    self.prev[0].grad = self._accumulate(
                        self.prev[0].grad,
                        self._unbroadcast(
                            self._grad_mul_scalar(grad, create_graph),
                            self.prev[0].shape,
                        ),
                        create_graph,
                    )
                else:
                    for i, t in enumerate(self.prev):
                        others = [p for j, p in enumerate(self.prev) if j != i]

                        t.grad = self._accumulate(
                            t.grad,
                            self._unbroadcast(self._grad_mul_tensors(
                                grad, others, create_graph), t.shape),
                            create_graph,
                        )

            case Operation.POWER:
                if len(self.prev) > 1:
                    a, b = self.prev
                    self.prev[0].grad = self._accumulate(
                        self.prev[0].grad,
                        (
                            grad * b * (a ** (b - 1.0))
                            if create_graph
                            else grad
                            * b.data
                            * (a.data ** (b.data - 1.0))
                        ),
                        create_graph,
                    )

                    self.prev[1].grad = self._accumulate(
                        self.prev[1].grad,
                        (
                            grad * a.log() * (a**b)
                            if create_graph
                            else grad
                            * np.log(a.data)
                            * (a.data**b.data)
                        ),
                        create_graph,
                    )
                else:
                    self.prev[0].grad = self._accumulate(
                        self.prev[0].grad,
                        self._grad_power_scalar(
                            grad, self.prev[0], create_graph),
                        create_graph,
                    )

            case Operation.LOG:
                self.prev[0].grad = self._accumulate(
                    self.prev[0].grad,
                    (
                        grad / self.prev[0]
                        if create_graph
                        else grad / self.prev[0].data
                    ),
                    create_graph,
                )
            case Operation.TRANSPOSE:
                self.prev[0].grad = self._accumulate(
                    self.prev[0].grad,
                    grad.transpose() if create_graph else grad.T,
                    create_graph,
                )
            case Operation.MATMUL:
                # self @ other
                if len(self.prev) != 2:

                    a = self.prev[0]
                    grad_a = (grad @ np.swapaxes(self._scalar, -1, -2)
                              )

                    self.prev[0].grad = self._accumulate(
                        self.prev[0].grad,
                        self._unbroadcast(grad_a, a.shape),
                        create_graph,
                    )
                else:
                    a, b = self.prev

                    grad_a = (grad @ b.swap_axis(-1, -2)
                              ) if create_graph else (grad @ np.swapaxes(b.data, -1, -2))

                    grad_b = (a.swap_axis(-1, -2) @
                              grad) if create_graph else (np.swapaxes(a.data, -1, -2) @ grad)

                    self.prev[0].grad = self._accumulate(
                        self.prev[0].grad,
                        self._unbroadcast(grad_a, a.shape),
                        create_graph,
                    )
                    self.prev[1].grad = self._accumulate(
                        self.prev[1].grad,
                        self._unbroadcast(grad_b, b.shape),
                        create_graph,
                    )
            case Operation.RMATMUL:
                # other @ self
                if len(self.prev) != 2:
                    self.prev[0].grad = self._accumulate(
                        self.prev[0].grad,
                        np.swapaxes(self._scalar, -1, -2) @
                        grad,
                        create_graph,
                    )
                else:
                    a, b = self.prev

                    grad_a = (
                        b.swap_axis(-1, -2) @ grad) if create_graph else (np.swapaxes(b.data, -1, -2) @ grad)
                    grad_b = (grad @ a.swap_axis(-1, -2)
                              ) if create_graph else (grad @ np.swapaxes(a.data, -1, -2))

                    self.prev[0].grad = self._accumulate(
                        self.prev[0].grad,
                        self._unbroadcast(grad_a, a.shape),
                        create_graph,
                    )
                    self.prev[1].grad = self._accumulate(
                        self.prev[1].grad,
                        self._unbroadcast(grad_b, b.shape),
                        create_graph,
                    )
            case Operation.SUM:
                axis, keepdims = self.sum_metadata if self.sum_metadata is not None else (
                    None, False)

                input_shape = self.prev[0].shape

                if axis is not None and keepdims is False:
                    if isinstance(grad, Tensor):
                        grad = grad.expand_dims(axis)
                    else:
                        grad = np.expand_dims(grad, axis)

                if isinstance(grad, Tensor):
                    grad = grad.broadcast_to(input_shape)
                else:
                    grad = np.broadcast_to(grad, input_shape)

                self.prev[0].grad = self._accumulate(
                    self.prev[0].grad, grad, create_graph)

            case Operation.MEAN:
                axis, keepdims = self.sum_metadata if self.sum_metadata is not None else (
                    None, False)

                input_shape = self.prev[0].shape
                divisor = np.prod(
                    input_shape) if axis is None else input_shape[axis]
                if axis is not None and keepdims is False:
                    if isinstance(grad, Tensor):
                        grad = grad.expand_dims(axis)
                    else:
                        grad = np.expand_dims(grad, axis)
                if isinstance(grad, Tensor):
                    grad = grad.broadcast_to(input_shape) / divisor
                else:
                    grad = np.broadcast_to(grad, input_shape) / divisor

                self.prev[0].grad = self._accumulate(
                    self.prev[0].grad, grad, create_graph)

            case Operation.VARIANCE:
                axis, keepdims = self.sum_metadata if self.sum_metadata is not None else (
                    None, False)
                x = self.prev[0]
                # number of elements reduced to get each output element
                N = x.data.size / self.data.size

                mu = x.data.mean(axis=axis, keepdims=True)

                grad_reshaped = grad

                if not keepdims and axis is not None:
                    axes = (axis,) if isinstance(axis, int) else axis
                    new_shape = list(x.data.shape)

                    for ax in axes:
                        new_shape[ax] = 1
                    grad_reshaped = grad.reshape(tuple(new_shape))

                if create_graph:
                    grad_input = (2.0 / N) * (x - mu) * grad_reshaped
                else:
                    grad_input = (2.0 / N) * (x.data - mu) * grad_reshaped

            case Operation.BROADCAST_TO:
                input_shape = self.prev[0].shape

                self.prev[0].grad = self._accumulate(
                    self.prev[0].grad, self._unbroadcast(grad, input_shape), create_graph)

            case Operation.RESHAPE:
                self.prev[0].grad = self._accumulate(
                    self.prev[0].grad, grad.reshape(self.prev[0].shape), create_graph)

            case Operation.EXPAND_DIMS:
                self.prev[0].grad = self._accumulate(
                    self.prev[0].grad, grad.reshape(self.prev[0].shape), create_graph)
            case Operation.SQUEEZE:
                self.prev[0].grad = self._accumulate(
                    self.prev[0].grad,
                    grad.reshape(self.prev[0].shape),
                    create_graph
                )
            case Operation.SWAP_AXIS:
                axis1, axis2 = self.swap_axis_metadata if self.swap_axis_metadata is not None else (
                    0, 0)
                self.prev[0].grad = self._accumulate(
                    self.prev[0].grad,
                    grad.swap_axis(axis1, axis2) if isinstance(grad, Tensor) else np.swapaxes(
                        grad, axis1, axis2),
                    create_graph)

            case Operation.EXP:
                self.prev[0].grad = self._accumulate(
                    self.prev[0].grad,
                    (self * self.exp()) *
                    grad if create_graph else (
                        self.data * np.exp(self.data)) * grad,
                    create_graph,
                )

            case Operation.RELU:
                self.prev[0].grad = self._accumulate(
                    self.prev[0].grad,
                    ((self.data > 0) * grad) if create_graph else ((self.data > 0) * grad),
                    create_graph,
                )

            case Operation.SOFTMAX:
                axis, softmax_data = self.softmax_data if self.softmax_data is not None else (
                    -1, np.array([]))
                if create_graph:
                    grad_input = self * grad - (self * grad).sum(
                        axis=axis, keepdims=True) * self
                else:
                    grad_input = softmax_data * grad - (softmax_data * grad).sum(
                        axis=axis, keepdims=True) * softmax_data

                self.prev[0].grad = self._accumulate(
                    self.prev[0].grad, grad_input, create_graph)

            case Operation.CUSTOM:
                if self.custom_grad_func is None:
                    raise ValueError("Custom grad func not defined")
                custom_grad = self.custom_grad_func(grad)
                self.prev[0].grad = self._accumulate(
                    self.prev[0].grad, custom_grad, create_graph)
