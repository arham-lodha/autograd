from typing import List, Optional, Union

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

    @property
    def shape(self):
        return self.data.shape

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

            if self.operation == Operation.MULTIPLY and not self.retain:
                new_prev.extend(self.prev)
            else:
                new_prev.append(self)

            return Tensor(
                self.data * other,
                operation=Operation.MULTIPLY,
                prev=new_prev,
                scalar=other,
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

    def __log__(self):
        return Tensor(np.log(self.data), operation=Operation.LOG, prev=[self])

    def _accumulate(self, existing: Union['Tensor', np.ndarray], new_grad: Union['Tensor', np.ndarray], create_graph: bool):
        if existing is None:
            return new_grad
        elif create_graph:
            return existing + new_grad

        return existing + (new_grad.data if isinstance(new_grad, Tensor) else new_grad)

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

    def backward(self, grad: Union["Tensor", np.ndarray], create_graph=False):

        if not create_graph and isinstance(grad, Tensor):
            grad = grad.data

        match self.operation:
            case Operation.CONSTANT:
                pass
            case Operation.ADD:
                for t in self.prev:
                    t.grad = self._accumulate(t.grad, grad, create_graph)

            case Operation.MULTIPLY:
                if len(self.prev) == 1:
                    self.prev[0].grad = self._accumulate(
                        self.prev[0].grad,
                        self._grad_mul_scalar(grad, create_graph),
                        create_graph,
                    )
                else:
                    for i, t in enumerate(self.prev):
                        others = [p for j, p in enumerate(self.prev) if j != i]

                        t.grad = self._accumulate(
                            t.grad,
                            self._grad_mul_tensors(
                                grad, others, create_graph
                            ),
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
                            grad * a.__log__() * (a**b)
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
