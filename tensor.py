import math
from typing import List, Optional, Union

from .ops import Operation


class Tensor:
    def __init__(
        self,
        data: Optional[float] = 1.0,
        operation: Optional[Operation] = Operation.CONSTANT,
        prev: Optional[List["Tensor"]] = None,
        scalar: float = 1.0,
        pow_scalar=1.0,
        retain=False,
    ):
        self.data: float = data or 1.0
        self.operation: Operation = operation or Operation.CONSTANT
        self.prev: List["Tensor"] = prev or []
        self.grad: Union[float, "Tensor"] = 0.0
        self._scalar: float = scalar
        self.pow_scalar: float = pow_scalar
        self.retain: bool = retain

    def __add__(self, other: Union["Tensor", float]):
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

    def __mul__(self, other: Union["Tensor", float]):
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
        new_scalar = 1.0
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

    def __pow__(self, other: Union["Tensor", float]):
        if not isinstance(other, Tensor):
            return Tensor(
                self.data**other,
                operation=Operation.POWER,
                prev=[self],
                pow_scalar=other,
            )

        out = Tensor(
            self.data**other.data, operation=Operation.POWER, prev=[self, other]
        )
        return out

    def __truediv__(self, other: Union["Tensor", float]):
        if not isinstance(other, Tensor):
            return self * (other**-1)  # scalar: just multiply by 1/other
        return self * (other**-1)  # Tensor: reuse pow and mul

    def __rtruediv__(self, other: Union["Tensor", float]):
        return self**-1 * other  # other / self

    def __sub__(self, other: Union["Tensor", float]):
        return self + (-other if not isinstance(other, Tensor) else other.__neg__())

    def __rsub__(self, other: Union["Tensor", float]):
        return self.__neg__() + other

    def __rmul__(self, other: Union["Tensor", float]):
        return self.__mul__(other)

    def __radd__(self, other: Union["Tensor", float]):
        return self.__add__(other)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad}, op={self.operation.name})"

    def __neg__(self):
        return self * -1

    def __log__(self):
        if self.data <= 0:
            raise ValueError("Cannot take the log of a non-positive number")
        return Tensor(math.log(self.data), operation=Operation.LOG, prev=[self])

    def _accumulate(self, existing, new_grad, create_graph: bool):
        if existing is None:
            return new_grad
        elif create_graph:
            return existing + new_grad

        return existing + (new_grad.data if isinstance(new_grad, Tensor) else new_grad)

    def _grad_mul_scalar(self, grad, create_graph: bool):
        if create_graph:
            return grad * self._scalar

        return self._scalar * (grad.data if isinstance(grad, Tensor) else grad)

    def _grad_mul_tensors(
        self, grad, others: List["Tensor"], scalar: float, create_graph: bool
    ):
        if create_graph:
            result = grad * scalar
            for t in others:
                result = result * t
            return result

        result = (grad.data if isinstance(grad, Tensor) else grad) * scalar
        for t in others:
            result = result * t.data
        return result

    def _grad_power_scalar(self, grad, base: "Tensor", create_graph: bool):
        if create_graph:
            return grad * (base ** (self.pow_scalar - 1.0)) * self.pow_scalar

        return (
            (grad.data if isinstance(grad, Tensor) else grad)
            * (base.data ** (self.pow_scalar - 1.0))
            * self.pow_scalar
        )

    def backward(self, grad, create_graph=True):
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
                                grad, others, self._scalar, create_graph
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
                            else (grad.data if isinstance(grad, Tensor) else grad)
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
                            else (grad.data if isinstance(grad, Tensor) else grad)
                            * math.log(a.data)
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
                        else (grad.data if isinstance(grad, Tensor) else grad)
                        / self.prev[0].data
                    ),
                    create_graph,
                )
