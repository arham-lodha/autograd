from typing import List

from tensor import Tensor


class Engine:
    def __init__(self):
        self.cache: dict[int, List[Tensor]] = {}

    def build_topo(self, tensor: Tensor):

        if id(tensor) in self.cache:
            return self.cache[id(tensor)]

        topo = []
        visited = set()
        stack = [tensor]

        while stack:
            node = stack[-1]
            if id(node) not in visited:
                unvisited_children = [
                    c for c in node.prev if id(c) not in visited]
                if unvisited_children:
                    stack.extend(unvisited_children)
                else:
                    visited.add(id(node))
                    topo.append(node)
                    stack.pop()
            else:
                stack.pop()

        self.cache[id(tensor)] = topo

        return topo

    def backward(self, tensor: Tensor, create_graph=False):
        topo = self.build_topo(tensor)

        tensor.grad = 1.0

        for node in reversed(topo):
            node.backward(node.grad, create_graph=create_graph)

    def zero_grad(self, tensor: Tensor):
        topo = self.build_topo(tensor)
        for node in topo:
            node.grad = 0.0
