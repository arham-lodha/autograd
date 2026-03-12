from typing import List, Dict, Callable, Set, Tuple
import numpy as np
from collections import Counter

from .ops import Operation
from .symbol import NonSymbolInputs, Symbol, Inputs


class Compiler:
    def __init__(self):
        self._eval_map: Dict[Operation, Callable] = {
            Operation.POWER: lambda x, y: x ** y,
            Operation.LOG: np.log,
            Operation.MATMUL: lambda x, y: x @ y,
            Operation.TRANSPOSE: lambda x: x.T,
            Operation.SUM: lambda x, axis=None: np.sum(x, axis=axis),
            Operation.EXP: np.exp,
            Operation.MEAN: lambda x, axis=None: np.mean(x, axis=axis),
            Operation.RELU: lambda x: np.maximum(0, x),
            Operation.SOFTMAX: lambda x, axis=None: np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True),
            Operation.VARIANCE: lambda x, axis=None: np.var(x, axis=axis),
            Operation.NEG: lambda x: -x,
            Operation.SUB: lambda x, y: x - y,
            Operation.SQRT: np.sqrt,
            Operation.IDENTITY: lambda x: x
        }

    def _build_topo(self, symbol: Symbol) -> List[Symbol]:
        topo: List[Symbol] = []
        visited = set()
        stack = [symbol]

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

        return topo

    def _make_constant(self, value: NonSymbolInputs) -> Symbol:
        return Symbol(operation=Operation.CONSTANT, value=value, prev=[], requires_grad=False)

    def _canonicalize(self, topo: List[Symbol]) -> Tuple[List[Symbol], bool]:

        # Convert all operations to a canonical form (e.g., convert subtraction to addition with negative, division to multiplication with reciprocal) to simplify further optimizations
        # This step ensures that we have a consistent representation of operations, which can make it easier to apply algebraic simplifications and other optimizations later on.
        # Once all optimizations are applied, we can convert back to the original operations if needed, but for the purpose of this compiler, we will keep everything in the canonical form.

        new_topo: List[Symbol] = []
        changed = False

        for node in topo:
            if node.operation is Operation.NEG:
                # Convert negation to multiplication by -1, to allow for constant folding. This also allows us to fold negation into addition and subtraction, which can enable further optimizations.

                # We want to do this transformation in place so that the node's properties are preserved.
                neg_one = self._make_constant(-1.0)
                node.operation = Operation.MULTIPLY
                node.prev.append(neg_one)
                new_topo.append(neg_one)
                changed = True

            elif node.operation is Operation.SUB:
                # Convert subtraction to addition of a negation, to allow for constant folding and further optimizations. This also allows us to fold subtraction into addition, which can enable further optimizations.
                neg_one = self._make_constant(-1.0)
                multiply_node = Symbol(operation=Operation.MULTIPLY, prev=[
                                       node.prev[1], neg_one])

                node.operation = Operation.ADD
                node.prev = [node.prev[0], multiply_node]
                new_topo.append(neg_one)
                new_topo.append(multiply_node)
                changed = True

            new_topo.append(node)

        return new_topo, changed

    def _fold_constants(self, topo: List[Symbol]) -> Tuple[List[Symbol], bool]:
        folded_topo = []
        changed = False
        for node in topo:

            if len(node.prev) != 0 and node.operation not in (Operation.ADD, Operation.MULTIPLY):

                if node.operation in self._eval_map and all(p.operation == Operation.CONSTANT for p in node.prev):
                    # Special case for sqrt to speed up.
                    if node.operation is Operation.POWER and np.all(node.prev[1].value == 0.5):
                        node.value = np.sqrt(
                            node.prev[0].value if node.prev[0].value is not None else 0.0)
                        node.operation = Operation.CONSTANT
                        node.prev = []
                        node.requires_grad = False
                    else:
                        inputs = [p.value for p in node.prev]
                        node_value = self._eval_map[node.operation](*inputs)

                        node.operation = Operation.CONSTANT
                        node.value = node_value
                        node.prev = []
                        node.requires_grad = False
                folded_topo.append(node)
                continue

            constants = [p for p in node.prev if p.operation ==
                         Operation.CONSTANT]
            non_constants = [
                p for p in node.prev if p.operation != Operation.CONSTANT]

            if len(constants) > 1:
                values = [p.value for p in constants if p.value is not None]
                if node.operation == Operation.ADD:
                    const_value = np.add.reduce(values)
                else:
                    const_value = np.multiply.reduce(values)

                constant = self._make_constant(
                    const_value)  # Add the folded constant back to the graph

                if not non_constants:
                    node.operation = Operation.CONSTANT
                    node.value = const_value
                    node.prev = []
                    node.requires_grad = False
                else:
                    folded_topo.append(constant)
                    node.prev = non_constants + [constant]

                changed = True
            folded_topo.append(node)

        return folded_topo, changed

    def _fold_addition(self, topo: List[Symbol]) -> Tuple[List[Symbol], bool]:
        folded_topo: List[Symbol] = []
        changed = False

        for node in topo:
            if node.operation == Operation.ADD:
                # Check if any of the parents is an addition
                add_parents = [
                    p for p in node.prev if p.operation == Operation.ADD]
                if add_parents:
                    # If there are addition parents, we can fold them into the current addition
                    new_prev = []
                    for p in node.prev:
                        if p.operation == Operation.ADD and not p.retain:
                            # Add the parents of the addition parent
                            new_prev.extend(p.prev)
                        else:
                            # Keep non-addition parents as they are
                            new_prev.append(p)
                    node.prev = new_prev  # Update the current node's parents to the folded list
                    changed = True

            folded_topo.append(node)

        return folded_topo, changed

    def _fold_multiplication(self, topo: List[Symbol]) -> Tuple[List[Symbol], bool]:
        folded_topo: List[Symbol] = []
        changed = False

        for node in topo:
            if node.operation == Operation.MULTIPLY:
                # Check if any of the parents is a multiplication
                mul_parents = [
                    p for p in node.prev if p.operation == Operation.MULTIPLY]
                if mul_parents:
                    # If there are multiplication parents, we can fold them into the current multiplication
                    new_prev = []
                    for p in node.prev:
                        if p.operation == Operation.MULTIPLY and not p.retain:
                            # Add the parents of the multiplication parent
                            new_prev.extend(p.prev)
                        else:
                            # Keep non-multiplication parents as they are
                            new_prev.append(p)
                    node.prev = new_prev  # Update the current node's parents to the folded list
                    changed = True

            folded_topo.append(node)

        return folded_topo, changed

    def _try_replace(self, node: Symbol, replacement: Symbol, replacements: Dict[int, Symbol]) -> bool:
        # If node has important metadata, we don't replace it but it just behaves like a wrapper.

        if node.retain or node.name:
            if replacement.operation == Operation.CONSTANT:
                node.operation = Operation.CONSTANT
                node.value = replacement.value
                node.prev = []
            else:
                node.operation = Operation.IDENTITY
                node.prev = [replacement]

            node.kwargs = {}
            return False

        replacements[id(node)] = replacement
        return True

    def _algebraic_simplification(self, topo: List[Symbol]) -> Tuple[List[Symbol], bool]:
        replacements: Dict[int, Symbol] = {}
        new_topo: List[Symbol] = []
        changed = False

        def get_actual(node: Symbol) -> Symbol:
            curr = node
            while id(curr) in replacements:
                curr = replacements[id(curr)]

            return curr

        for node in topo:
            # Update the parent nodes to get the actual nodes after replacement
            node.prev = [get_actual(p) for p in node.prev]

            # Optimization rules

            match node.operation:
                case Operation.ADD:
                    zeroes = [p for p in node.prev if p.operation ==
                              Operation.CONSTANT and np.all(p.value == 0)]

                    # If any parent is zero, we can simplify the addition
                    if zeroes:
                        non_zeroes = [
                            p for p in node.prev if p.operation != Operation.CONSTANT or not np.all(p.value == 0)]
                        if len(non_zeroes) == 0:
                            # If all parents are zero, replace with a single zero constant
                            zero_const = self._make_constant(0)
                            new_topo.append(zero_const)
                            changed = True

                            if not self._try_replace(node, zero_const, replacements):
                                new_topo.append(node)

                            continue
                        elif len(non_zeroes) == 1:
                            # If there's only one non-zero parent, replace with that parent if possible
                            changed = True

                            if not self._try_replace(node, non_zeroes[0], replacements):
                                new_topo.append(node)

                            continue

                        else:
                            node.prev = non_zeroes
                            changed = True

                    counts = Counter(id(p) for p in node.prev)

                    if any(count > 1 for count in counts.values()):
                        new_prev = []
                        seen = set()

                        for p in node.prev:
                            if id(p) in seen:
                                continue
                            count = counts[id(p)]
                            if count > 1:
                                constant = self._make_constant(float(count))
                                mul_node = Symbol(operation=Operation.MULTIPLY, prev=[
                                    constant, p]
                                )
                                new_topo.append(constant)
                                new_topo.append(mul_node)
                                new_prev.append(mul_node)
                            else:
                                new_prev.append(p)
                            seen.add(id(p))
                        node.prev = new_prev
                        changed = True

                case Operation.MULTIPLY:
                    ones = [p for p in node.prev if p.operation ==
                            Operation.CONSTANT and np.all(p.value == 1)]
                    zeroes = [p for p in node.prev if p.operation ==
                              Operation.CONSTANT and np.all(p.value == 0)]

                    # If any parent is zero, we can simplify the multiplication to zero
                    if zeroes:
                        zero_const = self._make_constant(0)
                        new_topo.append(zero_const)
                        changed = True

                        if not self._try_replace(node, zero_const, replacements):
                            new_topo.append(node)

                        continue
                    # If any parent is one, we can remove it from the multiplication
                    elif ones:
                        non_ones = [
                            p for p in node.prev if p.operation != Operation.CONSTANT or not np.all(p.value == 1)]
                        if len(non_ones) == 0:
                            # If all parents are one, replace with a single one constant
                            one_const = self._make_constant(1)
                            new_topo.append(one_const)
                            changed = True

                            if not self._try_replace(node, one_const, replacements):
                                new_topo.append(node)

                            continue
                        elif len(non_ones) == 1:
                            # If there's only one non-one parent, replace with that parent
                            changed = True

                            if not self._try_replace(node, non_ones[0], replacements):
                                new_topo.append(node)

                            continue
                        else:
                            node.prev = non_ones
                            changed = True

                case Operation.POWER:
                    base, exp = node.prev
                    # If the exponent is zero, we can simplify to one
                    if exp.operation == Operation.CONSTANT and np.all(exp.value == 0):
                        one_const = self._make_constant(1)
                        new_topo.append(one_const)
                        changed = True

                        if not self._try_replace(node, one_const, replacements):
                            new_topo.append(node)

                        continue
                    # If the exponent is one, we can simplify to the base
                    elif exp.operation == Operation.CONSTANT and np.all(exp.value == 1):
                        changed = True

                        if not self._try_replace(node, base, replacements):
                            new_topo.append(node)

                        continue

                    elif exp.operation == Operation.CONSTANT and exp.value is not None:
                        # if exponent is 2, 3, 4, we can simplify to multiplication
                        e = exp.value

                        if np.all(e == int(e.flat[0])) and 2 <= int(e.flat[0]) <= 4:
                            n = int(e.flat[0])
                            node.operation = Operation.MULTIPLY
                            node.prev = [base] * n
                            changed = True
                    elif base.operation == Operation.CONSTANT and np.all(base.value == 1):
                        # If the base is 1, we can simplify to 1
                        one_const = self._make_constant(1)
                        new_topo.append(one_const)
                        changed = True

                        if not self._try_replace(node, one_const, replacements):
                            new_topo.append(node)

                        continue
                    elif base.operation == Operation.POWER:
                        # If the base is also a power, we can simplify (x^a)^b to x^(a*b)
                        inner_base, inner_exp = base.prev
                        new_exp = Symbol(operation=Operation.MULTIPLY, prev=[
                                         inner_exp, exp])
                        node.prev = [inner_base, new_exp]
                        new_topo.append(new_exp)
                        changed = True
                case Operation.LOG:
                    arg = node.prev[0]
                    if arg.operation == Operation.EXP:
                        changed = True

                        if not self._try_replace(node, arg.prev[0], replacements):
                            new_topo.append(node)

                        continue
                case Operation.EXP:
                    arg = node.prev[0]
                    if arg.operation == Operation.LOG:
                        # exp(log(x)) simplifies to x
                        changed = True

                        if not self._try_replace(node, arg.prev[0], replacements):
                            new_topo.append(node)

                        continue
                case Operation.TRANSPOSE:
                    arg = node.prev[0]
                    if arg.operation == Operation.TRANSPOSE:
                        # Transpose of a transpose simplifies to the original matrix
                        changed = True

                        if not self._try_replace(node, arg.prev[0], replacements):
                            new_topo.append(node)

                        continue
                case Operation.RELU:
                    arg = node.prev[0]
                    if arg.operation == Operation.RELU and not arg.retain:
                        # ReLU of a ReLU simplifies to a single ReLU
                        node.prev = [arg]
                        changed = True
                        continue

            new_topo.append(node)

        return new_topo, changed

    def _dead_code_elimination(self, topo: List[Symbol]) -> Tuple[List[Symbol], bool]:

        if not topo:
            return [], False

        changed = False

        used = set()
        used.add(id(topo[-1]))

        live_topo = []

        for node in reversed(topo):
            if id(node) in used:
                live_topo.append(node)
                for p in node.prev:
                    used.add(id(p))
            else:
                changed = True

        return list(reversed(live_topo)), changed

    def _decanonicalize(self, topo: List[Symbol]) -> Tuple[List[Symbol], bool]:
        # Convert canonical expressions back to decanonical representations
        # If you have an element with multiplication with negative 1 convert to negation.

        new_topo: List[Symbol] = []
        changed = False
        for node in topo:

            if node.operation == Operation.MULTIPLY:
                neg_ones = [
                    p for p in node.prev
                    if p.operation == Operation.CONSTANT
                    and p.value is not None
                    and np.all(p.value == -1.0)
                ]

                operands = [p for p in node.prev if p not in neg_ones]

                if neg_ones:
                    changed = True
                    if len(neg_ones) % 2 == 1:
                        if len(operands) == 0:
                            node.operation = Operation.CONSTANT
                            node.prev = []
                            node.value = np.array(-1)
                            node.requires_grad = False
                        elif len(operands) == 1:
                            node.operation = Operation.NEG
                            node.prev = operands
                        else:
                            neg_one = self._make_constant(-1.0)
                            node.prev = operands + [neg_one]
                            new_topo.append(neg_one)
                    else:
                        if len(operands) == 0:
                            node.operation = Operation.CONSTANT
                            node.value = np.array(1)
                            node.requires_grad = False
                        elif len(operands) == 1:
                            node.operation = Operation.IDENTITY
                            node.prev = operands
                        else:
                            node.prev = operands
            new_topo.append(node)

        return new_topo, changed

    def compile(self, symbol: Symbol, max_iterations: int = 10, skip_optimization: bool = False) -> List[Symbol]:

        topo: List[Symbol] = self._build_topo(symbol)

        if not skip_optimization:
            for _ in range(max_iterations):
                graph_changed = False

                topo, changed = self._canonicalize(topo)
                graph_changed |= changed

                topo, changed = self._fold_addition(topo)
                graph_changed |= changed
                topo, changed = self._fold_multiplication(topo)
                graph_changed |= changed

                topo, changed = self._fold_constants(topo)
                graph_changed |= changed

                topo, changed = self._algebraic_simplification(
                    topo)
                graph_changed |= changed

                topo, changed = self._dead_code_elimination(
                    topo)
                graph_changed |= changed

                if not graph_changed:
                    break

            topo, changed = self._decanonicalize(topo)

            if changed:
                topo, changed = self._dead_code_elimination(topo)
        return topo
