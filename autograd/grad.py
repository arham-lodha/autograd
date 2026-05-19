from typing import Callable, Dict, List, Optional, Tuple, Union
import inspect

import numpy as np

from .ops import Operation
from .symbol import Symbol, NonSymbolInputs
from .compiler import Compiler
from .Executor import Executor


def grad(
    f: Callable[..., Symbol],
    argnums: Optional[Union[int, Tuple[int, ...]]] = None,
    compiler: Optional[Compiler] = None,
    executor: Optional[Executor] = None,
) -> Tuple[Callable[..., List[np.ndarray]], Callable[..., np.ndarray]]:
    if compiler is None:
        compiler = Compiler()
    if executor is None:
        executor = Executor()

    num_args: int = len(inspect.signature(f).parameters)

    # Normalise argnums: default is all arguments
    if argnums is None:
        active: set[int] = set(range(num_args))
    elif isinstance(argnums, int):
        active = {argnums}
    else:
        active = set(argnums)

    # VARIABLE for differentiated args, PLACEHOLDER for the rest
    syms: List[Symbol] = [
        Symbol(operation=Operation.VARIABLE, requires_grad=True)
        if i in active
        else Symbol(operation=Operation.PLACEHOLDER, requires_grad=False)
        for i in range(num_args)
    ]

    evaluation_tree: Symbol = f(*syms)

    forward_topo: List[Symbol] = compiler.compile(evaluation_tree, backwards=True)

    # Only compile gradient topos for active (VARIABLE) symbols
    active_syms: List[Symbol] = [syms[i] for i in sorted(active)]
    grad_topos: List[List[Symbol]] = []
    for sym in active_syms:
        assert sym.grad is not None, "No gradient for input symbol — check requires_grad"
        grad_topos.append(compiler.compile(sym.grad, skip_optimization=True))

    def evalf(*values: NonSymbolInputs) -> np.ndarray:
        feed: Dict[Symbol, NonSymbolInputs] = {s: np.asarray(v) for s, v in zip(syms, values)}
        return executor.forward(forward_topo, feed_dict=feed)

    def df(*values: NonSymbolInputs) -> List[np.ndarray]:
        feed: Dict[Symbol, NonSymbolInputs] = {s: np.asarray(v) for s, v in zip(syms, values)}
        executor.forward(forward_topo, feed_dict=feed)
        return [executor.forward(gt, feed_dict=feed) for gt in grad_topos]

    return df, evalf
