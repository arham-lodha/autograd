from typing import Callable, Dict, List, Optional, Tuple
import inspect

import numpy as np

from .ops import Operation
from .symbol import Symbol, NonSymbolInputs
from .compiler import Compiler
from .Executor import Executor


def grad(
    f: Callable[..., Symbol],
    compiler: Optional[Compiler] = None,
    executor: Optional[Executor] = None,
) -> Tuple[Callable[..., List[np.ndarray]], Callable[..., np.ndarray]]:
    if compiler is None:
        compiler = Compiler()
    if executor is None:
        executor = Executor()

    num_args: int = len(inspect.signature(f).parameters)
    syms: List[Symbol] = [
        Symbol(operation=Operation.VARIABLE, requires_grad=True)
        for _ in range(num_args)
    ]
    evaluation_tree: Symbol = f(*syms)

    forward_topo: List[Symbol] = compiler.compile(evaluation_tree, backwards=True)

    grad_topos: List[List[Symbol]] = []
    for sym in syms:
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
