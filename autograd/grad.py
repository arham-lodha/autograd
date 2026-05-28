from typing import Callable, Dict, List, Optional, Tuple, Union
import inspect

import numpy as np

from .ops import Operation
from .symbol import Symbol, NonSymbolInputs
from .compiler import Compiler
from .Executor import Executor

def _sym_grad(f: Union[Symbol, List[Symbol]], compiler: Compiler, executor: Executor, wrt: Optional[Union[Symbol, List[Symbol]]] = None):
    if isinstance(f, Symbol):
        compiled_f = compiler.compile(f, backwards=True)
        variables = [node for node in compiled_f if node.operation == Operation.VARIABLE]

        if wrt is None:
            wrt = variables
        if isinstance(wrt, Symbol):
            wrt = [wrt]

        # Save gradient Symbols before compiling — these are the handles for higher-order
        grad_syms: List[Optional[Symbol]] = [var.grad for var in wrt]
        grad_topos: List[Optional[List[Symbol]]] = [
            compiler.compile(gs) if gs is not None else None
            for gs in grad_syms
        ]

        def evalf(values: Dict[Symbol, NonSymbolInputs]) -> np.ndarray:
            return executor.forward(compiled_f, values)

        def df(values: Dict[Symbol, NonSymbolInputs]) -> List[np.ndarray]:
            return [
                executor.forward(gt, values) if gt is not None else np.zeros(())
                for gt in grad_topos
            ]

        return df, evalf, grad_syms

    else:
        if isinstance(wrt, Symbol):
            wrt = [wrt]

        forward_topos: List[List[Symbol]] = []
        # grad_maps[i] maps id(var) -> compiled grad topo for component i
        grad_maps: List[Dict[int, List[Symbol]]] = []
        all_vars: Dict[int, Symbol] = {}  # insertion-ordered, deduped across components

        for component in f:
            fwd = compiler.compile(component, backwards=True)
            forward_topos.append(fwd)

            # Collect variables from this topo and record their grad topos immediately —
            # the next compile_backwards call will overwrite .grad on shared nodes
            grad_map: Dict[int, List[Symbol]] = {}
            check = wrt if wrt is not None else [n for n in fwd if n.operation == Operation.VARIABLE]
            for var in check:
                if id(var) not in all_vars:
                    all_vars[id(var)] = var
                if var.grad is not None:
                    grad_map[id(var)] = compiler.compile(var.grad)
            grad_maps.append(grad_map)

        wrt_list: List[Symbol] = wrt if wrt is not None else list(all_vars.values())

        jacobian_topos: List[List[Optional[List[Symbol]]]] = [
            [grad_maps[i].get(id(var)) for var in wrt_list]
            for i in range(len(f))
        ]

        def evalf(values: Dict[Symbol, NonSymbolInputs]) -> List[np.ndarray]:
            return [executor.forward(fwd, values) for fwd in forward_topos]

        def df(values: Dict[Symbol, NonSymbolInputs]) -> List[List[np.ndarray]]:
            return [
                [executor.forward(topo, values) if topo is not None else np.zeros(())
                 for topo in row]
                for row in jacobian_topos
            ]

        return df, evalf








def grad(
    f: Union[Callable[..., Symbol], Symbol, List[Symbol]],
    wrt: Optional[Union[Symbol, List[Symbol]]] = None,
    argnums: Optional[Union[int, Tuple[int, ...]]] = None,
    compiler: Optional[Compiler] = None,
    executor: Optional[Executor] = None,
):
   
    if compiler is None:
        compiler = Compiler()
    if executor is None:
        executor = Executor()
    if isinstance(f, Symbol):
        return _sym_grad(f, compiler=compiler, executor=executor, wrt=wrt)
    if isinstance(f, list):
        return _sym_grad(f, compiler=compiler, executor = executor, wrt=wrt)

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
        grad_topos.append(compiler.compile(sym.grad))

    def evalf(*values: NonSymbolInputs) -> np.ndarray:
        feed: Dict[Symbol, NonSymbolInputs] = {s: np.asarray(v) for s, v in zip(syms, values)}
        return executor.forward(forward_topo, feed_dict=feed)

    def df(*values: NonSymbolInputs) -> List[np.ndarray]:
        feed: Dict[Symbol, NonSymbolInputs] = {s: np.asarray(v) for s, v in zip(syms, values)}
        executor.forward(forward_topo, feed_dict=feed)
        return [executor.forward(gt, feed_dict=feed) for gt in grad_topos]

    return df, evalf, active_syms
