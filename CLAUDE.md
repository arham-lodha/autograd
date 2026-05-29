# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Collaboration style

- Default to giving advice, architectural guidance, and next-step recommendations — not writing code — unless explicitly asked.
- Writing tests is an exception: do write them when asked.
- When the user says "update the HTML file", append the important recent decisions/advice to `python/docs/notes.html`. The file should be a complete, well-styled HTML document — embedded CSS, good typography, color, spacing, clear section headings, date stamps. If the file doesn't exist, create it with a full HTML skeleton. If it does exist, add new dated `<section>` entries and update the TOC.
- Each section must have a short anchor ID (e.g. `id="s3"`) so the user can reference it in future conversations by saying something like "see s2". When the user references a section ID, look it up in `python/docs/notes.html` for context.
- Only update the file when explicitly told to, not during normal conversation.

## Layout

```
python/   — NumPy prototype (reference implementation)
cpp/      — C++ implementation (Eigen → custom stride tensor)
```

## Python commands

All commands must be run from `python/`.

```bash
# Install in editable mode (required once)
pip install -e ".[test]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_backward.py

# Run a single test class or function
pytest tests/test_backward.py::TestAddBackward::test_scalar_add

# Run slow/stress tests (excluded by default)
pytest cases/ -m slow

# Skip slow tests explicitly
pytest -m "not slow"
```

## C++ commands

Run from `cpp/`.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

## Architecture

### Python prototype (`python/`)

This is a NumPy-based automatic differentiation library. The **symbolic/compiled mode is the current version**. The eager mode (`Tensor` + `Engine` in `autograd/eager/`) is a legacy prototype — do not treat it as the active codebase.

#### Symbolic / compiled mode (`Symbol` + `Compiler` + `Executor`)
- **`python/autograd/symbol.py`** — `Symbol` builds a pure computation graph with no values attached. It mirrors the old `Tensor` API but stores only operations and `prev` links. Extra kwargs (e.g., `axis`, `keepdims`, `shape`) are stored in `node.kwargs`.
- **`python/autograd/compiler.py`** — `Compiler.compile()` runs a fixed-point optimization loop over the graph: canonicalization (NEG→MUL(-1), SUB→ADD+MUL, SQRT→POW(0.5)), constant folding, addition/multiplication flattening, algebraic simplification (x+0, x*1, x^0, log(exp(x)), etc.), dead code elimination, then decanonicalization. `compile_backwards()` symbolically differentiates the compiled graph.
- **`python/autograd/Executor.py`** — `Executor.forward()` runs a compiled `topo` list, evaluating each node via a `match` on `Operation` and caching intermediate values for nodes that need gradients.
- **`python/autograd/ops.py`** — `Operation` enum shared across the codebase.

#### Key design patterns
- **ADD/MULTIPLY flattening**: `Compiler._fold_addition`/`_fold_multiplication` collapse chains of the same operation into a single n-ary node. The `retain=True` flag on a `Symbol` prevents it from being absorbed into its parent.
- **`_unbroadcast`**: Gradient un-broadcasting to handle shape mismatches from NumPy broadcasting.
- Operation kwargs (e.g., `axis`, `keepdims`, `shape`) are stored in `node.kwargs` on the `Symbol` and passed through to the `Executor` at runtime.

#### Tests
- `python/tests/` — unit tests; `conftest.py` provides the `engine` fixture and a `numerical_grad` central-differences helper used to verify analytical gradients.

### C++ implementation (`cpp/`)

- **`cpp/include/autograd/ops.hpp`** — `Op` enum mirroring the Python `Operation` enum.
- **`cpp/include/autograd/tensor.hpp`** — Tensor type (wraps Eigen `MatrixXd` initially; will become a custom stride array).
- **`cpp/include/autograd/symbol.hpp`** — `Symbol` node for the computation graph.
- **`cpp/include/autograd/compiler.hpp`** — Graph optimization passes.
- **`cpp/include/autograd/executor.hpp`** — Forward-pass evaluation.
- **`cpp/src/`** — Corresponding `.cpp` implementations.
- **`cpp/tests/`** — Catch2 unit tests.
