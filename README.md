# autograd

A from-scratch automatic differentiation library implemented twice: first as a NumPy prototype in Python, then as a compiled C++ library. Both share the same architecture — a symbolic computation graph, an optimising compiler, and a separate executor — but the C++ version adds a custom tensor type, a multi-pass optimising compiler, and a persistent execution buffer.

```
python/   — NumPy prototype (reference implementation)
cpp/      — C++ implementation
```

---

## Architecture overview

Both implementations follow the same three-phase pipeline:

```
Graph construction  →  Compiler  →  Executor
  (Symbol API)          (optimise)    (evaluate)
```

**Graph construction.** The user builds a computation graph using `Symbol` objects and arithmetic operators. No values are computed at this stage — every operation just records itself as a node in the graph. This separation means the same graph can be compiled once and run many times with different inputs.

**Compiler.** Takes the graph (or a subgraph reachable from specific output symbols), runs a sequence of optimisation passes over a working copy, and emits a flat, topologically-sorted `Program` (C++) or `topo` list (Python). The original graph is never mutated.

**Executor.** Walks the compiled program in order, evaluating each node given its already-evaluated inputs. In C++, the executor maintains a persistent scratch buffer across calls to avoid re-allocating storage on every forward pass.

**Symbolic differentiation.** `backwards()` / `_sym_grad()` traverses the graph in reverse topological order and appends gradient nodes back into the same graph using the chain rule. These gradient expressions are then compiled and executed like any other subgraph — there is no separate backward engine.

---

## Python (`python/`)

The Python implementation is the reference design. It uses NumPy arrays as values.

### `Symbol` (`autograd/symbol.py`)

A `Symbol` is a graph node. It holds:
- `operation` — an `Operation` enum value
- `prev` — list of input `Symbol`s
- `value` — a NumPy array for `CONSTANT` and `VARIABLE` nodes
- `kwargs` — operation parameters (e.g. `axis`, `keepdims`)
- `requires_grad` — propagated upward from any input that requires grad

Arithmetic operators (`+`, `*`, `**`, `@`, etc.) create new `Symbol` nodes. Scalars and arrays are wrapped in `CONSTANT` nodes automatically.

### `Compiler` (`autograd/compiler.py`)

`compile(symbol, backwards=False)` returns a topologically-sorted list of `Symbol`s reachable from `symbol`. When `backwards=True`, it first runs `compile_backwards()` to append gradient nodes, then compiles the augmented graph.

Optimisation runs as a fixed-point loop over the topological list:

1. **Canonicalisation** — rewrites sugar ops into primitives: `NEG → MULTIPLY(-1)`, `SUB → ADD + MULTIPLY(-1)`, `SQRT → POWER(0.5)`. This reduces the number of cases later passes need to handle.
2. **ADD / MULTIPLY flattening** — collapses chains of the same associative op into a single n-ary node: `ADD(ADD(a,b), c) → ADD(a,b,c)`.
3. **Constant folding** — evaluates ops whose every input is a known constant at compile time.
4. **Algebraic simplification** — applies identities: `x+0→x`, `x*1→x`, `x*0→0`, `x^0→1`, `x^1→x`, `log(exp(x))→x`, `exp(log(x))→x`.
5. **Dead code elimination** — drops nodes not reachable from any output.

The loop repeats until no pass reports a change.

### `Executor` (`autograd/Executor.py`)

`forward(topo, values)` walks the compiled list and evaluates each node. A `match` on `Operation` dispatches to the appropriate NumPy call. Intermediate results are cached by node identity.

### Symbolic differentiation (`autograd/grad.py`)

`_sym_grad` calls `compile(symbol, backwards=True)`, which internally calls `compile_backwards()`. That function traverses the graph in reverse, accumulates gradient `Symbol`s for each input using the chain rule, and attaches them as `.grad` on each node. The gradient expressions are then compiled independently (`compiler.compile(var.grad)`) so each partial derivative is its own optimised subgraph.

### Running

```bash
cd python
pip install -e ".[test]"
pytest
```

---

## C++ (`cpp/`)

The C++ implementation mirrors the Python design but adds static typing, a custom tensor type, and a more structured three-phase compiler.

### `Tensor` (`include/autograd/tensor.hpp`)

A custom N-dimensional tensor with **Small Buffer Optimisation (SBO)**. Up to `kInline` floats are stored directly inside the object with no heap allocation; larger tensors heap-allocate. Data is column-major in the last two dimensions to match Eigen's default layout, so `map()` and `matrix_slice()` produce zero-copy `Eigen::Map` views into the buffer.

```
kInline = 24   (inline storage for up to a 4×6 or 3×8 matrix)
kMaxDims = 4   (up to 4D tensors for batched operations)
```

Key design choices:
- `rows()` and `cols()` always refer to the last two dimensions, so 2D matrix operations compose naturally with batched 3D/4D tensors.
- `matrix_slice(b)` provides a view into batch slice `b` as a `rows×cols` Eigen matrix, enabling batched matmul via a simple loop.
- Copy and move follow value semantics. A move from a heap tensor steals the pointer; a move from an inline tensor memcopies.

### `Graph` and `Symbol` (`include/autograd/graph.hpp`, `include/autograd/symbol.hpp`)

`Graph` owns all nodes. A `Symbol` is a lightweight handle — just a `(node_index, Graph*)` pair. Arithmetic operators on `Symbol` call `graph.add_node()` and return a new `Symbol`. `requires_grad` is propagated eagerly: a node sets it if any input has it set.

Nodes store up to two inputs inline (`inputs[2]`); n-ary ops (ADD, MULTIPLY with more than two operands after flattening) spill into a CSR pool (`Graph::inputs`). Constants store an index into `Graph::values`.

### `Compiler` (`include/autograd/compiler.hpp`)

`Compiler::compile(graph, inputs, outputs)` is a static method that runs the full pipeline and returns a `Program`. It never modifies the original graph.

**Phase 1 — subgraph extraction** (`passes/extract.cpp`). Iterative post-order DFS from the requested outputs. Only nodes reachable from the outputs are copied into working buffers. All node indices are remapped from original-graph coordinates to working-buffer coordinates.

**Phase 2 — optimisation loop** (`passes/canonicalize.cpp`, `fold.cpp`, `algebra.cpp`). Runs up to 10 times (configurable) until no pass fires. Passes append new nodes and mark obsolete ones as `Op::ALIAS`; `resolve()` chases alias chains transparently so later passes always see the real underlying node.

- **Canonicalise** — `NEG→MULTIPLY(-1)`, `SUB→ADD(MULTIPLY(-1))`, `SQRT→POWER(0.5)`.
- **Flatten ADD / MULTIPLY** — absorbs child nodes of the same op into a single n-ary node.
- **Constant folding** — evaluates scalar-constant ops at compile time; reuses four well-known slots (0, 1, -1, 0.5) to avoid duplicate constant nodes.
- **Algebraic simplification** — `x+0→x`, `x*1→x`, `x*0→0`, `x^0→1`, `x^1→x`, `log(exp(x))→x`.

**Phase 3 — compaction** (`passes/compact.cpp`). Post-order DFS over the working graph, chasing aliases, remapping indices into the final `Program`. Dead nodes (unreachable or aliased away) are silently dropped.

### `Program` (`include/autograd/ir.hpp`)

A self-contained flat representation of a computation:
- `nodes` — topologically sorted, alias-free
- `values` — constant tensors referenced by index
- `inputs` — CSR pool for n-ary op inputs
- `input_nodes` / `output_nodes` — maps runtime feeds to node indices

### `Executor` (`include/autograd/executor.hpp`)

`forward(prog, feed)` fills a `values` scratch buffer (one `Tensor` per node) then walks `prog.nodes` in order. VARIABLE and PLACEHOLDER nodes are pre-filled from `feed`; CONSTANT nodes copy from `prog.values`; all other nodes call `eval_node`.

The `values` buffer is a member of `Executor` and persists across calls. It only reallocates when the new program is larger than any previously seen, avoiding repeated allocation in training loops.

Autobroadcasting follows NumPy rules and is handled entirely in the executor — shapes are aligned from the right, each dimension must be equal or one operand must be 1. This keeps the compiler free of shape inference.

### Symbolic differentiation (`src/backward.cpp`)

`backwards(symbol)` traverses the graph in reverse topological order using an explicit stack. For each node it computes the local gradient and accumulates it into `grads[input_idx]` by appending new nodes to the **same** graph. The returned `grads` vector maps original node indices to gradient `Symbol`s, which can then be compiled and executed independently.

**Important implementation note.** `backwards()` copies each `Node` by value before processing it (`Node node = graph.nodes[index]`). It must not hold a reference, because `graph.nodes` may be reallocated by `push_back` calls inside the backward pass itself.

### Running

```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/tests/tests          # Catch2 unit tests (104 assertions)
./build/basic                # end-to-end integration tests
```

---

## C++ design choices and optimisations

### Tensor SBO — eliminating heap allocations for small tensors

The most common tensors in an autograd library are scalars and small constants (weights, bias vectors, gradient accumulators). A plain `Eigen::MatrixXf` heap-allocates for every matrix, including 1×1 scalars. `Tensor` instead stores up to `kInline = 24` floats directly inside the object in a union with the heap pointer:

```
[ndim_ 1B][padding 3B][shape_[4] 16B][union: inline_data_[24] or heap_data* 8B]
```

A scalar, a bias row, or any matrix up to 4×6 never touches the heap. The Eigen bridge (`map()`, `matrix_slice()`) returns a zero-copy `Eigen::Map` into whichever buffer is active, so Eigen operations work identically on inline and heap tensors.

Column-major layout was chosen to match Eigen's default, so `Tensor(const Eigen::MatrixXf &m)` copies `m.data()` directly with `memcpy` and `map()` round-trips without any transposition.

### Persistent executor scratch buffer

The naive approach allocates a fresh `std::vector<Tensor>` on every `forward()` call. Instead, `Executor` keeps `values` as a member and only `resize`s it when the program is larger than any previously seen program. In a training loop running thousands of iterations, this reduces allocator traffic to a one-time cost.

### Compiler working buffers with ALIAS nodes

The compiler never modifies the original `Graph`. All optimisation happens in private working buffers (`working_nodes`, `working_inputs`, `working_values`). Passes only ever *append* new nodes — they never remove or reorder. When a node is made redundant by an optimisation, it is replaced with an `Op::ALIAS` entry whose single input points at the surviving node. `resolve()` chases alias chains with path compression so every subsequent pass pays at most one indirection per lookup.

This means the working buffer is a monotonically growing append-only structure, which avoids index invalidation and keeps passes simple.

### CSR storage for n-ary inputs
Most nodes have two or fewer inputs, which fit in `Node::inputs[2]` with no extra allocation. Flattened ADD and MULTIPLY nodes with more inputs spill into a flat CSR pool (`Graph::inputs`, `Program::inputs`) indexed by `(input_pool_offset, input_count)`. This keeps `Node` small and cache-friendly (only 32 bytes) while still supporting n-ary ops without a `std::vector` per node.
 
### Well-known constant deduplication

The four most common scalar constants (0, 1, -1, 0.5) are allocated once into the working buffer via `ensure_well_known()` and reused by any pass that needs them. Constant folding checks the result against these slots before appending a fresh constant node. This prevents the working graph from accumulating many identical scalar nodes during repeated canonicalisations and gradient accumulation.

### Subgraph extraction — only compile what's needed

`extract_subgraph()` performs a post-order DFS from the requested output symbols and copies only reachable nodes. If the user compiles a gradient subgraph for a single variable, none of the unrelated forward nodes reach the working buffer. Dead code elimination happens structurally, not as a separate pass.

### Autobroadcasting in the executor, not the compiler

NumPy-style broadcasting is resolved at runtime by the executor (`merge_broadcast_shape`, `broadcast_to_shape`), not during compilation. This keeps the compiler free of shape inference — it never needs to know tensor shapes — which makes passes simpler and avoids a whole class of shape-propagation bugs. The executor fast-paths the common case where shapes already match with a direct return.

### Separation of graph construction from execution

`Compiler::compile` is a static method that produces a self-contained `Program` with no references back to the `Graph`. The `Program` owns its own copy of constants, input/output indices, and the flattened node list. At runtime the executor only touches `Program` memory; the `Graph` and all `Symbol` handles can be destroyed. This makes compiled programs safe to cache, reuse across threads, or serialize.

### Backward pass node copying

`backwards()` appends new gradient nodes to the graph during traversal. Any `Node&` reference into `graph.nodes` becomes dangling if `push_back` causes a reallocation. Both the outer traversal loop and `backwards_node` therefore copy the current node by value before doing any work:

```cpp
const Node node = graph.nodes[index];  // copy — safe across push_back
```

This is a correctness requirement, not a performance trade-off — without it, `node.input_count` is read from freed memory after the first gradient node is appended, causing the traversal to silently skip child nodes.

---

## Supported operations

| Category | Operations |
|---|---|
| Arithmetic | ADD, MULTIPLY, SUB, NEG, POWER, SQRT |
| Transcendental | EXP, LOG, ABS, SIGN |
| Linear algebra | MATMUL, TRANSPOSE |
| Reductions | SUM, MEAN, VARIANCE, SIZE |
| Activation | RELU, SOFTMAX |
| Shape | RESHAPE, EXPAND_DIMS, SQUEEZE, SWAP_AXIS |
| Broadcasting | BROADCAST_TO, BROADCAST_TO_MATCH, UNBROADCAST |
| Stacking | VECTOR, GET_ITEM, SCATTER_LIKE |
