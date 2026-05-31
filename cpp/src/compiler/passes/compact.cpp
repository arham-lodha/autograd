#include "autograd/compiler.hpp"
#include "autograd/ir.hpp"
#include "autograd/ops.hpp"
#include "autograd/symbol.hpp"
#include <cstdint>
#include <stack>
#include <stdexcept>

namespace autograd {

// ── Phase 3: Finalization & Compaction ───────────────────────────────────
//
// Post-order DFS over working_nodes starting from working_outputs.
// ALIAS nodes are chased transparently on every input read; they are never
// emitted into the final Program, so the Executor never sees them.
//
// Because this is post-order, every child is emitted before its parent,
// guaranteeing strict topological ordering with no further sorting step.

Program Compiler::finalize_and_sort(const std::vector<Symbol> &original_inputs) {
  Program prog;
  const uint32_t n_working = static_cast<uint32_t>(working_nodes.size());

  // Maps working_nodes index → prog.nodes index.  UINT32_MAX = not yet emitted.
  std::vector<uint32_t> working_to_prog(n_working, UINT32_MAX);

  // state: 0 = unvisited, 1 = children pushed, 2 = emitted
  std::vector<uint8_t> state(n_working, 0);
  std::stack<uint32_t> stack;

  for (uint32_t out_w : working_outputs) {
    uint32_t root = resolve(out_w);
    if (state[root] == 0)
      stack.push(root);
  }

  while (!stack.empty()) {
    uint32_t idx = stack.top(); // idx is always resolve()-d before pushing

    if (state[idx] == 0) {
      state[idx] = 1;
      const Node &node = working_nodes[idx];
      for (uint32_t i = 0; i < node.input_count; i++) {
        uint32_t child = resolve(get_input(idx, i));
        if (state[child] == 0)
          stack.push(child);
      }
    } else if (state[idx] == 1) {
      state[idx] = 2;
      stack.pop();

      const Node &node = working_nodes[idx];
      Node n = node;
      n.input_pool_offset = static_cast<uint32_t>(prog.inputs.size());

      for (uint32_t i = 0; i < n.input_count; i++) {
        uint32_t child_w = resolve(get_input(idx, i));
        uint32_t child_p = working_to_prog[child_w];
        if (child_p == UINT32_MAX)
          throw std::runtime_error(
              "finalize_and_sort: child was not emitted before parent");

        if (i < 2)
          n.inputs[i] = child_p;
        else
          prog.inputs.push_back(child_p);
      }

      if (n.operation == Op::CONSTANT) {
        n.value_index = static_cast<uint32_t>(prog.values.size());
        prog.values.push_back(working_values[node.value_index]);
      }

      if (n.operation == Op::RESHAPE || n.operation == Op::BROADCAST_TO) {
        uint32_t old_off = node.shape.offset;
        n.shape.offset = static_cast<uint32_t>(prog.shapes.size());
        for (uint32_t i = 0; i < node.shape.count; i++)
          prog.shapes.push_back(working_shapes[old_off + i]);
      }

      working_to_prog[idx] = static_cast<uint32_t>(prog.nodes.size());
      prog.nodes.push_back(n);

    } else {
      stack.pop(); // diamond: already emitted, nothing to do
    }
  }

  // ── Map input symbols → prog indices ────────────────────────────────────
  prog.input_nodes.reserve(original_inputs.size());
  for (const Symbol &s : original_inputs) {
    uint32_t w = orig_to_working[s.node_index];
    // An input that was optimised away (w == UINT32_MAX) or that resolves to
    // an unreachable node will have working_to_prog == UINT32_MAX.  We store
    // UINT32_MAX and let the caller decide whether that is an error.
    uint32_t p = (w != UINT32_MAX) ? working_to_prog[resolve(w)] : UINT32_MAX;
    prog.input_nodes.push_back(p);
  }

  // ── Map output symbols → prog indices ───────────────────────────────────
  prog.output_nodes.reserve(working_outputs.size());
  for (uint32_t out_w : working_outputs)
    prog.output_nodes.push_back(working_to_prog[resolve(out_w)]);

  return prog;
}

} // namespace autograd
