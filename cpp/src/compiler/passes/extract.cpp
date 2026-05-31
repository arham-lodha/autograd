#include "autograd/compiler.hpp"
#include "autograd/graph.hpp"
#include "autograd/ir.hpp"
#include "autograd/ops.hpp"
#include "autograd/symbol.hpp"
#include <cstdint>
#include <stack>
#include <stdexcept>

namespace autograd {

// ── Utility helpers ────────────────────────────────────────────────────────

uint32_t Compiler::resolve(uint32_t index) {
  uint32_t root = index;
  while (working_nodes[root].operation == Op::ALIAS)
    root = working_nodes[root].inputs[0];

  while (working_nodes[index].operation == Op::ALIAS) {
    uint32_t next = working_nodes[index].inputs[0];
    working_nodes[index].inputs[0] = root;
    index = next;
  }

  return index;
}

uint32_t Compiler::get_input(uint32_t working_idx, uint32_t i) const {
  const Node &n = working_nodes[working_idx];
  if (i < 2)
    return n.inputs[i];
  return working_inputs[n.input_pool_offset + (i - 2)];
}

// ── Phase 1: Subgraph extraction ──────────────────────────────────────────
//
// Iterative post-order DFS over the original graph starting from `outputs`.
// Only reachable nodes are copied; all input indices are remapped from
// original-graph coordinates to working-buffer coordinates.

void Compiler::extract_subgraph(const std::vector<Symbol> &outputs) {
  const std::size_t n_orig = original_graph.nodes.size();
  orig_to_working.assign(n_orig, UINT32_MAX);

  // state: 0 = unvisited, 1 = children pushed, 2 = emitted
  std::vector<uint8_t> state(n_orig, 0);
  std::stack<uint32_t> stack;

  for (const Symbol &s : outputs) {
    if (s.node_index >= n_orig)
      throw std::runtime_error("extract_subgraph: output symbol out of range");
    if (state[s.node_index] == 0)
      stack.push(s.node_index);
  }

  while (!stack.empty()) {
    uint32_t idx = stack.top();
    const Node &node = original_graph.nodes[idx];

    if (state[idx] == 0) {
      // First visit: mark and push children so they are processed first.
      state[idx] = 1;
      for (uint32_t i = 0; i < node.input_count; i++) {
        uint32_t child =
            i < 2 ? node.inputs[i]
                  : original_graph.inputs[node.input_pool_offset + (i - 2)];
        if (state[child] == 0)
          stack.push(child);
        // Cycle detection is the graph builder's responsibility.
      }
    } else if (state[idx] == 1) {
      // All children are already emitted — emit this node now.
      state[idx] = 2;
      stack.pop();

      Node n = node; // copy; we will mutate input indices
      n.input_pool_offset = static_cast<uint32_t>(working_inputs.size());

      for (uint32_t i = 0; i < n.input_count; i++) {
        uint32_t orig_child =
            i < 2 ? node.inputs[i]
                  : original_graph.inputs[node.input_pool_offset + (i - 2)];

        uint32_t mapped = orig_to_working[orig_child];
        if (mapped == UINT32_MAX)
          throw std::runtime_error(
              "extract_subgraph: child node was not emitted before parent");

        if (i < 2)
          n.inputs[i] = mapped;
        else
          working_inputs.push_back(mapped);
      }

      // Copy constant value into the working arena.
      if (n.operation == Op::CONSTANT) {
        n.value_index = static_cast<uint32_t>(working_values.size());
        working_values.push_back(original_graph.values[node.value_index]);
      }

      // Copy shape slice for RESHAPE / BROADCAST_TO.
      if (n.operation == Op::RESHAPE || n.operation == Op::BROADCAST_TO) {
        uint32_t old_offset = node.shape.offset;
        n.shape.offset = static_cast<uint32_t>(working_shapes.size());
        for (uint32_t i = 0; i < node.shape.count; i++)
          working_shapes.push_back(original_graph.shapes[old_offset + i]);
      }

      orig_to_working[idx] = static_cast<uint32_t>(working_nodes.size());
      working_nodes.push_back(n);

    } else {
      // state == 2: already emitted via another path (diamond dependency).
      stack.pop();
    }
  }

  working_outputs.reserve(outputs.size());
  for (const Symbol &s : outputs)
    working_outputs.push_back(orig_to_working[s.node_index]);
}

} // namespace autograd
