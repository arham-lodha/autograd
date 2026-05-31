#include "autograd/compiler.hpp"
#include "autograd/graph.hpp"
#include "autograd/ir.hpp"
#include "autograd/ops.hpp"
#include "autograd/symbol.hpp"
#include "autograd/tensor.hpp"
#include <cstdint>

namespace autograd {

Compiler::Compiler(const Graph &graph, CompilerConfig cfg)
    : original_graph(graph), config(cfg) {
  well_known_slots.fill(UINT32_MAX);
}

Program Compiler::compile(const Graph &graph, const std::vector<Symbol> &inputs,
                          const std::vector<Symbol> &outputs,
                          CompilerConfig config) {
  Compiler c(graph, config);
  c.extract_subgraph(outputs);
  c.run_optimization_loop();
  return c.finalize_and_sort(inputs);
}

void Compiler::run_optimization_loop() {
  // These are all heuristics:
  // canonical basically adds at most 2 nodes,
  working_values.reserve(working_values.size() * 2);
  working_inputs.reserve(working_inputs.size() * 2);
  working_values.reserve(K_COUNT + working_values.size() + 8);

  for (int i = 0; i < config.optimization_passes; i++) {
    bool changed = false;
    changed |= canonicalize();
    changed |= fold_addition_multiplication();
    if (config.enable_constant_folding)
      changed |= fold_constants();
    changed |= algebraic_simplification();
    if (!changed)
      break;
  }
}

// ── Shared utilities
// ──────────────────────────────────────────────────────────

uint32_t Compiler::ensure_well_known(uint32_t slot) {
  if (well_known_slots[slot] != UINT32_MAX)
    return well_known_slots[slot];

  constexpr float kValues[K_COUNT] = {1.0f, 0.0f, -1.0f, 0.5f};
  uint32_t val_idx = static_cast<uint32_t>(working_values.size());
  working_values.push_back(Tensor(kValues[slot]));

  Node c{};
  c.operation = Op::CONSTANT;
  c.value_index = val_idx;

  uint32_t node_idx = static_cast<uint32_t>(working_nodes.size());
  working_nodes.push_back(c);
  well_known_slots[slot] = node_idx;
  return node_idx;
}

bool Compiler::is_scalar_const(uint32_t idx, float expected) {
  idx = resolve(idx);
  const Node &n = working_nodes[idx];
  if (n.operation != Op::CONSTANT)
    return false;
  const Tensor &t = working_values[n.value_index];
  return t.is_scalar() && t(0, 0) == expected;
}

void Compiler::alias(uint32_t target, uint32_t source) {
  uint32_t real = resolve(source);
  Node &t = working_nodes[target];
  t = Node{};
  t.operation = Op::ALIAS;
  t.inputs[0] = real;
  t.input_count = 1;
}

} // namespace autograd
