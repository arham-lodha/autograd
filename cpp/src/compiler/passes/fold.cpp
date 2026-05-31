#include "autograd/compiler.hpp"
#include "autograd/ir.hpp"
#include "autograd/ops.hpp"
#include "autograd/tensor.hpp"
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace autograd {

// ── fold_addition_multiplication ─────────────────────────────────────────────
//
// Flattens chains of the same associative op into a single n-ary node.
// Example: ADD(ADD(a, b), c) → ADD(a, b, c)
//
// Strategy (in-place, no second buffer):
//   For each ADD or MULTIPLY node at index i:
//     DFS backward through inputs, absorbing any input that is itself the same
//     op and is not marked `retain`.  Collect the true leaf inputs.
//     If the leaf set differs from the current inputs:
//       1. Append a new flat node (children in inline slots + working_inputs).
//       2. ALIAS node i → the new flat node.
//     Absorbed intermediate nodes become unreachable; Phase 3 drops them.
//
// Passes run left-to-right in working-index order (topological: children have
// lower indices than parents after extract_subgraph's post-order emit).
// By the time we process a parent ADD, any child ADD is already aliased;
// resolve() chases those links so the DFS reaches the real flat children.

bool Compiler::fold_addition_multiplication() {
  bool changed = false;
  const uint32_t n = static_cast<uint32_t>(working_nodes.size());

  std::vector<uint32_t> leaves;
  std::vector<uint32_t> stk;
  std::unordered_set<uint32_t> pushed;

  for (uint32_t i = 0; i < n; i++) {
    uint32_t ri = resolve(i);
    if (ri != i) continue; // already aliased away

    const Node &node = working_nodes[ri];
    if (node.operation != Op::ADD && node.operation != Op::MULTIPLY)
      continue;

    Op target = node.operation;

    // ── DFS to collect flat leaves ──────────────────────────────────────────
    leaves.clear();
    stk.clear();
    pushed.clear();

    for (uint32_t j = 0; j < node.input_count; j++) {
      uint32_t inp = resolve(get_input(ri, j));
      if (pushed.insert(inp).second) stk.push_back(inp);
    }

    while (!stk.empty()) {
      uint32_t cur = stk.back(); stk.pop_back();
      const Node &cn = working_nodes[cur];
      if (cn.operation == target && !cn.retain) {
        // Absorb: push its inputs rather than recording it as a leaf.
        for (uint32_t j = 0; j < cn.input_count; j++) {
          uint32_t inp = resolve(get_input(cur, j));
          if (pushed.insert(inp).second) stk.push_back(inp);
        }
      } else {
        leaves.push_back(cur);
      }
    }

    if (leaves.size() == node.input_count) continue; // nothing to flatten

    // ── Append a new flat node, then alias i → it ───────────────────────────
    // Capture fields before push_back potentially reallocates.
    bool rg  = working_nodes[ri].requires_grad;
    bool ret = working_nodes[ri].retain;

    Node flat{};
    flat.operation         = target;
    flat.input_count       = static_cast<uint16_t>(leaves.size());
    flat.input_pool_offset = static_cast<uint32_t>(working_inputs.size());
    flat.requires_grad     = rg;
    flat.retain            = ret;

    for (uint32_t j = 0; j < static_cast<uint32_t>(leaves.size()); j++) {
      if (j < 2) flat.inputs[j] = leaves[j];
      else       working_inputs.push_back(leaves[j]);
    }

    uint32_t flat_idx = static_cast<uint32_t>(working_nodes.size());
    working_nodes.push_back(flat);
    alias(ri, flat_idx);
    changed = true;
  }

  return changed;
}

// ── fold_constants ────────────────────────────────────────────────────────────
//
// Evaluates ops whose every input is a CONSTANT at compile time, replacing
// the op node with a CONSTANT containing the pre-computed result.
//
// Handles ADD, MULTIPLY, POWER, EXP, LOG where all inputs are scalar (1×1)
// constants.  The pattern generalises to full matrices but runtime shape
// information is not always available at compile time for shape-dependent ops,
// so scope is intentionally narrow.

bool Compiler::fold_constants() {
  bool changed = false;
  const uint32_t n = static_cast<uint32_t>(working_nodes.size());

  for (uint32_t i = 0; i < n; i++) {
    uint32_t ri = resolve(i);
    if (ri != i) continue;

    const Node &node = working_nodes[ri];
    if (node.operation == Op::CONSTANT  ||
        node.operation == Op::VARIABLE  ||
        node.operation == Op::PLACEHOLDER ||
        node.operation == Op::ALIAS)
      continue;

    // Collect resolved inputs and verify they are all CONSTANTs.
    std::vector<uint32_t> inp;
    inp.reserve(node.input_count);
    for (uint32_t j = 0; j < node.input_count; j++)
      inp.push_back(resolve(get_input(ri, j)));

    bool all_const = true;
    for (uint32_t idx : inp)
      if (working_nodes[idx].operation != Op::CONSTANT) { all_const = false; break; }
    if (!all_const) continue;

    // Only fold scalar (1×1) constants for now.
    bool all_scalar = true;
    for (uint32_t idx : inp) {
      if (!working_values[working_nodes[idx].value_index].is_scalar()) {
        all_scalar = false; break;
      }
    }
    if (!all_scalar) continue;

    auto val = [&](uint32_t idx) -> float {
      return working_values[working_nodes[idx].value_index](0, 0);
    };

    float result   = 0.0f;
    bool  computed = true;

    switch (node.operation) {
    case Op::ADD: {
      result = 0.0f;
      for (uint32_t idx : inp) result += val(idx);
      break;
    }
    case Op::MULTIPLY: {
      result = 1.0f;
      for (uint32_t idx : inp) result *= val(idx);
      break;
    }
    case Op::POWER:
      if (inp.size() == 2) result = std::pow(val(inp[0]), val(inp[1]));
      else computed = false;
      break;
    case Op::EXP:
      if (inp.size() == 1) result = std::exp(val(inp[0]));
      else computed = false;
      break;
    case Op::LOG:
      if (inp.size() == 1) result = std::log(val(inp[0]));
      else computed = false;
      break;
    default:
      computed = false;
      break;
    }

    if (!computed) continue;

    // Reuse a well-known slot if the value matches; otherwise append fresh.
    uint32_t const_node = UINT32_MAX;
    constexpr float kWK[K_COUNT] = {1.0f, 0.0f, -1.0f, 0.5f};
    for (uint32_t s = 0; s < K_COUNT; s++) {
      if (result == kWK[s]) { const_node = ensure_well_known(s); break; }
    }

    if (const_node == UINT32_MAX) {
      uint32_t val_idx = static_cast<uint32_t>(working_values.size());
      working_values.push_back(Tensor(result));

      Node c{};
      c.operation   = Op::CONSTANT;
      c.value_index = val_idx;
      const_node = static_cast<uint32_t>(working_nodes.size());
      working_nodes.push_back(c);
    }

    alias(ri, const_node);
    changed = true;
  }

  return changed;
}

} // namespace autograd
