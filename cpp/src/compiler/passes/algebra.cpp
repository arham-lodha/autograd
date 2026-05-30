#include "autograd/compiler.hpp"
#include "autograd/ir.hpp"
#include "autograd/ops.hpp"
#include <cstdint>
#include <vector>

namespace autograd {

// ── algebraic_simplification ──────────────────────────────────────────────────
//
// Pattern-matches algebraic identities and replaces the node with an ALIAS to
// the surviving operand (or a well-known constant).  All input reads go through
// resolve() so ALIAS chains from earlier passes are transparent.
//
// Rules applied (after canonicalize has eliminated NEG/SUB/SQRT):
//
//   ADD:      x + 0  → x          (any order among n-ary inputs)
//             0 + 0  → 0          (all-zero add)
//   MULTIPLY: x * 1  → x
//             x * 0  → 0
//             1 * 1  → 1          (all-one multiply)
//   POWER:    x ^ 1  → x
//             x ^ 0  → 1
//   EXP/LOG:  log(exp(x)) → x
//             exp(log(x)) → x     (valid for x > 0; kept as an opt-in rule)
//
// When a rule eliminates the node, it becomes Op::ALIAS pointing at the
// surviving child or at a well-known constant node.

bool Compiler::algebraic_simplification() {
  bool changed = false;
  const uint32_t n = static_cast<uint32_t>(working_nodes.size());

  for (uint32_t i = 0; i < n; i++) {
    uint32_t ri = resolve(i);
    if (ri != i) continue; // already aliased

    Node &node = working_nodes[ri];

    switch (node.operation) {

    // ── ADD ──────────────────────────────────────────────────────────────────
    case Op::ADD: {
      // Collect non-zero inputs.
      std::vector<uint32_t> non_zero;
      for (uint32_t j = 0; j < node.input_count; j++) {
        uint32_t inp = resolve(get_input(ri, j));
        if (!is_scalar_const(inp, 0.0f))
          non_zero.push_back(inp);
      }

      if (non_zero.size() == node.input_count) break; // nothing to simplify

      if (non_zero.empty()) {
        // 0 + 0 + ... → 0
        alias(ri, ensure_well_known(K_ZERO));
      } else if (non_zero.size() == 1) {
        // x + 0 → x
        alias(ri, non_zero[0]);
      } else {
        // Shrink the n-ary node: append a new ADD with only the surviving inputs,
        // then alias old node to it.
        bool rg  = node.requires_grad;
        bool ret = node.retain;

        Node flat{};
        flat.operation         = Op::ADD;
        flat.input_count       = static_cast<uint16_t>(non_zero.size());
        flat.input_pool_offset = static_cast<uint32_t>(working_inputs.size());
        flat.requires_grad     = rg;
        flat.retain            = ret;

        for (uint32_t j = 0; j < static_cast<uint32_t>(non_zero.size()); j++) {
          if (j < 2) flat.inputs[j] = non_zero[j];
          else       working_inputs.push_back(non_zero[j]);
        }

        uint32_t flat_idx = static_cast<uint32_t>(working_nodes.size());
        working_nodes.push_back(flat);
        alias(ri, flat_idx);
      }
      changed = true;
      break;
    }

    // ── MULTIPLY ─────────────────────────────────────────────────────────────
    case Op::MULTIPLY: {
      // If any input is 0, the whole product is 0.
      for (uint32_t j = 0; j < node.input_count; j++) {
        uint32_t inp = resolve(get_input(ri, j));
        if (is_scalar_const(inp, 0.0f)) {
          alias(ri, ensure_well_known(K_ZERO));
          changed = true;
          goto next_node;
        }
      }

      {
        // Collect non-one inputs.
        std::vector<uint32_t> non_one;
        for (uint32_t j = 0; j < node.input_count; j++) {
          uint32_t inp = resolve(get_input(ri, j));
          if (!is_scalar_const(inp, 1.0f))
            non_one.push_back(inp);
        }

        if (non_one.size() == node.input_count) break; // nothing to simplify

        if (non_one.empty()) {
          // 1 * 1 * ... → 1
          alias(ri, ensure_well_known(K_ONE));
        } else if (non_one.size() == 1) {
          // x * 1 → x
          alias(ri, non_one[0]);
        } else {
          bool rg  = node.requires_grad;
          bool ret = node.retain;

          Node flat{};
          flat.operation         = Op::MULTIPLY;
          flat.input_count       = static_cast<uint16_t>(non_one.size());
          flat.input_pool_offset = static_cast<uint32_t>(working_inputs.size());
          flat.requires_grad     = rg;
          flat.retain            = ret;

          for (uint32_t j = 0; j < static_cast<uint32_t>(non_one.size()); j++) {
            if (j < 2) flat.inputs[j] = non_one[j];
            else       working_inputs.push_back(non_one[j]);
          }

          uint32_t flat_idx = static_cast<uint32_t>(working_nodes.size());
          working_nodes.push_back(flat);
          alias(ri, flat_idx);
        }
        changed = true;
      }
      break;
    }

    // ── POWER ─────────────────────────────────────────────────────────────────
    case Op::POWER: {
      if (node.input_count != 2) break;
      uint32_t base = resolve(node.inputs[0]);
      uint32_t exp  = resolve(node.inputs[1]);

      if (is_scalar_const(exp, 0.0f)) {
        // x^0 → 1
        alias(ri, ensure_well_known(K_ONE));
        changed = true;
      } else if (is_scalar_const(exp, 1.0f)) {
        // x^1 → x
        alias(ri, base);
        changed = true;
      }
      break;
    }

    // ── LOG(EXP(x)) → x ───────────────────────────────────────────────────────
    case Op::LOG: {
      if (node.input_count != 1) break;
      uint32_t inp = resolve(node.inputs[0]);
      if (working_nodes[inp].operation == Op::EXP) {
        uint32_t x = resolve(working_nodes[inp].inputs[0]);
        alias(ri, x);
        changed = true;
      }
      break;
    }

    // ── EXP(LOG(x)) → x ───────────────────────────────────────────────────────
    case Op::EXP: {
      if (node.input_count != 1) break;
      uint32_t inp = resolve(node.inputs[0]);
      if (working_nodes[inp].operation == Op::LOG) {
        uint32_t x = resolve(working_nodes[inp].inputs[0]);
        alias(ri, x);
        changed = true;
      }
      break;
    }

    default:
      break;
    }

    next_node:;
  }

  return changed;
}

} // namespace autograd
