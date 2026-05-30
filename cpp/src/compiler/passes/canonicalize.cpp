#include "autograd/compiler.hpp"
#include "autograd/ir.hpp"
#include "autograd/ops.hpp"
#include <cstdint>

namespace autograd {

// ── canonicalize ──────────────────────────────────────────────────────────────
//
// Rewrites sugar ops into primitives so later passes only need to handle a
// smaller set of operations.
//
//   NEG(x)    → MULTIPLY(x, -1)          (in-place: node becomes MULTIPLY)
//   SUB(a,b)  → ADD(a, MULTIPLY(b, -1))  (append MULTIPLY; node becomes ADD)
//   SQRT(x)   → POWER(x, 0.5)            (in-place: node becomes POWER)
//
// IMPORTANT: ensure_well_known() may call working_nodes.push_back(), which
// can reallocate the vector.  We must never hold a Node& reference across any
// call that appends to working_nodes.  The pattern used here is:
//   1. Read all necessary values (inputs, flags) by index.
//   2. Call ensure_well_known (may reallocate).
//   3. Write back to working_nodes[ri] by index only.

bool Compiler::canonicalize() {
  bool changed = false;
  const uint32_t n = static_cast<uint32_t>(working_nodes.size());

  for (uint32_t i = 0; i < n; i++) {
    uint32_t ri = resolve(i);

    // Read operation by index — do NOT hold a reference across push_back calls.
    switch (working_nodes[ri].operation) {

    case Op::NEG: {
      // Step 1: read inputs before any potential reallocation.
      uint32_t x = resolve(working_nodes[ri].inputs[0]);
      // Step 2: ensure_well_known may push_back → realloc.
      uint32_t neg_one = ensure_well_known(K_NEG_ONE);
      // Step 3: write back by index (safe after realloc).
      working_nodes[ri].operation   = Op::MULTIPLY;
      working_nodes[ri].input_count = 2;
      working_nodes[ri].inputs[0]   = x;
      working_nodes[ri].inputs[1]   = neg_one;
      changed = true;
      break;
    }

    case Op::SUB: {
      // Step 1: read inputs.
      uint32_t a   = resolve(working_nodes[ri].inputs[0]);
      uint32_t b   = resolve(working_nodes[ri].inputs[1]);
      bool rg      = working_nodes[ri].requires_grad;
      bool ret     = working_nodes[ri].retain;
      // Step 2: ensure_well_known may realloc.
      uint32_t neg_one = ensure_well_known(K_NEG_ONE);

      // Append MULTIPLY(b, -1).  This push_back may also realloc.
      Node mul{};
      mul.operation   = Op::MULTIPLY;
      mul.input_count = 2;
      mul.inputs[0]   = b;
      mul.inputs[1]   = neg_one;
      mul.requires_grad = working_nodes[b].requires_grad;
      uint32_t mul_idx = static_cast<uint32_t>(working_nodes.size());
      working_nodes.push_back(mul);

      // Step 3: re-index ri after all reallocations.
      working_nodes[ri].operation   = Op::ADD;
      working_nodes[ri].input_count = 2;
      working_nodes[ri].inputs[0]   = a;
      working_nodes[ri].inputs[1]   = mul_idx;
      working_nodes[ri].requires_grad = rg;
      working_nodes[ri].retain        = ret;
      changed = true;
      break;
    }

    case Op::SQRT: {
      // Step 1: read input.
      uint32_t x = resolve(working_nodes[ri].inputs[0]);
      // Step 2: ensure_well_known may realloc.
      uint32_t half = ensure_well_known(K_HALF);
      // Step 3: write back by index.
      working_nodes[ri].operation   = Op::POWER;
      working_nodes[ri].input_count = 2;
      working_nodes[ri].inputs[0]   = x;
      working_nodes[ri].inputs[1]   = half;
      changed = true;
      break;
    }

    default:
      break;
    }
  }

  return changed;
}

} // namespace autograd
