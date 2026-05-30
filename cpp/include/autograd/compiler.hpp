#pragma once

#include "Eigen/Dense"
#include "autograd/graph.hpp"
#include "autograd/ir.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace autograd {

// Well-known scalar constants pre-populated into working_values on demand.
enum : uint32_t { K_ONE = 0, K_ZERO = 1, K_NEG_ONE = 2, K_HALF = 3, K_COUNT = 4 };

struct CompilerConfig {
  int optimization_passes = 10;
  bool enable_constant_folding = true;
  bool debug_mode = false;
};

class Compiler {
public:
  static Program compile(const Graph &graph, const std::vector<Symbol> &inputs,
                         const std::vector<Symbol> &outputs,
                         CompilerConfig config = {});

private:
  Compiler(const Graph &graph, CompilerConfig config);

  // ── Persistent references ──────────────────────────────────────────────────
  const Graph &original_graph;
  CompilerConfig config;

  // ── Phase 1 / Phase 2 working buffers ────────────────────────────────────
  // All indices inside working_nodes refer to positions within this array.
  // Passes append new nodes via push_back; they never remove or reorder.
  std::vector<Node>            working_nodes;
  std::vector<uint32_t>        working_inputs;  // CSR pool for n-ary ops (>2 inputs)
  std::vector<Eigen::MatrixXf> working_values;  // constants, including well-knowns
  std::vector<uint32_t>        working_shapes;  // CSR shape data for RESHAPE/BROADCAST_TO

  // Indices into working_nodes for the requested outputs.
  std::vector<uint32_t> working_outputs;

  // Mapping from original_graph node index → working_nodes index.
  // Kept as a member so Phase 3 can look up input symbols.
  std::vector<uint32_t> orig_to_working;

  // Lazily-created well-known constant nodes; UINT32_MAX = not yet created.
  std::array<uint32_t, K_COUNT> well_known_slots;

  // ── Phase 1 ───────────────────────────────────────────────────────────────
  void extract_subgraph(const std::vector<Symbol> &outputs);

  // ── Phase 2 ───────────────────────────────────────────────────────────────
  // Each pass mutates working_nodes in place.  Redundant nodes are replaced
  // with Op::ALIAS entries; resolve() chases those links transparently.
  void run_optimization_loop();
  bool canonicalize();
  bool fold_addition_multiplication();
  bool fold_constants();
  bool algebraic_simplification();

  // ── Phase 3 ───────────────────────────────────────────────────────────────
  Program finalize_and_sort(const std::vector<Symbol> &original_inputs);

  // ── Utilities ─────────────────────────────────────────────────────────────

  // Chase Op::ALIAS links; returns the index of the real underlying node.
  uint32_t resolve(uint32_t index) const;

  // Return the working-space input index for slot i of node at working_idx.
  uint32_t get_input(uint32_t working_idx, uint32_t i) const;

  // Ensure a well-known scalar constant node exists; return its working index.
  // Values: K_ONE=1.0, K_ZERO=0.0, K_NEG_ONE=-1.0, K_HALF=0.5
  uint32_t ensure_well_known(uint32_t slot);

  // True iff working_nodes[resolve(idx)] is a CONSTANT whose value is a 1×1
  // matrix equal to `expected`.
  bool is_scalar_const(uint32_t idx, float expected) const;

  // Alias working_nodes[target] to working_nodes[resolve(source)].
  // Sets target's operation to ALIAS and inputs[0] to the resolved source.
  void alias(uint32_t target, uint32_t source);
};

} // namespace autograd
