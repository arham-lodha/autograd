#pragma once
#include "autograd/graph.hpp"
#include "autograd/ops.hpp"
#include "autograd/symbol.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace autograd {

// ── Frozen executable form (CSR layout, consumed by Executor) ────────────────

struct ProgramNode {
  Op op;
  uint32_t inputs_offset = 0;
  uint32_t inputs_count = 0;
  uint32_t value_index = UINT32_MAX;
  uint32_t axis = UINT32_MAX;
  uint32_t axis2 = UINT32_MAX;
  bool keepdims = false;
  bool requires_grad = false;
};

struct Program {
  std::vector<ProgramNode> nodes;
  std::vector<uint32_t> inputs;        // flat CSR input-index list
  std::vector<Eigen::MatrixXf> values; // constant pool
  std::unordered_map<uint32_t, Symbol>
      grads; // gradient symbols for each node (if requires_grad)
};

// ── Mutable IR (used by compiler optimization passes) ────────────────────────
//
// inputs is a plain vector, so passes can freely add, remove, or rewrite edges
// without touching CSR offsets.  nodes and values are self-contained; no Graph
// pointer is needed once the IR is built.

struct IntermediateProgramNode {
  Op op;
  std::vector<uint32_t> inputs;
  uint32_t value_index = UINT32_MAX;
  uint32_t axis = UINT32_MAX;
  uint32_t axis2 = UINT32_MAX;
  bool keepdims = false;
  bool retain = false;
  bool requires_grad = false;
};

struct IntermediateProgram {
  std::vector<IntermediateProgramNode> nodes; // topological order
  std::vector<Eigen::MatrixXf> values;        // constant pool
  uint32_t output_idx = 0; // which node is the graph output (root)
  bool changed = false;
};

// ── Compiler
// ──────────────────────────────────────────────────────────────────

class Compiler {
public:
  Compiler() = default;

  Program compile(const Symbol &output, bool backwards = true,
                  uint optimization_passes = 10,
                  bool skip_optimization = false);

  // Bookend steps: Graph → IR and IR → Program
  IntermediateProgram make_ir(const Graph &graph,
                              const std::vector<uint32_t> &topo);
  Program freeze(const IntermediateProgram &prog);

  // Optimization pipeline (each pass takes and returns IR by value)
  IntermediateProgram optimize(IntermediateProgram prog);
  IntermediateProgram canonicalize(IntermediateProgram prog);
  IntermediateProgram decanonicalize(IntermediateProgram prog);
  IntermediateProgram constant_fold(IntermediateProgram prog);
  IntermediateProgram addition_folding(IntermediateProgram prog);
  IntermediateProgram multiplication_folding(IntermediateProgram prog);
  IntermediateProgram algebraic_simplification(IntermediateProgram prog);
  IntermediateProgram dead_code_elimination(IntermediateProgram prog);

  std::vector<uint32_t> topological_sort(const Symbol &output);

  Program compile_backwards(const Symbol &output);

private:
  Symbol make_constant(Graph &g, float value);
  Symbol accumulate(const Symbol &existing, const Symbol &grad);
  void backwards_node(Graph &g, uint32_t node_idx,
                      std::unordered_map<uint32_t, Symbol> &grads);
};

} // namespace autograd
