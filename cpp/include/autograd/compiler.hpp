#pragma once
#include "autograd/graph.hpp"
#include "autograd/ops.hpp"
#include "autograd/symbol.hpp"
#include <cstdint>
#include <unordered_map>

namespace autograd {

struct ProgramNode {
  Op op;
  uint32_t inputs_offset = 0;
  uint32_t inputs_count = 0;
  uint32_t value_index = UINT32_MAX; // CONSTANT nodes only
  uint32_t axis = UINT32_MAX;        // GET_ITEM / SCATTER_LIKE only
  uint32_t axis2 = UINT32_MAX;       // swap_axis only
  bool keepdims = false;             // REDUCE_* only
};

struct Program {
  std::vector<ProgramNode> nodes;
  std::vector<uint32_t> inputs;        // flat CSR input-index list
  std::vector<Eigen::MatrixXf> values; // constant values
};

struct IntermediateProgram {
  std::vector<uint32_t> nodes;
  bool changed = false;
};

class Compiler {
public:
  Compiler() = default;

  Program compile(const Symbol &output, uint optimization_passes = 10,
                  bool skip_optimization = false);

  IntermediateProgram optimize(const Graph &graph,
                               const IntermediateProgram &program);
  IntermediateProgram constant_fold(const Graph &graph,
                                    const IntermediateProgram &program);
  IntermediateProgram addition_folding(const Graph &graph,
                                       const IntermediateProgram &program);
  IntermediateProgram
  multiplication_folding(const Graph &graph,
                         const IntermediateProgram &program);
  IntermediateProgram
  algebraic_simplification(const Graph &graph,
                           const IntermediateProgram &program);
  IntermediateProgram dead_code_elimination(const Graph &graph,
                                            const IntermediateProgram &program);
  IntermediateProgram canonicalize(const Graph &graph,
                                   const IntermediateProgram &program);
  IntermediateProgram decanonicalize(const Graph &graph,
                                     const IntermediateProgram &program);

  std::vector<uint32_t> topological_sort(const Symbol &output);

  Program compile_backwards(const Symbol &output);

private:
  Symbol make_constant(Graph &g, float value);
  Symbol accumulate(const Symbol &existing, const Symbol &grad);
  void backwards_node(Graph &g, uint32_t node_idx,
                      std::unordered_map<uint32_t, Symbol> &grads);
  bool try_replace(uint32_t node_idx, const Symbol &replacement,
                   std::unordered_map<uint32_t, Symbol> &replacements,
                   IntermediateProgram &prog);
};

} // namespace autograd
