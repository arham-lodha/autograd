#include "autograd/compiler.hpp"
#include "autograd/symbol.hpp"

namespace autograd {

std::vector<uint32_t> Compiler::topological_sort(const Symbol &output) {
  std::vector<uint32_t> sorted;
  std::unordered_map<uint32_t, bool> visited;
  std::stack<uint32_t> stack;
  const Graph *graph = output.graph;
  stack.push(output.node_index);

  while (!stack.empty()) {
    uint32_t node_id = stack.top();

    if (visited[node_id]) {
      stack.pop();
      continue;
    }

    bool unvisited_inputs = false;

    for (uint32_t i = 0; i < graph->nodes[node_id].input_count; i++) {
      uint32_t input_id = graph->nodes[node_id].input_offset + i;
      if (!visited[input_id]) {
        stack.push(input_id);
        unvisited_inputs = true;
      }
    }

    if (!unvisited_inputs) {
      visited[node_id] = true;
      sorted.push_back(node_id);
      stack.pop();
    }
  }

  return sorted;
}

Program Compiler::compile(const Symbol &output, uint optimization_passes,
                          bool skip_optimization) {
  Program program;
  const Graph *graph = output.graph;
  std::vector<uint32_t> sorted_nodes = topological_sort(output);

  if (!skip_optimization) {
    IntermediateProgram intermediate_program;
    intermediate_program.nodes = sorted_nodes;

    for (uint i = 0; i < optimization_passes; i++) {
      intermediate_program = optimize(*graph, intermediate_program);
      if (!intermediate_program.changed) {
        break;
      }
    }

    intermediate_program = decanonicalize(*graph, intermediate_program);

    sorted_nodes = intermediate_program.nodes;
  }

  // populate Program from sorted nodes and graph

  return program;
}

IntermediateProgram Compiler::optimize(const Graph &graph,
                                       const IntermediateProgram &program) {
  IntermediateProgram optimized_program = program;
  optimized_program.changed = false;

  // Apply optimization passes in sequence
  optimized_program = canonicalize(graph, optimized_program);
  optimized_program = addition_folding(graph, optimized_program);
  optimized_program = multiplication_folding(graph, optimized_program);
  optimized_program = constant_fold(graph, optimized_program);
  optimized_program = algebraic_simplification(graph, optimized_program);
  optimized_program = dead_code_elimination(graph, optimized_program);

  return optimized_program;
}

IntermediateProgram Compiler::canonicalize(const Graph &graph,
                                           const IntermediateProgram &program) {
  IntermediateProgram canonical_program = program;
  canonical_program.changed = false;

  // Implement canonicalization logic here

  for (uint32_t node_id : program.nodes) {
    const Node &node = graph.nodes[node_id];
    // Example: canonicalize addition to a specific order
    if (node.operation == Op::NEG) {
      // Replace Neg with Multiply by -1
      const Symbol neg_one = graph.make_constant(-1.0f);
      node.operation = Op::MULTIPLY;
    }
  }

  return canonical_program;
}

} // namespace autograd
