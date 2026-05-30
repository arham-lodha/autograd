#include "autograd/graph.hpp"
#include "autograd/symbol.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace autograd {

std::vector<uint32_t>
Graph::topological_sort(const std::vector<Symbol> &output) {
  std::vector<uint32_t> sorted_nodes;
  std::unordered_map<uint32_t, int>
      state; // 0 = unvisited, 1 = visiting, 2 = visited

  std::stack<uint32_t> stack;

  for (const Symbol symbol : output) {
    uint32_t node_index = symbol.node_index;

    if (state[node_index] == 2)
      continue; // already visited

    stack.push(node_index);

    while (!stack.empty()) {
      uint32_t current = stack.top();

      if (state[current] == 0) {
        // First time seeing this node, mark as visiting
        state[current] = 1;

        for (uint32_t i = 0; i < nodes[current].input_count; ++i) {
          uint32_t input_node =
              i < 2 ? nodes[current].inputs[i]
                    : this->inputs[nodes[current].input_pool_offset + (i - 2)];
          if (state[input_node] == 0) {
            stack.push(input_node);
          } else if (state[input_node] == 1) {
            throw std::runtime_error("Graph contains a cycle");
          }
        }
      } else if (state[current] == 1) {
        state[current] = 2; // mark as visited
        sorted_nodes.push_back(current);
        stack.pop();
      } else {
        stack.pop(); // already visited, just pop
      }
    }
  }

  return sorted_nodes;
}

Program Graph::compile(const std::vector<Symbol> &inputs,
                       const std::vector<Symbol> &output,
                       int optimization_passes) {
  std::vector<bool> is_provided_input(nodes.size(), false);
  for (const Symbol &input : inputs) {
    if (input.node_index >= nodes.size()) {
      throw std::runtime_error("Input symbol node index out of bounds");
    }
    is_provided_input[input.node_index] = true;
  }

  std::vector<uint32_t> sorted_nodes = topological_sort(output);

  Program program;
  program.nodes.reserve(sorted_nodes.size() *
                        1.5); // heuristic to reduce reallocations

  std::vector<uint32_t> old_to_new_index(nodes.size() * 1.5, UINT32_MAX);

  // Copy nodes in topologically sorted order, remapping input indices and
  // pooling
  for (uint32_t old_idx : sorted_nodes) {
    Node n{this->nodes[old_idx]}; // copy node to modify in place

    if (n.operation == Op::PLACEHOLDER || n.operation == Op::VARIABLE) {
      if (!is_provided_input[old_idx]) {
        throw std::runtime_error(
            "PLACEHOLDER or VARIABLE node in graph is not provided as input");
      }
    }

    n.input_pool_offset =
        program.inputs.size(); // set input pool offset for n-ary ops

    for (uint32_t i = 0; i < n.input_count; i++) {
      if (i < 2) {
        n.inputs[i] = old_to_new_index[n.inputs[i]];
      } else {
        program.inputs.push_back(
            old_to_new_index[this->inputs[n.input_pool_offset + (i - 2)]]);
      }
    }

    if (n.operation == Op::CONSTANT) {
      program.values.push_back(this->values[this->nodes[old_idx].value_index]);
      n.value_index = program.values.size() - 1;
    }

    if (n.operation == Op::RESHAPE || n.operation == Op::BROADCAST_TO) {
      uint32_t old_shape_offset = n.shape.offset;
      n.shape.offset = program.shapes.size();

      for (uint32_t i = 0; i < n.shape.count; i++) {
        program.shapes.push_back(this->shapes[old_shape_offset + i]);
      }
    }

    program.nodes.push_back(n);
    old_to_new_index[old_idx] =
        program.nodes.size() - 1; // map old index to new
  }

  program.input_nodes.reserve(inputs.size());
  program.output_nodes.reserve(output.size());

  for (const Symbol &input : inputs) {
    // Note if old_to_new_index[input.node_index] is UINT32_MAX, it means the
    // input node is not part of the graph leading to the output. It shouldn't
    // matter since we won't be using that input, but we can also choose to
    // throw an error here if we want to enforce that all inputs must be part of
    // the graph. But that isn't optimal because variables or placeholders can
    // be optimized away during compilation, so we allow it for now.

    if (this->nodes[input.node_index].operation != Op::PLACEHOLDER ||
        this->nodes[input.node_index].operation != Op::VARIABLE) {
      throw std::runtime_error("Input symbol must correspond to a PLACEHOLDER "
                               "or VARIABLE node in the graph");
    }

    program.input_nodes.push_back(old_to_new_index[input.node_index]);
  }

  for (const Symbol &out : output) {
    program.output_nodes.push_back(old_to_new_index[out.node_index]);
  }

  // pingpong optimization: we can only optimize after we have the full program
  program = this->optimize(program, optimization_passes, old_to_new_index);

  return program;
}

Program Graph::optimize(Program &program, int optimization_passes,
                        std::vector<uint32_t> &old_to_new_index) {
  // Allocate a secondary buffer for ping-pong optimization. We will alternate
  // between writing to program and alt_program on each pass to avoid
  // unnecessary copying. The final result will be copied back to program at the
  // end if needed. program should be allocated with enough capacity to alllow
  // for growth during optimization, but alt_program starts empty and will be
  // filled in on the first pass.
  Program alt_program;
  alt_program.nodes.reserve(program.nodes.size());
  alt_program.inputs.reserve(program.inputs.size());
  alt_program.values.reserve(program.values.size());

  for (int i = 0; i < optimization_passes; i++) {
    bool changed = false;
    changed |= this->canonicalize(program, alt_program, old_to_new_index);
    changed |= this->addition_folding(alt_program, program, old_to_new_index);
    changed |=
        this->multiplication_folding(program, alt_program, old_to_new_index);
    changed |= this->constant_folding(alt_program, program, old_to_new_index);
    changed |=
        this->algebraic_simplification(program, alt_program, old_to_new_index);
    changed |=
        this->dead_code_elimination(alt_program, program, old_to_new_index);

    if (!changed) {
      break; // stop if no changes were made in this pass
    }
  }

  this->decanonicalize(program, alt_program, old_to_new_index);

  return program;
}

void Graph::canonicalize(const Program &read, Program &write) {
  // We want to turn various operations into a canonical form temporarily for
  // optimization_passes SUB(a, b) -> ADD(a, MULT(-1, b)) NEG(a) -> MULT(-1, b);

  write.nodes.clear();
  write.inputs.clear();
  write.values.clear();

  bool changed = false;

  return changed;
}

void Graph::save_node(const Node &n, const Program &read, Program &write,
                      std::vector<uint32_t> &old_to_new_index) {
  Node new_node{n}; // copy to modify in place

  for (size_t i = 0; i < n.input_count; i++) {
    if (i < 2) {
      new_node.inputs[i] = old_to_new_index[n.inputs[i]];
    } else {
      write.inputs.push_back(read.inputs[n.input_pool_offset + (i - 2)]);
    }
  }

  if (new_node.operation == Op::CONSTANT) {
    write.values.push_back(read.values[n.value_index]);
    new_node.value_index = write.values.size() - 1;
  }

  if (new_node.operation == Op::RESHAPE ||
      new_node.operation == Op::BROADCAST_TO) {
    uint32_t old_shape_offset = new_node.shape.offset;
    new_node.shape.offset = write.shapes.size();

    for (uint32_t i = 0; i < new_node.shape.count; i++) {
      write.shapes.push_back(read.shapes[old_shape_offset + i]);
    }
  }

  write.nodes.push_back(new_node);
  old_to_new_index[&n - &read.nodes[0]] =
      write.nodes.size() - 1; // map old index to new
}

} // namespace autograd
