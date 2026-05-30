#include "autograd/graph.hpp"
#include "autograd/ops.hpp"
#include "autograd/symbol.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
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

static Eigen::MatrixXf scalar_mat(float v) {
  Eigen::MatrixXf m(1, 1);
  m(0, 0) = v;
  return m;
}

static uint32_t well_known(uint32_t slot, std::array<uint32_t, K_COUNT> &slots,
                           Program &write) {
  if (slots[slot] != UINT32_MAX)
    return slots[slot];

  Node c;
  c.operation = Op::CONSTANT;
  c.value_index = slot;
  write.nodes.push_back(c);
  slots[slot] = write.nodes.size() - 1;
  return write.nodes.size() - 1;
}

Program Graph::optimize(Program &program, int optimization_passes,
                        std::vector<uint32_t> &old_to_new_index) {
  // One append only value arena, shared by both buffers and every pass
  // Nodes reference it via value_index; matrixes are appended, never copied
  std::vector<Eigen::MatrixXf> arena;
  arena.reserve(K_COUNT + program.values.size());
  arena.push_back(scalar_mat(1.0f));
  arena.push_back(scalar_mat(0.0f));
  arena.push_back(scalar_mat(-1.0f));
  arena.push_back(scalar_mat(0.5f));
  for (Eigen::MatrixXf &m : program.values)
    arena.push_back(std::move(m));

  for (Node &n : program.nodes) {
    if (n.operation == Op::CONSTANT)
      n.value_index += K_COUNT;
  }

  // Allocate a secondary buffer for ping-pong optimization. We will alternate
  // between writing to program and alt_program on each pass to avoid
  // unnecessary copying. The final result will be copied back to program at the
  // end if needed. program should be allocated with enough capacity to alllow
  // for growth during optimization, but alt_program starts empty and will be
  // filled in on the first pass.
  Program a = std::move(program), b;
  a.values.clear();
  b.nodes.reserve(a.nodes.size());
  b.inputs.reserve(a.inputs.size());
  b.shapes.reserve(a.shapes.size());
  b.output_nodes.reserve(a.output_nodes.size());
  b.input_nodes.reserve(a.input_nodes.size());

  Program *read = &a, *write = &b;
  auto flip = [&] { std::swap(read, write); };

  std::array<uint32_t, K_COUNT> slots = {UINT32_MAX, UINT32_MAX, UINT32_MAX,
                                         UINT32_MAX};

  for (int i = 0; i < optimization_passes; i++) {
    bool changed = false;
    changed |= this->canonicalize(*read, *write, old_to_new_index, slots);
    flip();
    changed |=
        this->addition_multiplication_folding(*read, *write, old_to_new_index);
    flip();
    changed |= this->constant_folding(*read, *write, old_to_new_index, arena);
    flip();
    changed |=
        this->algebraic_simplification(*read, *write, old_to_new_index, arena);
    flip();
    changed |=
        this->dead_code_elimination(*read, *write, old_to_new_index, arena);
    flip();
    if (!changed) {
      break; // stop if no changes were made in this pass
    }
  }

  this->decanonicalize(*read, *write, old_to_new_index, arena);

  // clean up arena and attach to write

  return *write;
}

void Graph::clean_write_and_remap(const Program &read, Program &write,
                                  std::vector<uint32_t> &old_to_new_index) {
  write.nodes.clear();
  write.inputs.clear();
  write.shapes.clear();
  write.input_nodes.clear();
  write.output_nodes.clear();

  old_to_new_index.assign(read.nodes.size(), UINT32_MAX);
}

bool Graph::canonicalize(const Program &read, Program &write,
                         std::vector<uint32_t> &old_to_new_index,
                         std::array<uint32_t, K_COUNT> &slots) {

  // We want to turn various operations into a canonical form temporarily for
  // optimization_passes SUB(a, b) -> ADD(a, MULT(-1, b)) NEG(a) -> MULT(-1, b);

  clean_write_and_remap(read, write, old_to_new_index);
  bool changed = false;

  for (uint32_t old_idx = 0; old_idx < read.nodes.size(); old_idx++) {
    const Node &node = read.nodes[old_idx];
    switch (node.operation) {
    case Op::NEG: {
      uint32_t param = old_to_new_index[node.inputs[0]];
      uint32_t neg_one = well_known(K_NEG_ONE, slots, write);
      Node multiply;
      multiply.operation = Op::MULTIPLY;
      multiply.input_count = 2;
      multiply.inputs[0] = param;
      multiply.inputs[1] = neg_one;
      multiply.requires_grad = node.requires_grad;
      multiply.retain = node.retain;
      write.nodes.push_back(multiply);
      old_to_new_index[old_idx] = static_cast<uint32_t>(write.nodes.size() - 1);
      changed = true;
      break;
    }
    case Op::SUB: {
      uint32_t a = old_to_new_index[node.inputs[0]];
      uint32_t b = old_to_new_index[node.inputs[1]];
      uint32_t neg_one = well_known(K_NEG_ONE, slots, write);
      Node multiply;
      multiply.operation = Op::MULTIPLY;
      multiply.input_count = 2;
      multiply.inputs[0] = b;
      multiply.inputs[1] = neg_one;
      multiply.requires_grad = read.nodes[node.inputs[1]].requires_grad;
      write.nodes.push_back(multiply);
      uint32_t multiply_idx = static_cast<uint32_t>(write.nodes.size() - 1);

      Node add;
      add.operation = Op::ADD;
      add.input_count = 2;
      add.inputs[0] = a;
      add.inputs[1] = multiply_idx;
      add.requires_grad = node.requires_grad;
      add.retain = node.retain;
      write.nodes.push_back(add);
      old_to_new_index[old_idx] = static_cast<uint32_t>(write.nodes.size() - 1);
      changed = true;

      break;
    }
    case Op::SQRT: {
      uint32_t param = old_to_new_index[node.inputs[0]];
      uint32_t half = well_known(K_HALF, slots, write);
      Node power;
      power.operation = Op::POWER;
      power.inputs[0] = param;
      power.inputs[1] = half;
      power.requires_grad = node.requires_grad;
      power.retain = node.retain;
      write.nodes.push_back(power);
      old_to_new_index[old_idx] = static_cast<uint32_t>(write.nodes.size() - 1);
      changed = true;
      break;
    }
    default: {
      save_node(node, read, write, old_to_new_index, old_idx);
      break;
    }
    }
  }

  this->move_inputs_outputs(read, write, old_to_new_index);

  return changed;
}

bool Graph::addition_multiplication_folding(
    const Program &read, Program &write,
    std::vector<uint32_t> &old_to_new_index) {

  // This function folds both addition and multiplication. It performs a dfs
  // search though the children of a addition/multiplication node
  this->clean_write_and_remap(read, write, old_to_new_index);

  bool changed = false;

  for (uint32_t old_idx = 0; old_idx < read.nodes.size(); old_idx++) {

    const Node &node = read.nodes[old_idx];

    if (node.operation == Op::ADD || node.operation == Op::MULTIPLY) {
      Op target = node.operation;
      std::vector<uint32_t> children;
      std::stack<uint32_t> stack;

      for (uint32_t i = 0; i < node.input_count; i++) {
        stack.push(i < 2 ? old_to_new_index[node.inputs[i]]
                         : old_to_new_index[read.inputs[node.input_pool_offset +
                                                        (i - 2)]]);
      }

      while (!stack.empty()) {
        uint32_t node_index = stack.top();
        stack.pop();
        if (write.nodes[node_index].operation != target ||
            write.nodes[node_index].retain) {
          children.push_back(node_index);
        } else {
          for (uint32_t i = 0; i < write.nodes[node_index].input_count; i++) {
            stack.push(
                i < 2 ? write.nodes[node_index].inputs[i]
                      : write.inputs[write.nodes[node_index].input_pool_offset +
                                     (i - 2)]);
          }
        }
      }

      if (children.size() != node.input_count) {
        Node new_add;
        new_add.operation = target;
        new_add.input_count = children.size();
        new_add.input_pool_offset = write.inputs.size();
        new_add.requires_grad = node.requires_grad;
        new_add.retain = node.retain;

        for (uint32_t i = 0; i < children.size(); i++) {
          if (i < 2) {
            new_add.inputs[i] = children[i];
          } else {
            write.inputs.push_back(children[i]);
          }
        }

        // technically folding the addition shouldn't change the number of total
        // nodes. it just introduces unreachable nodes
        old_to_new_index[old_idx] = write.nodes.size();
        write.nodes.push_back(new_add);
        changed = true;
      } else
        this->save_node(node, read, write, old_to_new_index, old_idx);
    } else
      this->save_node(node, read, write, old_to_new_index, old_idx);
  }

  this->move_inputs_outputs(read, write, old_to_new_index);
  return changed;
}

bool Graph::constant_folding(const Program &read, Program &write,
                             std::vector<uint32_t> &old_to_new_index,
                             std::vector<Eigen::MatrixXf> &arena) {
  this->clean_write_and_remap(read, write, old_to_new_index);
  bool changed = false;

  this->move_inputs_outputs(read, write, old_to_new_index);
  return changed;
}

void Graph::save_node(const Node &n, const Program &read, Program &write,
                      std::vector<uint32_t> &old_to_new_index,
                      uint32_t old_idx) {
  Node new_node{n}; // copy to modify in place

  new_node.input_pool_offset = write.inputs.size();

  for (size_t i = 0; i < n.input_count; i++) {
    if (i < 2) {
      new_node.inputs[i] = old_to_new_index[n.inputs[i]];
    } else {
      write.inputs.push_back(
          old_to_new_index[read.inputs[n.input_pool_offset + (i - 2)]]);
    }
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
  old_to_new_index[old_idx] = write.nodes.size() - 1; // map old index to new
}

void Graph::move_inputs_outputs(const Program &read, Program &write,
                                std::vector<uint32_t> &old_to_new_index) {
  for (size_t i = 0; i < read.input_nodes.size(); i++) {
    write.input_nodes.push_back(read.input_nodes[i] != UINT32_MAX
                                    ? old_to_new_index[read.input_nodes[i]]
                                    : UINT32_MAX);
  }

  for (size_t i = 0; i < read.output_nodes.size(); i++) {
    write.output_nodes.push_back(read.output_nodes[i] != UINT32_MAX
                                     ? old_to_new_index[read.output_nodes[i]]
                                     : UINT32_MAX);
  }
}

} // namespace autograd
