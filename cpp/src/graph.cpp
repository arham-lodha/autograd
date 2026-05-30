#include "autograd/graph.hpp"
#include "autograd/symbol.hpp"
#include <cstdint>

namespace autograd {

Symbol Graph::variable() { return add_node(Op::VARIABLE, {}); }
Symbol Graph::placeholder() { return add_node(Op::PLACEHOLDER, {}); }
Symbol Graph::constant(const Eigen::MatrixXf &value) {
  uint32_t node_index = nodes.size();
  uint32_t value_index = values.size();

  nodes.push_back(Node{
      .operation = Op::CONSTANT,
      .value_index = value_index,
  });

  values.push_back(value);

  return Symbol{.node_index = node_index, .graph = this};
}

Symbol Graph::constant(const std::vector<float> &value) {
  Eigen::MatrixXf mat =
      Eigen::Map<const Eigen::MatrixXf>(value.data(), value.size(), 1);
  return constant(mat);
}

Symbol Graph::constant(float scalar) {
  Eigen::MatrixXf mat(1, 1);
  mat(0, 0) = scalar;

  return constant(mat);
}

Symbol Graph::add_node(Op op, std::initializer_list<uint32_t> node_inputs,
                       int32_t axis, int32_t axis2, bool keepdims) {
  uint32_t node_index = nodes.size();
  uint32_t input_offset = inputs.size();
  uint16_t input_count = node_inputs.size();
  uint32_t inline_inputs[2] = {0, 0};

  bool requires_grad = op == Op::VARIABLE;

  for (uint32_t i = 0; i < input_count; i++) {
    uint32_t inp_idx = node_inputs.begin()[i];

    if (inp_idx >= nodes.size())
      throw std::runtime_error("Input index out of bounds");

    requires_grad |= nodes[inp_idx].requires_grad;

    if (i < 2)
      inline_inputs[i] = inp_idx;
    else
      this->inputs.push_back(inp_idx);
  }

  nodes.push_back(Node{
      .inputs = {inline_inputs[0], inline_inputs[1]},
      .input_pool_offset = input_offset,
      .axes = {axis, axis2},
      .operation = op,
      .input_count = input_count,
      .requires_grad = requires_grad,
      .keepdims = keepdims,
  });

  return Symbol{.node_index = node_index, .graph = this};
}

void Graph::retain(Symbol sym) {
  if (sym.graph != this) {
    throw std::runtime_error("Symbol does not belong to this graph");
  }

  nodes[sym.node_index].retain = true;
}

} // namespace autograd
