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
                       uint32_t axis, uint32_t axis2, bool keepdims) {
  uint32_t node_index = nodes.size();
  uint32_t input_offset = inputs.size();
  uint32_t input_count = node_inputs.size();

  bool requires_grad = op == Op::VARIABLE;

  for (uint32_t input : node_inputs) {
    if (input >= node_index) {
      throw std::runtime_error("Input node index out of bounds");
    }

    if (nodes[input].requires_grad) {
      requires_grad = true;
      break;
    }
  }

  nodes.push_back(Node{
      .operation = op,
      .input_offset = input_offset,
      .input_count = input_count,
      .requires_grad = requires_grad,
      .axis = axis,
      .axis2 = axis2,
      .keepdims = keepdims,
  });

  inputs.insert(inputs.end(), node_inputs);

  return Symbol{.node_index = node_index, .graph = this};
}

void Graph::retain(Symbol sym) {
  if (sym.graph != this) {
    throw std::runtime_error("Symbol does not belong to this graph");
  }

  nodes[sym.node_index].retain = true;
}

} // namespace autograd
