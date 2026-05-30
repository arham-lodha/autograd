#pragma once

#include "autograd/ir.hpp"
#include "autograd/ops.hpp"
#include <Eigen/Dense>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace autograd {

struct Symbol; // defined in symbol.hpp — do NOT include symbol.hpp here (circular)

class Graph {
public:
  Graph() = default;
  ~Graph() = default;

  std::vector<Node> nodes;
  std::vector<uint32_t> inputs;        // flat CSR input-index list
  std::vector<Eigen::MatrixXf> values; // constant values
  std::vector<int> shapes;             // flat CSR shape data

  Symbol variable();
  Symbol constant(const Eigen::MatrixXf &value);
  Symbol constant(const std::vector<float> &value);
  Symbol constant(float scalar);
  Symbol placeholder();

  Symbol add_node(Op op, std::initializer_list<uint32_t> node_inputs,
                  int32_t axis = -1, int32_t axis2 = -1, bool keepdims = false);

  void retain(Symbol sym);
};

} // namespace autograd
