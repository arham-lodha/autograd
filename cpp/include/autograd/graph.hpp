#pragma once

#include "autograd/ops.hpp"
#include <Eigen/Dense>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace autograd {

struct Symbol; // defined in symbol.hpp — do NOT include symbol.hpp here
               // (circular)

struct Node {
  Op operation;
  uint32_t input_offset = 0;         // start index into Graph::inputs (CSR)
  uint32_t input_count = 0;          // number of inputs from input_offset
  uint32_t value_index = UINT32_MAX; // CONSTANT nodes only

  bool requires_grad = false;
  bool retain = false;

  // axis doubles as index for GET_ITEM / SCATTER_LIKE; UINT32_MAX = not set
  uint32_t axis = UINT32_MAX;
  uint32_t axis2 = UINT32_MAX; // swap_axis only
  bool keepdims = false;

  // RESHAPE / BROADCAST_TO: CSR slice into Graph::shapes
  uint32_t shape_offset = 0;
  uint32_t shape_count = 0;
};

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
  Symbol constant(const std::vector<float> &value); // 1D vector constant
  Symbol constant(float scalar);
  Symbol placeholder();

  // Appends a node and returns its handle — used internally by Symbol operators
  Symbol add_node(Op op, std::initializer_list<uint32_t> node_inputs,
                  uint32_t axis = UINT32_MAX, uint32_t axis2 = UINT32_MAX,
                  bool keepdims = false);
  void retain(Symbol sym); // mark a node to be retained in the forward
                           // compilation output
};

} // namespace autograd
