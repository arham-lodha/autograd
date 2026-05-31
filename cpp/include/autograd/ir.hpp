#pragma once
#include "autograd/ops.hpp"
#include "autograd/tensor.hpp"
#include <cstdint>
#include <vector>

namespace autograd {

struct Axes {
  int32_t axis, axis2;
};
struct Shape {
  uint32_t offset;
  uint16_t count;
};

struct Node {
  uint32_t inputs[2] = {0, 0};
  uint32_t input_pool_offset = 0; // start index into Graph::input_pool (CSR)
                                  // where n-ary inputs are listed

  union {
    uint32_t value_index; // for CONSTANT nodes: index into Graph::values

    // for OP::SUM, MEAN, SWAP_AXIS, SQUEEZE
    Axes axes;

    // OP::RESHAPE, OP::BROADCAST_TO: slice into Graph::shapes for the new shape
    Shape shape;
  };

  Op operation;
  uint16_t input_count = 0; // for n-ary ops; ignored for fixed-arity ops

  uint8_t requires_grad : 1 = 0;
  uint8_t retain : 1 =
      0; // whether to retain this node in forward compilation output
  uint8_t keepdims : 1 = 0; // whether to keep reduced dims in REDUCE ops
  uint8_t _padding : 5 = 0; // padding

  uint8_t _reserved[7] = {
      0}; // reserved for future use; ensures sizeof(Node) is a multiple of 16
};

struct Program {
  std::vector<Node> nodes;

  std::vector<uint32_t> inputs; // flat CSR input-index list
  std::vector<uint32_t> shapes; // flat CSR shape data
  std::vector<Tensor> values;   // constant values

  std::vector<uint32_t>
      input_nodes; // indices of input nodes (for multi-input graphs)
  std::vector<uint32_t>
      output_nodes; // indices of output nodes (for multi-output graphs)
};

} // namespace autograd
