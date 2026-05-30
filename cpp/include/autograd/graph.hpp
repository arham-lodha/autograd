#pragma once

#include "autograd/ops.hpp"
#include <Eigen/Dense>
#include <cstdint>
#include <initializer_list>
#include <unordered_map>
#include <vector>

namespace autograd {

struct Symbol; // defined in symbol.hpp — do NOT include symbol.hpp here
               // (circular)

enum : uint32_t {
  K_ONE = 0,
  K_ZERO = 1,
  K_NEG_ONE = 2,
  K_HALF = 3,
  K_COUNT = 4
};

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

  uint8_t _reserved[3] = {
      0, 0,
      0}; // reserved for future use; ensures sizeof(Node) is a multiple of 16
};

struct Program {
  std::vector<Node> nodes;

  std::vector<uint32_t> inputs;        // flat CSR input-index list
  std::vector<uint32_t> shapes;        // flat CSR shape data
  std::vector<Eigen::MatrixXf> values; // constant values

  std::vector<uint32_t>
      input_nodes; // indices of input nodes (for multi-input graphs)
  std::vector<uint32_t>
      output_nodes; // indices of output nodes (for multi-output graphs)
};

class Graph {
public:
  Graph() = default;
  ~Graph() = default;

  std::vector<Node> nodes;
  std::vector<uint32_t> inputs;        // flat CSR input-index list
  std::vector<Eigen::MatrixXf> values; // constant values
  std::vector<int> shapes;             // flat CSR shape data
  std::unordered_map<float, uint32_t>
      scalar_constants; // map from scalar value to constant node

  Symbol variable();
  Symbol constant(const Eigen::MatrixXf &value);
  Symbol constant(const std::vector<float> &value); // 1D vector constant
  Symbol constant(float scalar);
  Symbol placeholder();

  // Appends a node and returns its handle — used internally by Symbol operators
  Symbol add_node(Op op, std::initializer_list<uint32_t> node_inputs,
                  int32_t axis = -1, int32_t axis2 = -1, bool keepdims = false);
  void retain(Symbol sym); // mark a node to be retained in the forward
                           // compilation output

  Program compile(const std::vector<Symbol> &inputs,
                  const std::vector<Symbol> &outputs,
                  int optimization_passes = 10);

private:
  std::vector<uint32_t> topological_sort(const std::vector<Symbol> &outputs);
  Program optimize(Program &prog, int optimization_passes,
                   std::vector<uint32_t> &old_to_new_index);
  bool canonicalize(const Program &read, Program &write,
                    std::vector<uint32_t> &old_to_new_index,
                    std::array<uint32_t, K_COUNT> &slots);
  bool addition_multiplication_folding(const Program &read, Program &write,
                                       std::vector<uint32_t> &old_to_new_index);
  bool constant_folding(const Program &read, Program &write,
                        std::vector<uint32_t> &old_to_new_index,
                        std::vector<Eigen::MatrixXf> &arena);
  bool algebraic_simplification(const Program &read, Program &write,
                                std::vector<uint32_t> &old_to_new_index,
                                std::vector<Eigen::MatrixXf> &arena);
  bool decanonicalize(const Program &read, Program &write,
                      std::vector<uint32_t> &old_to_new_index,
                      std::vector<Eigen::MatrixXf> &arena);
  bool dead_code_elimination(const Program &read, Program &write,
                             std::vector<uint32_t> &old_to_new_index,
                             std::vector<Eigen::MatrixXf> &arena);
  void save_node(const Node &n, const Program &read, Program &write,
                 std::vector<uint32_t> &old_to_new_index, uint32_t old_idx);
  void move_inputs_outputs(const Program &read, Program &write,
                           std::vector<uint32_t> &old_to_new_index);

  void clean_write_and_remap(const Program &read, Program &write,
                             std::vector<uint32_t> &old_to_new_index);
};

} // namespace autograd
