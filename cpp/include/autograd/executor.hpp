#pragma once

#include "autograd/ir.hpp"
#include "autograd/tensor.hpp"
#include <span>
#include <vector>

namespace autograd {

class Executor {
public:
  Executor() = default;

  // Run a compiled Program.  Returns one Tensor per output_node, in order.
  // feed maps input_node indices (from Program::input_nodes) to runtime values.
  std::vector<Tensor> forward(const Program &prog,
                              std::span<const Tensor> feed) const;

private:
  // Dispatch a single node given its already-evaluated inputs.
  Tensor eval_node(const Program &prog, uint32_t node_idx,
                   std::vector<Tensor> &values) const;

  // Non-trivial op helpers.
  static Tensor softmax(const Tensor &x, int32_t axis);
  static Tensor unbroadcast(const Tensor &x, uint32_t target_rows,
                            uint32_t target_cols);
  static Tensor scatter_like(const Tensor &x, uint32_t target_rows,
                             uint32_t target_cols, int32_t index);
};

} // namespace autograd
