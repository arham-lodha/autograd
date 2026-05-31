#include "autograd/executor.hpp"
#include "autograd/ir.hpp"
#include "autograd/ops.hpp"
#include "autograd/tensor.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace autograd {

// ── broadcast helpers
// ─────────────────────────────────────────────────────────

// Merge an accumulated shape (a_ndim, a_shape[]) with tensor b into out.
// NumPy rules: align from the right, each dim must be equal or one of them 1.
static void merge_broadcast_shape(uint8_t a_ndim, const uint32_t *a_shape,
                                  const Tensor &b, uint8_t &out_ndim,
                                  uint32_t *out_shape) {
  uint8_t b_ndim = b.ndim();
  out_ndim = std::max(a_ndim, b_ndim);
  for (uint8_t i = 0; i < out_ndim; i++) {
    int ai = static_cast<int>(i) - (out_ndim - a_ndim);
    int bi = static_cast<int>(i) - (out_ndim - b_ndim);
    uint32_t da = ai >= 0 ? a_shape[ai] : 1;
    uint32_t db = bi >= 0 ? b.shape(bi) : 1;
    assert(da == db || da == 1 || db == 1);
    out_shape[i] = std::max(da, db);
  }
}

// Returns x broadcast to (out_ndim, out_shape[]).
// Fast-path: if x already matches, returns a copy without touching its data.
static Tensor broadcast_to_shape(const Tensor &x, uint8_t out_ndim,
                                 const uint32_t *out_shape) {
  // Fast path: same shape already.
  if (x.ndim() == out_ndim) {
    bool match = true;
    for (uint8_t i = 0; i < out_ndim; i++)
      if (x.shape(i) != out_shape[i]) {
        match = false;
        break;
      }
    if (match)
      return x;
  }

  Tensor result(std::span<const uint32_t>(out_shape, out_ndim));

  // x's effective shape padded with 1s on the left to match out_ndim.
  uint32_t src[Tensor::kMaxDims] = {};
  uint8_t pad = out_ndim - x.ndim();
  for (uint8_t i = 0; i < out_ndim; i++)
    src[i] = i < pad ? 1 : x.shape(i - pad);

  // Batch dims = all dims except the last two.
  uint32_t out_b0 = out_ndim > 2 ? out_shape[0] : 1;
  uint32_t out_b1 = out_ndim > 3 ? out_shape[1] : 1;
  uint32_t src_b0 = out_ndim > 2 ? src[0] : 1;
  uint32_t src_b1 = out_ndim > 3 ? src[1] : 1;
  uint32_t src_rows = out_ndim >= 2 ? src[out_ndim - 2] : 1;
  uint32_t src_cols = out_ndim >= 1 ? src[out_ndim - 1] : 1;
  uint32_t out_rows = out_ndim >= 2 ? out_shape[out_ndim - 2] : 1;
  uint32_t out_cols = out_ndim >= 1 ? out_shape[out_ndim - 1] : 1;

  for (uint32_t b0 = 0; b0 < out_b0; b0++) {
    for (uint32_t b1 = 0; b1 < out_b1; b1++) {
      uint32_t out_batch = b0 * out_b1 + b1;
      uint32_t src_batch = (b0 % src_b0) * src_b1 + (b1 % src_b1);

      auto dst = result.matrix_slice(out_batch);
      auto src_m = x.matrix_slice(src_batch);

      if (src_rows == out_rows && src_cols == out_cols)
        dst = src_m;
      else if (src_rows == 1 && src_cols == 1)
        dst.fill(src_m(0, 0));
      else if (src_rows == 1)
        dst = src_m.row(0).replicate(out_rows, 1);
      else // src_cols == 1
        dst = src_m.col(0).replicate(1, out_cols);
    }
  }

  return result;
}

// Macro-like helper used in binary ops: compute output shape, broadcast both.
#define BROADCAST_BINARY(a, b, out_ndim, out_shape, ba, bb)                    \
  uint8_t out_ndim = (a).ndim();                                               \
  uint32_t out_shape[Tensor::kMaxDims] = {};                                   \
  for (uint8_t _i = 0; _i < out_ndim; _i++)                                    \
    out_shape[_i] = (a).shape(_i);                                             \
  merge_broadcast_shape(out_ndim, out_shape, (b), out_ndim, out_shape);        \
  Tensor ba = broadcast_to_shape((a), out_ndim, out_shape);                    \
  Tensor bb = broadcast_to_shape((b), out_ndim, out_shape);

// ── forward
// ───────────────────────────────────────────────────────────────────

std::vector<Tensor> Executor::forward(const Program &prog,
                                      std::span<const Tensor> feed) {
  if (prog.input_nodes.size() != feed.size())
    throw std::runtime_error("Executor::forward: feed size does not match "
                             "number of program input nodes");

  if (values.size() < prog.nodes.size()) {
    values.resize(prog.nodes.size());
  }

  for (size_t i = 0; i < feed.size(); i++)
    values[prog.input_nodes[i]] = feed[i];

  for (uint32_t n = 0; n < prog.nodes.size(); n++) {
    const Op op = prog.nodes[n].operation;
    if (op == Op::VARIABLE || op == Op::PLACEHOLDER)
      continue;
    if (op == Op::CONSTANT) {
      values[n] = prog.values[prog.nodes[n].value_index];
      continue;
    }
    values[n] = eval_node(prog, n, values);
  }

  std::vector<Tensor> output;
  output.reserve(prog.output_nodes.size());
  for (uint32_t idx : prog.output_nodes)
    output.push_back(values[idx]);
  return output;
}

// ── eval_node
// ─────────────────────────────────────────────────────────────────

Tensor Executor::eval_node(const Program &prog, uint32_t node_idx,
                           std::vector<Tensor> &values) const {
  const Node &node = prog.nodes[node_idx];

  auto inp = [&](uint32_t i) -> const Tensor & {
    uint32_t idx =
        i < 2 ? node.inputs[i] : prog.inputs[node.input_pool_offset + (i - 2)];
    return values[idx];
  };

  switch (node.operation) {

  // ── n-ary elementwise (with autobroadcast) ─────────────────────────────────
  case Op::ADD: {
    // Fold broadcast shape over all inputs.
    uint8_t out_ndim = inp(0).ndim();
    uint32_t out_shape[Tensor::kMaxDims] = {};
    for (uint8_t i = 0; i < out_ndim; i++)
      out_shape[i] = inp(0).shape(i);
    for (uint32_t i = 1; i < node.input_count; i++)
      merge_broadcast_shape(out_ndim, out_shape, inp(i), out_ndim, out_shape);

    Tensor result(std::span<const uint32_t>(out_shape, out_ndim)); // zero-init
    uint32_t batch = result.batch_size();
    for (uint32_t i = 0; i < node.input_count; i++) {
      Tensor t = broadcast_to_shape(inp(i), out_ndim, out_shape);
      for (uint32_t b = 0; b < batch; b++)
        result.matrix_slice(b) += t.matrix_slice(b);
    }
    return result;
  }
  case Op::MULTIPLY: {
    uint8_t out_ndim = inp(0).ndim();
    uint32_t out_shape[Tensor::kMaxDims] = {};
    for (uint8_t i = 0; i < out_ndim; i++)
      out_shape[i] = inp(0).shape(i);
    for (uint32_t i = 1; i < node.input_count; i++)
      merge_broadcast_shape(out_ndim, out_shape, inp(i), out_ndim, out_shape);

    // Start from ones, multiply each broadcast input in.
    Tensor result = broadcast_to_shape(inp(0), out_ndim, out_shape);
    uint32_t batch = result.batch_size();
    for (uint32_t i = 1; i < node.input_count; i++) {
      Tensor t = broadcast_to_shape(inp(i), out_ndim, out_shape);
      for (uint32_t b = 0; b < batch; b++)
        result.matrix_slice(b).array() *= t.matrix_slice(b).array();
    }
    return result;
  }

  // ── binary (with autobroadcast) ────────────────────────────────────────────
  case Op::POWER: {
    BROADCAST_BINARY(inp(0), inp(1), out_ndim, out_shape, a, b)
    Tensor result(std::span<const uint32_t>(out_shape, out_ndim));
    for (uint32_t bi = 0; bi < result.batch_size(); bi++)
      result.matrix_slice(bi).array() =
          a.matrix_slice(bi).array().pow(b.matrix_slice(bi).array());
    return result;
  }
  case Op::SUB: {
    BROADCAST_BINARY(inp(0), inp(1), out_ndim, out_shape, a, b)
    Tensor result(std::span<const uint32_t>(out_shape, out_ndim));
    for (uint32_t bi = 0; bi < result.batch_size(); bi++)
      result.matrix_slice(bi) = a.matrix_slice(bi) - b.matrix_slice(bi);
    return result;
  }
  case Op::GREATER_THAN: {
    BROADCAST_BINARY(inp(0), inp(1), out_ndim, out_shape, a, b)
    Tensor result(std::span<const uint32_t>(out_shape, out_ndim));
    for (uint32_t bi = 0; bi < result.batch_size(); bi++)
      result.matrix_slice(bi).array() =
          (a.matrix_slice(bi).array() > b.matrix_slice(bi).array())
              .cast<float>();
    return result;
  }
  case Op::LESS_THAN: {
    BROADCAST_BINARY(inp(0), inp(1), out_ndim, out_shape, a, b)
    Tensor result(std::span<const uint32_t>(out_shape, out_ndim));
    for (uint32_t bi = 0; bi < result.batch_size(); bi++)
      result.matrix_slice(bi).array() =
          (a.matrix_slice(bi).array() < b.matrix_slice(bi).array())
              .cast<float>();
    return result;
  }
  case Op::GREATER_THAN_OR_EQUAL: {
    BROADCAST_BINARY(inp(0), inp(1), out_ndim, out_shape, a, b)
    Tensor result(std::span<const uint32_t>(out_shape, out_ndim));
    for (uint32_t bi = 0; bi < result.batch_size(); bi++)
      result.matrix_slice(bi).array() =
          (a.matrix_slice(bi).array() >= b.matrix_slice(bi).array())
              .cast<float>();
    return result;
  }
  case Op::LESS_THAN_OR_EQUAL: {
    BROADCAST_BINARY(inp(0), inp(1), out_ndim, out_shape, a, b)
    Tensor result(std::span<const uint32_t>(out_shape, out_ndim));
    for (uint32_t bi = 0; bi < result.batch_size(); bi++)
      result.matrix_slice(bi).array() =
          (a.matrix_slice(bi).array() <= b.matrix_slice(bi).array())
              .cast<float>();
    return result;
  }
  case Op::EQUALS_TO: {
    BROADCAST_BINARY(inp(0), inp(1), out_ndim, out_shape, a, b)
    Tensor result(std::span<const uint32_t>(out_shape, out_ndim));
    for (uint32_t bi = 0; bi < result.batch_size(); bi++)
      result.matrix_slice(bi).array() =
          (a.matrix_slice(bi).array() == b.matrix_slice(bi).array())
              .cast<float>();
    return result;
  }

  // ── matmul (batched, no element-wise broadcast) ────────────────────────────
  case Op::MATMUL: {
    const Tensor &a = inp(0), &b = inp(1);
    uint32_t out_shape[Tensor::kMaxDims] = {};
    uint8_t out_ndim = a.ndim();
    for (uint8_t i = 0; i + 2 < a.ndim(); i++)
      out_shape[i] = a.shape(i);
    out_shape[out_ndim - 2] = a.rows();
    out_shape[out_ndim - 1] = b.cols();
    Tensor result(std::span<const uint32_t>(out_shape, out_ndim));
    for (uint32_t bi = 0; bi < a.batch_size(); bi++)
      result.matrix_slice(bi) = a.matrix_slice(bi) * b.matrix_slice(bi);
    return result;
  }
  case Op::RMATMUL: {
    const Tensor &a = inp(0), &b = inp(1);
    uint32_t out_shape[Tensor::kMaxDims] = {};
    uint8_t out_ndim = b.ndim();
    for (uint8_t i = 0; i + 2 < b.ndim(); i++)
      out_shape[i] = b.shape(i);
    out_shape[out_ndim - 2] = b.rows();
    out_shape[out_ndim - 1] = a.cols();
    Tensor result(std::span<const uint32_t>(out_shape, out_ndim));
    for (uint32_t bi = 0; bi < b.batch_size(); bi++)
      result.matrix_slice(bi) = b.matrix_slice(bi) * a.matrix_slice(bi);
    return result;
  }

  // ── explicit broadcast / unbroadcast ───────────────────────────────────────
  case Op::BROADCAST_TO: {
    const Shape &sh = node.shape;
    uint32_t target[Tensor::kMaxDims] = {};
    for (uint16_t i = 0; i < sh.count; i++)
      target[i] = prog.shapes[sh.offset + i];
    return broadcast_to_shape(inp(0), static_cast<uint8_t>(sh.count), target);
  }
  case Op::BROADCAST_TO_MATCH: {
    const Tensor &other = inp(1);
    uint32_t target[Tensor::kMaxDims] = {};
    for (uint8_t i = 0; i < other.ndim(); i++)
      target[i] = other.shape(i);
    return broadcast_to_shape(inp(0), other.ndim(), target);
  }
  case Op::UNBROADCAST: {
    return unbroadcast(inp(0), inp(1).rows(), inp(1).cols());
  }

  // ── other binary ───────────────────────────────────────────────────────────
  case Op::RESHAPE_LIKE: {
    const Tensor &other = inp(1);
    uint32_t target[Tensor::kMaxDims] = {};
    for (uint8_t i = 0; i < other.ndim(); i++)
      target[i] = other.shape(i);
    return Tensor(std::span<const uint32_t>(target, other.ndim()),
                  inp(0).data());
  }
  case Op::GET_ITEM: {
    const Tensor &x = inp(0);
    int32_t index = node.axes.axis;
    Tensor result(1, x.cols());
    result.map().row(0) = x.map().row(index);
    return result;
  }
  case Op::SCATTER_LIKE: {
    return scatter_like(inp(0), inp(1).rows(), inp(1).cols(), node.axes.axis);
  }

  // ── unary ──────────────────────────────────────────────────────────────────
  case Op::NEG: {
    uint32_t sh[Tensor::kMaxDims] = {};
    for (uint8_t i = 0; i < inp(0).ndim(); i++)
      sh[i] = inp(0).shape(i);
    Tensor result(std::span<const uint32_t>(sh, inp(0).ndim()));
    for (uint32_t b = 0; b < inp(0).batch_size(); b++)
      result.matrix_slice(b) = -inp(0).matrix_slice(b);
    return result;
  }

// Unary ops that operate batch-slice-wise via Eigen array.
#define UNARY_ARRAY(op_expr)                                                   \
  {                                                                            \
    uint32_t sh[Tensor::kMaxDims] = {};                                        \
    for (uint8_t i = 0; i < inp(0).ndim(); i++)                                \
      sh[i] = inp(0).shape(i);                                                 \
    Tensor result(std::span<const uint32_t>(sh, inp(0).ndim()));               \
    for (uint32_t b = 0; b < inp(0).batch_size(); b++)                         \
      result.matrix_slice(b).array() = (op_expr);                              \
    return result;                                                             \
  }

  case Op::EXP:
    UNARY_ARRAY(inp(0).matrix_slice(b).array().exp())
  case Op::LOG:
    UNARY_ARRAY(inp(0).matrix_slice(b).array().log())
  case Op::SQRT:
    UNARY_ARRAY(inp(0).matrix_slice(b).array().sqrt())
  case Op::ABS:
    UNARY_ARRAY(inp(0).matrix_slice(b).array().abs())
  case Op::SIGN:
    UNARY_ARRAY(inp(0).matrix_slice(b).array().sign())
  case Op::RELU:
    UNARY_ARRAY(inp(0).matrix_slice(b).array().max(0.0f))

#undef UNARY_ARRAY

  case Op::TRANSPOSE: {
    uint32_t sh[Tensor::kMaxDims] = {};
    for (uint8_t i = 0; i < inp(0).ndim(); i++)
      sh[i] = inp(0).shape(i);
    std::swap(sh[inp(0).ndim() - 2], sh[inp(0).ndim() - 1]);
    Tensor result(std::span<const uint32_t>(sh, inp(0).ndim()));
    for (uint32_t b = 0; b < inp(0).batch_size(); b++)
      result.matrix_slice(b) = inp(0).matrix_slice(b).transpose();
    return result;
  }
  case Op::IDENTITY: {
    return inp(0);
  }
  case Op::SOFTMAX: {
    return softmax(inp(0), node.axes.axis);
  }

  // ── reductions ─────────────────────────────────────────────────────────────
  case Op::SUM: {
    const Tensor &x = inp(0);
    int32_t axis = node.axes.axis;
    bool kd = node.keepdims;
    if (axis < 0) {
      Tensor result(1, 1);
      result(0, 0) = x.map().sum();
      return result;
    } else if (axis == 0) {
      Tensor result(kd ? 1 : 1, x.cols());
      result.map() = x.map().colwise().sum();
      return result;
    } else {
      Tensor result(x.rows(), kd ? 1 : 1);
      result.map() = x.map().rowwise().sum();
      return result;
    }
  }
  case Op::MEAN: {
    const Tensor &x = inp(0);
    int32_t axis = node.axes.axis;
    bool kd = node.keepdims;
    if (axis < 0) {
      Tensor result(1, 1);
      result(0, 0) = x.map().mean();
      return result;
    } else if (axis == 0) {
      Tensor result(kd ? 1 : 1, x.cols());
      result.map() = x.map().colwise().mean();
      return result;
    } else {
      Tensor result(x.rows(), kd ? 1 : 1);
      result.map() = x.map().rowwise().mean();
      return result;
    }
  }
  case Op::VARIANCE: {
    const Tensor &x = inp(0);
    int32_t axis = node.axes.axis;
    if (axis < 0) {
      float mean = x.map().mean();
      Tensor result(1, 1);
      result(0, 0) = (x.map().array() - mean).square().mean();
      return result;
    } else if (axis == 0) {
      Eigen::RowVectorXf mean = x.map().colwise().mean();
      Tensor result(1, x.cols());
      result.map() =
          (x.map().rowwise() - mean).array().square().colwise().mean();
      return result;
    } else {
      Eigen::VectorXf mean = x.map().rowwise().mean();
      Tensor result(x.rows(), 1);
      result.map() =
          (x.map().colwise() - mean).array().square().rowwise().mean();
      return result;
    }
  }
  case Op::SIZE: {
    const Tensor &x = inp(0);
    int32_t axis = node.axes.axis;
    Tensor result(1, 1);
    if (axis < 0)
      result(0, 0) = static_cast<float>(x.size());
    else if (axis == 0)
      result(0, 0) = static_cast<float>(x.rows());
    else
      result(0, 0) = static_cast<float>(x.cols());
    return result;
  }

  // ── shape ──────────────────────────────────────────────────────────────────
  case Op::RESHAPE: {
    const Shape &sh = node.shape;
    uint32_t target[Tensor::kMaxDims] = {};
    for (uint16_t i = 0; i < sh.count; i++)
      target[i] = prog.shapes[sh.offset + i];
    return Tensor(std::span<const uint32_t>(target, sh.count), inp(0).data());
  }
  case Op::EXPAND_DIMS: {
    const Tensor &x = inp(0);
    int32_t axis = node.axes.axis;
    assert(x.ndim() + 1 <= Tensor::kMaxDims);
    uint32_t new_shape[Tensor::kMaxDims] = {};
    uint8_t j = 0;
    for (uint8_t i = 0; i <= x.ndim(); i++)
      new_shape[i] = (i == static_cast<uint8_t>(axis)) ? 1 : x.shape(j++);
    return Tensor(std::span<const uint32_t>(new_shape, x.ndim() + 1), x.data());
  }
  case Op::SQUEEZE: {
    const Tensor &x = inp(0);
    int32_t axis = node.axes.axis;
    uint32_t new_shape[Tensor::kMaxDims] = {};
    uint8_t new_ndim = 0;
    for (uint8_t i = 0; i < x.ndim(); i++) {
      bool drop =
          (axis < 0) ? (x.shape(i) == 1) : (i == static_cast<uint8_t>(axis));
      if (!drop)
        new_shape[new_ndim++] = x.shape(i);
    }
    if (new_ndim == 0) {
      new_shape[0] = 1;
      new_ndim = 1;
    }
    return Tensor(std::span<const uint32_t>(new_shape, new_ndim), x.data());
  }
  case Op::SWAP_AXIS: {
    const Tensor &x = inp(0);
    int32_t a1 = node.axes.axis, a2 = node.axes.axis2;
    uint32_t new_shape[Tensor::kMaxDims] = {};
    for (uint8_t i = 0; i < x.ndim(); i++)
      new_shape[i] = x.shape(i);
    std::swap(new_shape[a1], new_shape[a2]);
    if ((a1 == x.ndim() - 2 && a2 == x.ndim() - 1) ||
        (a2 == x.ndim() - 2 && a1 == x.ndim() - 1)) {
      Tensor result(std::span<const uint32_t>(new_shape, x.ndim()));
      for (uint32_t b = 0; b < x.batch_size(); b++)
        result.matrix_slice(b) = x.matrix_slice(b).transpose();
      return result;
    }
    throw std::runtime_error(
        "SWAP_AXIS: non-matrix axis swap not yet supported");
  }

  // ── stacking ───────────────────────────────────────────────────────────────
  case Op::VECTOR: {
    uint32_t rows = inp(0).rows();
    Tensor result(rows, node.input_count);
    for (uint32_t i = 0; i < node.input_count; i++)
      result.map().col(i) = inp(i).map().col(0);
    return result;
  }

  default:
    throw std::runtime_error("Executor::eval_node: unhandled Op");
  }
}

#undef BROADCAST_BINARY

// ── non-trivial op helpers
// ────────────────────────────────────────────────────

Tensor Executor::softmax(const Tensor &x, int32_t axis) {
  Tensor result(x.rows(), x.cols());
  if (axis == 0) {
    Eigen::RowVectorXf col_max = x.map().colwise().maxCoeff();
    Eigen::MatrixXf shifted = x.map().rowwise() - col_max;
    shifted = shifted.array().exp().matrix();
    Eigen::RowVectorXf col_sum = shifted.colwise().sum();
    result.map() = shifted.array().rowwise() / col_sum.array();
  } else {
    Eigen::VectorXf row_max = x.map().rowwise().maxCoeff();
    Eigen::MatrixXf shifted = x.map().colwise() - row_max;
    shifted = shifted.array().exp().matrix();
    Eigen::VectorXf row_sum = shifted.rowwise().sum();
    result.map() = shifted.array().colwise() / row_sum.array();
  }
  return result;
}

Tensor Executor::unbroadcast(const Tensor &x, uint32_t target_rows,
                             uint32_t target_cols) {
  if (x.rows() == target_rows && x.cols() == target_cols)
    return x;
  Tensor result(target_rows, target_cols);
  if (target_rows == 1 && target_cols == 1)
    result(0, 0) = x.map().sum();
  else if (target_rows == 1)
    result.map() = x.map().colwise().sum();
  else if (target_cols == 1)
    result.map() = x.map().rowwise().sum();
  else
    result.map() = x.map();
  return result;
}

Tensor Executor::scatter_like(const Tensor &x, uint32_t target_rows,
                              uint32_t target_cols, int32_t index) {
  Tensor result(target_rows, target_cols);
  result.map().row(index) = x.map().row(0);
  return result;
}

} // namespace autograd
