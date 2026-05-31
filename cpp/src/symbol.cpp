#include "autograd/symbol.hpp"
#include <cassert>
#include <span>

namespace autograd {

// ── Helper
// ────────────────────────────────────────────────────────────────────

static Symbol set_shape(Symbol result, const std::span<int> &shape) {
  uint32_t shape_offset = result.graph->shapes.size();
  for (int s : shape)
    result.graph->shapes.push_back(s);
  result.graph->nodes[result.node_index].shape.offset = shape_offset;
  result.graph->nodes[result.node_index].shape.count =
      static_cast<uint16_t>(shape.size());
  return result;
}

// ── operator[] ───────────────────────────────────────────────────────────────

Symbol Symbol::operator[](int index) const {
  assert(graph != nullptr && "Symbol must be associated with a graph");
  return graph->add_node(Op::GET_ITEM, {node_index}, index);
}

// ── Addition
// ──────────────────────────────────────────────────────────────────

Symbol operator+(Symbol lhs, Symbol rhs) {
  assert(lhs.graph == rhs.graph && "Symbols must belong to the same graph");
  assert(lhs.graph != nullptr && "Symbols must be associated with a graph");
  return lhs.graph->add_node(Op::ADD, {lhs.node_index, rhs.node_index});
}

Symbol operator+(Symbol lhs, float rhs) {
  return lhs + lhs.graph->constant(rhs);
}
Symbol operator+(float lhs, Symbol rhs) { return rhs + lhs; }
Symbol operator+(Symbol lhs, const Eigen::MatrixXf &rhs) {
  return lhs + lhs.graph->constant(rhs);
}
Symbol operator+(const Eigen::MatrixXf &lhs, Symbol rhs) { return rhs + lhs; }

// ── Multiplication
// ────────────────────────────────────────────────────────────

Symbol operator*(Symbol lhs, Symbol rhs) {
  assert(lhs.graph == rhs.graph && "Symbols must belong to the same graph");
  assert(lhs.graph != nullptr && "Symbols must be associated with a graph");
  return lhs.graph->add_node(Op::MULTIPLY, {lhs.node_index, rhs.node_index});
}

Symbol operator*(Symbol lhs, float rhs) {
  return lhs * lhs.graph->constant(rhs);
}
Symbol operator*(float lhs, Symbol rhs) { return rhs * lhs; }
Symbol operator*(Symbol lhs, const Eigen::MatrixXf &rhs) {
  return lhs * lhs.graph->constant(rhs);
}
Symbol operator*(const Eigen::MatrixXf &lhs, Symbol rhs) { return rhs * lhs; }

// ── Negation / Subtraction
// ────────────────────────────────────────────────────

Symbol operator-(Symbol x) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  return x.graph->add_node(Op::NEG, {x.node_index});
}

Symbol operator-(Symbol lhs, Symbol rhs) {
  assert(lhs.graph == rhs.graph && "Symbols must belong to the same graph");
  assert(lhs.graph != nullptr && "Symbols must be associated with a graph");
  return lhs.graph->add_node(Op::SUB, {lhs.node_index, rhs.node_index});
}

Symbol operator-(Symbol lhs, float rhs) {
  return lhs - lhs.graph->constant(rhs);
}
Symbol operator-(float lhs, Symbol rhs) {
  return rhs.graph->constant(lhs) - rhs;
}
Symbol operator-(Symbol lhs, const Eigen::MatrixXf &rhs) {
  return lhs - lhs.graph->constant(rhs);
}
Symbol operator-(const Eigen::MatrixXf &lhs, Symbol rhs) {
  return rhs.graph->constant(lhs) - rhs;
}

// ── Division
// ────────────────────────────────────────────────────────────────── No DIV op
// — mirrors Python: lhs / rhs = lhs * pow(rhs, -1)

Symbol operator/(Symbol lhs, Symbol rhs) {
  assert(lhs.graph == rhs.graph && "Symbols must belong to the same graph");
  assert(lhs.graph != nullptr && "Symbols must be associated with a graph");
  return lhs * pow(rhs, -1.0f);
}

Symbol operator/(Symbol lhs, float rhs) {
  return lhs / lhs.graph->constant(rhs);
}
Symbol operator/(float lhs, Symbol rhs) {
  return rhs.graph->constant(lhs) / rhs;
}
Symbol operator/(Symbol lhs, const Eigen::MatrixXf &rhs) {
  return lhs / lhs.graph->constant(rhs);
}
Symbol operator/(const Eigen::MatrixXf &lhs, Symbol rhs) {
  return rhs.graph->constant(lhs) / rhs;
}

// ── Power
// ─────────────────────────────────────────────────────────────────────

Symbol pow(Symbol base, Symbol exp) {
  assert(base.graph == exp.graph && "Symbols must belong to the same graph");
  assert(base.graph != nullptr && "Symbols must be associated with a graph");
  return base.graph->add_node(Op::POWER, {base.node_index, exp.node_index});
}

Symbol pow(Symbol base, float exp) {
  return pow(base, base.graph->constant(exp));
}
Symbol pow(float base, Symbol exp) {
  return pow(exp.graph->constant(base), exp);
}

// ── Matmul
// ────────────────────────────────────────────────────────────────────

Symbol matmul(Symbol lhs, Symbol rhs) {
  assert(lhs.graph == rhs.graph && "Symbols must belong to the same graph");
  assert(lhs.graph != nullptr && "Symbols must be associated with a graph");
  return lhs.graph->add_node(Op::MATMUL, {lhs.node_index, rhs.node_index});
}

Symbol matmul(Symbol lhs, const Eigen::MatrixXf &rhs) {
  return matmul(lhs, lhs.graph->constant(rhs));
}
Symbol matmul(const Eigen::MatrixXf &lhs, Symbol rhs) {
  return matmul(rhs.graph->constant(lhs), rhs);
}

// ── Unary
// ─────────────────────────────────────────────────────────────────────

Symbol exp(Symbol x) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  return x.graph->add_node(Op::EXP, {x.node_index});
}
Symbol log(Symbol x) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  return x.graph->add_node(Op::LOG, {x.node_index});
}
Symbol relu(Symbol x) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  return x.graph->add_node(Op::RELU, {x.node_index});
}
Symbol sqrt(Symbol x) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  return x.graph->add_node(Op::SQRT, {x.node_index});
}
Symbol abs(Symbol x) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  return x.graph->add_node(Op::ABS, {x.node_index});
}
Symbol sign(Symbol x) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  return x.graph->add_node(Op::SIGN, {x.node_index});
}
Symbol transpose(Symbol x) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  return x.graph->add_node(Op::TRANSPOSE, {x.node_index});
}

// ── Reductions
// ────────────────────────────────────────────────────────────────

Symbol sum(Symbol x, std::optional<int> axis, bool keepdims) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  int32_t ax = axis.value_or(-1);
  return x.graph->add_node(Op::SUM, {x.node_index}, ax, UINT32_MAX, keepdims);
}
Symbol mean(Symbol x, std::optional<int> axis, bool keepdims) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  int32_t ax = axis.value_or(-1);
  return x.graph->add_node(Op::MEAN, {x.node_index}, ax, UINT32_MAX, keepdims);
}
Symbol variance(Symbol x, std::optional<int> axis, bool keepdims) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  int32_t ax = axis.value_or(-1);
  return x.graph->add_node(Op::VARIANCE, {x.node_index}, ax, UINT32_MAX,
                           keepdims);
}
Symbol softmax(Symbol x, std::optional<int> axis) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  int32_t ax = axis.value_or(-1);
  return x.graph->add_node(Op::SOFTMAX, {x.node_index}, ax);
}
Symbol size(Symbol x, std::optional<int> axis) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  int32_t ax = axis.value_or(-1);
  return x.graph->add_node(Op::SIZE, {x.node_index}, ax);
}

// ── Shape
// ─────────────────────────────────────────────────────────────────────

Symbol broadcast_to(Symbol x, std::span<int> shape) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  Symbol result = x.graph->add_node(Op::BROADCAST_TO, {x.node_index});
  return set_shape(result, shape);
}
Symbol broadcast_to_match(Symbol x, Symbol other) {
  assert(x.graph == other.graph && "Symbols must belong to the same graph");
  assert(x.graph != nullptr && "Symbols must be associated with a graph");
  return x.graph->add_node(Op::BROADCAST_TO_MATCH,
                           {x.node_index, other.node_index});
}
Symbol reshape(Symbol x, std::span<int> shape) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  Symbol result = x.graph->add_node(Op::RESHAPE, {x.node_index});
  return set_shape(result, shape);
}
Symbol reshape_like(Symbol x, Symbol other) {
  assert(x.graph == other.graph && "Symbols must belong to the same graph");
  assert(x.graph != nullptr && "Symbols must be associated with a graph");
  return x.graph->add_node(Op::RESHAPE_LIKE, {x.node_index, other.node_index});
}
Symbol expand_dims(Symbol x, int axis) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  return x.graph->add_node(Op::EXPAND_DIMS, {x.node_index}, axis);
}
Symbol squeeze(Symbol x, std::optional<int> axis) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  int32_t ax = axis.value_or(-1);
  return x.graph->add_node(Op::SQUEEZE, {x.node_index}, ax);
}
Symbol swap_axis(Symbol x, int axis1, int axis2) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  return x.graph->add_node(Op::SWAP_AXIS, {x.node_index}, axis1, axis2);
}
Symbol unbroadcast(Symbol x, Symbol other) {
  assert(x.graph == other.graph && "Symbols must belong to the same graph");
  assert(x.graph != nullptr && "Symbols must be associated with a graph");
  return x.graph->add_node(Op::UNBROADCAST, {x.node_index, other.node_index});
}

// ── Comparison
// ────────────────────────────────────────────────────────────────

Symbol operator==(Symbol lhs, Symbol rhs) {
  assert(lhs.graph == rhs.graph && "Symbols must belong to the same graph");
  assert(lhs.graph != nullptr && "Symbols must be associated with a graph");
  return lhs.graph->add_node(Op::EQUALS_TO, {lhs.node_index, rhs.node_index});
}
Symbol operator==(Symbol lhs, float rhs) {
  return lhs == lhs.graph->constant(rhs);
}
Symbol operator==(float lhs, Symbol rhs) { return rhs == lhs; }

Symbol operator>(Symbol lhs, Symbol rhs) {
  assert(lhs.graph == rhs.graph && "Symbols must belong to the same graph");
  assert(lhs.graph != nullptr && "Symbols must be associated with a graph");
  return lhs.graph->add_node(Op::GREATER_THAN,
                             {lhs.node_index, rhs.node_index});
}
Symbol operator>(Symbol lhs, float rhs) {
  return lhs > lhs.graph->constant(rhs);
}
Symbol operator>(float lhs, Symbol rhs) {
  return rhs.graph->constant(lhs) > rhs;
}

Symbol operator>=(Symbol lhs, Symbol rhs) {
  assert(lhs.graph == rhs.graph && "Symbols must belong to the same graph");
  assert(lhs.graph != nullptr && "Symbols must be associated with a graph");
  return lhs.graph->add_node(Op::GREATER_THAN_OR_EQUAL,
                             {lhs.node_index, rhs.node_index});
}
Symbol operator>=(Symbol lhs, float rhs) {
  return lhs >= lhs.graph->constant(rhs);
}
Symbol operator>=(float lhs, Symbol rhs) {
  return rhs.graph->constant(lhs) >= rhs;
}

Symbol operator<(Symbol lhs, Symbol rhs) {
  assert(lhs.graph == rhs.graph && "Symbols must belong to the same graph");
  assert(lhs.graph != nullptr && "Symbols must be associated with a graph");
  return lhs.graph->add_node(Op::LESS_THAN, {lhs.node_index, rhs.node_index});
}
Symbol operator<(Symbol lhs, float rhs) {
  return lhs < lhs.graph->constant(rhs);
}
Symbol operator<(float lhs, Symbol rhs) {
  return rhs.graph->constant(lhs) < rhs;
}

Symbol operator<=(Symbol lhs, Symbol rhs) {
  assert(lhs.graph == rhs.graph && "Symbols must belong to the same graph");
  assert(lhs.graph != nullptr && "Symbols must be associated with a graph");
  return lhs.graph->add_node(Op::LESS_THAN_OR_EQUAL,
                             {lhs.node_index, rhs.node_index});
}
Symbol operator<=(Symbol lhs, float rhs) {
  return lhs <= lhs.graph->constant(rhs);
}
Symbol operator<=(float lhs, Symbol rhs) {
  return rhs.graph->constant(lhs) <= rhs;
}

// ── Indexing
// ──────────────────────────────────────────────────────────────────

Symbol get_item(Symbol x, int index) {
  assert(x.graph != nullptr && "Symbol must be associated with a graph");
  return x.graph->add_node(Op::GET_ITEM, {x.node_index},
                           static_cast<uint32_t>(index));
}
Symbol scatter_like(Symbol x, Symbol other, int index) {
  assert(x.graph == other.graph && "Symbols must belong to the same graph");
  assert(x.graph != nullptr && "Symbols must be associated with a graph");
  return x.graph->add_node(Op::SCATTER_LIKE, {x.node_index, other.node_index},
                           static_cast<uint32_t>(index));
}

} // namespace autograd
