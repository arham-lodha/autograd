#pragma once

#include "autograd/graph.hpp"
#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <vector>

namespace autograd {

struct Symbol {
  uint32_t node_index;
  Graph *graph;

  // operator[] is the one op that must be a member in C++
  Symbol operator[](int index) const;
};

// ── Arithmetic
// ────────────────────────────────────────────────────────────────
Symbol operator+(Symbol lhs, Symbol rhs);
Symbol operator+(Symbol lhs, float rhs);
Symbol operator+(float lhs, Symbol rhs);
Symbol operator+(Symbol lhs, const Eigen::MatrixXf &rhs);
Symbol operator+(const Eigen::MatrixXf &lhs, Symbol rhs);

Symbol operator*(Symbol lhs, Symbol rhs);
Symbol operator*(Symbol lhs, float rhs);
Symbol operator*(float lhs, Symbol rhs);
Symbol operator*(Symbol lhs, const Eigen::MatrixXf &rhs);
Symbol operator*(const Eigen::MatrixXf &lhs, Symbol rhs);

Symbol operator-(Symbol x); // unary negation
Symbol operator-(Symbol lhs, Symbol rhs);
Symbol operator-(Symbol lhs, float rhs);
Symbol operator-(float lhs, Symbol rhs);
Symbol operator-(Symbol lhs, const Eigen::MatrixXf &rhs);
Symbol operator-(const Eigen::MatrixXf &lhs, Symbol rhs);

Symbol operator/(Symbol lhs, Symbol rhs);
Symbol operator/(Symbol lhs, float rhs);
Symbol operator/(float lhs, Symbol rhs);
Symbol operator/(Symbol lhs, const Eigen::MatrixXf &rhs);
Symbol operator/(const Eigen::MatrixXf &lhs, Symbol rhs);

Symbol pow(Symbol base, Symbol exp);
Symbol pow(Symbol base, float exp);
Symbol pow(float base, Symbol exp);

Symbol matmul(Symbol lhs, Symbol rhs);
Symbol matmul(Symbol lhs, const Eigen::MatrixXf &rhs);
Symbol matmul(const Eigen::MatrixXf &lhs, Symbol rhs);

// ── Unary
// ─────────────────────────────────────────────────────────────────────
Symbol exp(Symbol x);
Symbol log(Symbol x);
Symbol relu(Symbol x);
Symbol sqrt(Symbol x);
Symbol abs(Symbol x);
Symbol sign(Symbol x);
Symbol transpose(Symbol x);

// ── Reductions
// ────────────────────────────────────────────────────────────────
Symbol sum(Symbol x, std::optional<int> axis = std::nullopt,
           bool keepdims = false);
Symbol mean(Symbol x, std::optional<int> axis = std::nullopt,
            bool keepdims = false);
Symbol variance(Symbol x, std::optional<int> axis = std::nullopt,
                bool keepdims = false);
Symbol softmax(Symbol x, std::optional<int> axis = std::nullopt);
Symbol size(Symbol x, std::optional<int> axis = std::nullopt);

// ── Shape
// ─────────────────────────────────────────────────────────────────────
Symbol broadcast_to(Symbol x, std::vector<int> shape);
Symbol broadcast_to_match(Symbol x, Symbol other);
Symbol reshape(Symbol x, std::vector<int> shape);
Symbol reshape_like(Symbol x, Symbol other);
Symbol expand_dims(Symbol x, int axis);
Symbol squeeze(Symbol x, std::optional<int> axis = std::nullopt);
Symbol swap_axis(Symbol x, int axis1, int axis2);
Symbol unbroadcast(Symbol x, Symbol other);

// ── Comparison
// ────────────────────────────────────────────────────────────────
Symbol operator==(Symbol lhs, Symbol rhs);
Symbol operator==(Symbol lhs, float rhs);
Symbol operator==(float lhs, Symbol rhs);
Symbol operator>(Symbol lhs, Symbol rhs);
Symbol operator>(Symbol lhs, float rhs);
Symbol operator>(float lhs, Symbol rhs);
Symbol operator>=(Symbol lhs, Symbol rhs);
Symbol operator>=(Symbol lhs, float rhs);
Symbol operator>=(float lhs, Symbol rhs);
Symbol operator<(Symbol lhs, Symbol rhs);
Symbol operator<(Symbol lhs, float rhs);
Symbol operator<(float lhs, Symbol rhs);
Symbol operator<=(Symbol lhs, Symbol rhs);
Symbol operator<=(Symbol lhs, float rhs);
Symbol operator<=(float lhs, Symbol rhs);

// ── Indexing
// ──────────────────────────────────────────────────────────────────
Symbol get_item(Symbol x, int index);
Symbol scatter_like(Symbol x, Symbol other, int index);

} // namespace autograd
