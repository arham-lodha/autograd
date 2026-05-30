#include "autograd/compiler.hpp"
#include "autograd/symbol.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace autograd;

// ── Graph node construction ───────────────────────────────────────────────────

TEST_CASE("variable has VARIABLE op and requires_grad", "[graph]") {
  Graph g;
  Symbol x = g.variable();
  REQUIRE(g.nodes[x.node_index].operation == Op::VARIABLE);
  REQUIRE(g.nodes[x.node_index].requires_grad == true);
}

TEST_CASE("scalar constant has CONSTANT op, correct value, no grad", "[graph]") {
  Graph g;
  Symbol c = g.constant(3.0f);
  const Node &n = g.nodes[c.node_index];
  REQUIRE(n.operation == Op::CONSTANT);
  REQUIRE(n.requires_grad == false);
  REQUIRE(g.values[n.value_index](0, 0) == Catch::Approx(3.0f));
}

TEST_CASE("matrix constant stores correct shape and values", "[graph]") {
  Graph g;
  Eigen::MatrixXf m(2, 3);
  m << 1, 2, 3, 4, 5, 6;
  Symbol c = g.constant(m);
  const Eigen::MatrixXf &stored = g.values[g.nodes[c.node_index].value_index];
  REQUIRE(stored.rows() == 2);
  REQUIRE(stored.cols() == 3);
  REQUIRE(stored(0, 0) == Catch::Approx(1.0f));
  REQUIRE(stored(1, 2) == Catch::Approx(6.0f));
}

TEST_CASE("placeholder has PLACEHOLDER op", "[graph]") {
  Graph g;
  Symbol p = g.placeholder();
  REQUIRE(g.nodes[p.node_index].operation == Op::PLACEHOLDER);
}

// ── CSR wiring ────────────────────────────────────────────────────────────────
// Binary ops (input_count == 2) store both inputs inline in node.inputs[0/1].
// The g.inputs CSR pool is only used for n-ary ops with more than 2 children.

TEST_CASE("binary op wires two inputs inline", "[graph]") {
  Graph g;
  Symbol x = g.variable();
  Symbol y = g.variable();
  Symbol z = x + y;
  const Node &n = g.nodes[z.node_index];
  REQUIRE(n.input_count == 2);
  uint32_t a = n.inputs[0];
  uint32_t b = n.inputs[1];
  REQUIRE(((a == x.node_index && b == y.node_index) ||
           (a == y.node_index && b == x.node_index)));
}

TEST_CASE("unary op wires exactly one input inline", "[graph]") {
  Graph g;
  Symbol x = g.variable();
  Symbol e = exp(x);
  const Node &n = g.nodes[e.node_index];
  REQUIRE(n.input_count == 1);
  REQUIRE(n.inputs[0] == x.node_index);
}

TEST_CASE("multiple ops build independent nodes", "[graph]") {
  Graph g;
  Symbol x   = g.variable();
  Symbol y   = g.variable();
  Symbol add = x + y;
  Symbol mul = x * y;
  REQUIRE(add.node_index != mul.node_index);
  REQUIRE(g.nodes[add.node_index].input_count == 2);
  REQUIRE(g.nodes[mul.node_index].input_count == 2);
}

// ── Operator overloads ────────────────────────────────────────────────────────

TEST_CASE("arithmetic operators produce correct ops", "[symbol]") {
  Graph g;
  Symbol x = g.variable();
  Symbol y = g.variable();
  REQUIRE(g.nodes[(x + y).node_index].operation == Op::ADD);
  REQUIRE(g.nodes[(x * y).node_index].operation == Op::MULTIPLY);
  REQUIRE(g.nodes[(x - y).node_index].operation == Op::SUB);
  REQUIRE(g.nodes[(-x).node_index].operation == Op::NEG);
  REQUIRE(g.nodes[pow(x, y).node_index].operation == Op::POWER);
  REQUIRE(g.nodes[matmul(x, y).node_index].operation == Op::MATMUL);
}

TEST_CASE("float rhs operators insert a constant node", "[symbol]") {
  Graph g;
  Symbol x = g.variable();
  Symbol z = x + 5.0f;
  const Node &n = g.nodes[z.node_index];
  REQUIRE(n.operation == Op::ADD);
  REQUIRE(n.input_count == 2);
  bool found_const = false;
  for (uint32_t i = 0; i < 2; i++) {
    uint32_t inp = n.inputs[i];
    if (g.nodes[inp].operation == Op::CONSTANT)
      found_const = true;
  }
  REQUIRE(found_const);
}

TEST_CASE("division is implemented as multiply-by-reciprocal", "[symbol]") {
  Graph g;
  Symbol x = g.variable();
  Symbol y = g.variable();
  Symbol z = x / y;
  REQUIRE(g.nodes[z.node_index].operation == Op::MULTIPLY);
}

TEST_CASE("unary ops produce correct ops", "[symbol]") {
  Graph g;
  Symbol x = g.variable();
  REQUIRE(g.nodes[exp(x).node_index].operation == Op::EXP);
  REQUIRE(g.nodes[log(x).node_index].operation == Op::LOG);
  REQUIRE(g.nodes[relu(x).node_index].operation == Op::RELU);
  REQUIRE(g.nodes[sqrt(x).node_index].operation == Op::SQRT);
  REQUIRE(g.nodes[abs(x).node_index].operation == Op::ABS);
  REQUIRE(g.nodes[sign(x).node_index].operation == Op::SIGN);
  REQUIRE(g.nodes[transpose(x).node_index].operation == Op::TRANSPOSE);
}

TEST_CASE("comparison operators produce correct ops", "[symbol]") {
  Graph g;
  Symbol x = g.variable();
  Symbol y = g.variable();
  REQUIRE(g.nodes[(x > y).node_index].operation == Op::GREATER_THAN);
  REQUIRE(g.nodes[(x < y).node_index].operation == Op::LESS_THAN);
}

TEST_CASE("reduction ops store axis correctly", "[symbol]") {
  Graph g;
  Symbol x = g.variable();
  Symbol s = sum(x, 1, true);
  const Node &n = g.nodes[s.node_index];
  REQUIRE(n.operation == Op::SUM);
  REQUIRE(n.axes.axis == 1);
  REQUIRE(n.keepdims == true);
}

// ── requires_grad propagation ─────────────────────────────────────────────────

TEST_CASE("op with a variable input requires grad", "[symbol]") {
  Graph g;
  Symbol x = g.variable();
  Symbol c = g.constant(2.0f);
  REQUIRE(g.nodes[(x + c).node_index].requires_grad == true);
  REQUIRE(g.nodes[(x * c).node_index].requires_grad == true);
  REQUIRE(g.nodes[exp(x).node_index].requires_grad == true);
}

TEST_CASE("op of two constants does not require grad", "[symbol]") {
  Graph g;
  Symbol a = g.constant(2.0f);
  Symbol b = g.constant(3.0f);
  REQUIRE(g.nodes[(a + b).node_index].requires_grad == false);
  REQUIRE(g.nodes[(a * b).node_index].requires_grad == false);
}

TEST_CASE("grad propagates through a chain", "[symbol]") {
  Graph g;
  Symbol x = g.variable();
  Symbol y = exp(log(x));
  REQUIRE(g.nodes[y.node_index].requires_grad == true);
}

// ── Topological ordering via Compiler::compile ────────────────────────────────
// Phase 3 of the compiler emits nodes in strict post-order (children before
// parents), so we verify ordering by inspecting Program::nodes.

TEST_CASE("compiled single-node program has exactly one node", "[compiler]") {
  Graph g;
  Symbol x = g.variable();
  // Zero optimization passes so no constants are injected.
  Program p = Compiler::compile(g, {x}, {x}, {.optimization_passes = 0});
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].operation == Op::VARIABLE);
}

TEST_CASE("compiled linear chain places input before output", "[compiler]") {
  Graph g;
  Symbol x   = g.variable();
  Symbol ex  = exp(x);
  Symbol lex = log(ex);
  Program p  = Compiler::compile(g, {x}, {lex}, {.optimization_passes = 0});

  // Find positions of x, exp(x), log(exp(x)) in the emitted program.
  auto find_op = [&](Op op, uint32_t after = 0) -> uint32_t {
    for (uint32_t i = after; i < p.nodes.size(); i++)
      if (p.nodes[i].operation == op) return i;
    return UINT32_MAX;
  };

  uint32_t xi   = find_op(Op::VARIABLE);
  uint32_t exi  = find_op(Op::EXP);
  uint32_t lexi = find_op(Op::LOG);

  REQUIRE(xi   != UINT32_MAX);
  REQUIRE(exi  != UINT32_MAX);
  REQUIRE(lexi != UINT32_MAX);
  REQUIRE(xi < exi);
  REQUIRE(exi < lexi);
}

TEST_CASE("compiled diamond visits shared node once", "[compiler]") {
  Graph g;
  Symbol x  = g.variable();
  Symbol ex = exp(x);
  Symbol lx = log(x);
  Symbol s  = ex + lx;
  Program p = Compiler::compile(g, {x}, {s}, {.optimization_passes = 0});

  // Exactly one VARIABLE node.
  uint32_t var_count = 0;
  for (const Node &n : p.nodes)
    if (n.operation == Op::VARIABLE) var_count++;
  REQUIRE(var_count == 1);
}

TEST_CASE("compiled program contains no ALIAS nodes", "[compiler]") {
  Graph g;
  Symbol x = g.variable();
  Symbol y = x + g.constant(0.0f); // should be simplified away
  Program p = Compiler::compile(g, {x}, {y});
  for (const Node &n : p.nodes)
    REQUIRE(n.operation != Op::ALIAS);
}
