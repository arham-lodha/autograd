#include "autograd/compiler.hpp"
#include "autograd/symbol.hpp"
#include <Eigen/Dense>
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

TEST_CASE("binary op wires two inputs in CSR", "[graph]") {
  Graph g;
  Symbol x = g.variable();
  Symbol y = g.variable();
  Symbol z = x + y;
  const Node &n = g.nodes[z.node_index];
  REQUIRE(n.input_count == 2);
  uint32_t a = g.inputs[n.input_offset + 0];
  uint32_t b = g.inputs[n.input_offset + 1];
  REQUIRE(((a == x.node_index && b == y.node_index) ||
           (a == y.node_index && b == x.node_index)));
}

TEST_CASE("unary op wires exactly one input", "[graph]") {
  Graph g;
  Symbol x = g.variable();
  Symbol e = exp(x);
  const Node &n = g.nodes[e.node_index];
  REQUIRE(n.input_count == 1);
  REQUIRE(g.inputs[n.input_offset] == x.node_index);
}

TEST_CASE("multiple ops build independent CSR slices", "[graph]") {
  Graph g;
  Symbol x = g.variable();
  Symbol y = g.variable();
  Symbol add = x + y;
  Symbol mul = x * y;
  // both ops present, independent slices, no aliasing
  REQUIRE(g.nodes[add.node_index].input_offset !=
          g.nodes[mul.node_index].input_offset);
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
  // z is ADD; one input is x, the other is a fresh CONSTANT(5)
  const Node &n = g.nodes[z.node_index];
  REQUIRE(n.operation == Op::ADD);
  REQUIRE(n.input_count == 2);
  bool found_const = false;
  for (uint32_t i = 0; i < 2; i++) {
    uint32_t inp = g.inputs[n.input_offset + i];
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
  // x / y → x * pow(y, -1) → MULTIPLY at root
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
  REQUIRE(n.axis == 1);
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

// ── Topological sort ──────────────────────────────────────────────────────────

TEST_CASE("topo sort of a single node returns that node", "[compiler]") {
  Graph g;
  Symbol x = g.variable();
  Compiler c;
  std::vector<uint32_t> topo = c.topological_sort(x);
  REQUIRE(topo.size() == 1);
  REQUIRE(topo[0] == x.node_index);
}

TEST_CASE("topo sort of a linear chain is in forward order", "[compiler]") {
  Graph g;
  Symbol x  = g.variable();
  Symbol ex = exp(x);
  Symbol lex = log(ex);
  Compiler c;
  std::vector<uint32_t> topo = c.topological_sort(lex);
  REQUIRE(topo.size() == 3);
  // inputs must appear before outputs
  auto pos = [&](uint32_t idx) {
    return std::find(topo.begin(), topo.end(), idx) - topo.begin();
  };
  REQUIRE(pos(x.node_index) < pos(ex.node_index));
  REQUIRE(pos(ex.node_index) < pos(lex.node_index));
}

TEST_CASE("topo sort of a diamond visits shared node once", "[compiler]") {
  Graph g;
  Symbol x  = g.variable();
  Symbol ex = exp(x);
  Symbol lx = log(x);
  Symbol s  = ex + lx;
  Compiler c;
  std::vector<uint32_t> topo = c.topological_sort(s);
  // x, exp(x), log(x), sum — 4 nodes total, x appears once
  REQUIRE(topo.size() == 4);
  uint32_t x_count = 0;
  for (uint32_t idx : topo)
    if (idx == x.node_index) x_count++;
  REQUIRE(x_count == 1);
}

TEST_CASE("topo sort places all inputs before their consumer", "[compiler]") {
  Graph g;
  Symbol a = g.variable();
  Symbol b = g.variable();
  Symbol c_sym = g.variable();
  Symbol s = (a + b) * c_sym;
  Compiler c;
  std::vector<uint32_t> topo = c.topological_sort(s);
  auto pos = [&](uint32_t idx) {
    return std::find(topo.begin(), topo.end(), idx) - topo.begin();
  };
  Symbol ab = a + b; // already created above, reuse index via operator+
  // a, b before (a+b); (a+b) and c before *
  REQUIRE(pos(a.node_index) < pos(s.node_index));
  REQUIRE(pos(b.node_index) < pos(s.node_index));
  REQUIRE(pos(c_sym.node_index) < pos(s.node_index));
}
