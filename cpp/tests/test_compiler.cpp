#include "autograd/compiler.hpp"
#include "autograd/symbol.hpp"
#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace autograd;

// ── Helpers ───────────────────────────────────────────────────────────────────

static uint32_t count_op(const Program &prog, Op op) {
  uint32_t n = 0;
  for (const Node &node : prog.nodes)
    if (node.operation == op) n++;
  return n;
}

static bool has_op(const Program &prog, Op op) {
  return count_op(prog, op) > 0;
}

// Verify that a compiled program with a single CONSTANT node holds `expected`.
static float sole_constant(const Program &prog) {
  REQUIRE(prog.nodes.size() == 1);
  REQUIRE(prog.nodes[0].operation == Op::CONSTANT);
  return prog.values[prog.nodes[0].value_index](0, 0);
}

// Convenience: compile a single-output graph with default optimisation.
static Program compile1(Graph &g, Symbol x, Symbol out,
                        CompilerConfig cfg = {}) {
  return Compiler::compile(g, {x}, {out}, cfg);
}

// ── Subgraph extraction ───────────────────────────────────────────────────────

TEST_CASE("compile: single variable produces one-node program", "[extract]") {
  Graph g;
  Symbol x = g.variable();
  Program p = compile1(g, x, x, {.optimization_passes = 0});
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].operation == Op::VARIABLE);
}

TEST_CASE("compile: only reachable nodes are extracted", "[extract]") {
  Graph g;
  Symbol x  = g.variable();
  Symbol y  = g.variable();
  Symbol ex = exp(x);
  // y is not reachable from ex
  Program p = compile1(g, x, ex, {.optimization_passes = 0});
  // Should contain VARIABLE(x) and EXP, not VARIABLE(y)
  REQUIRE(count_op(p, Op::VARIABLE) == 1);
  REQUIRE(has_op(p, Op::EXP));
}

TEST_CASE("compile: diamond dependency visits shared node once", "[extract]") {
  Graph g;
  Symbol x  = g.variable();
  Symbol ex = exp(x);
  Symbol lx = log(x);
  Symbol s  = ex + lx;
  Program p = compile1(g, x, s, {.optimization_passes = 0});
  REQUIRE(count_op(p, Op::VARIABLE) == 1);
}

TEST_CASE("compile: no ALIAS nodes survive into Program", "[extract]") {
  Graph g;
  Symbol x = g.variable();
  Symbol z = x + g.constant(0.0f);
  Program p = Compiler::compile(g, {x}, {z});
  REQUIRE(!has_op(p, Op::ALIAS));
}

TEST_CASE("compile: output_nodes points at the last emitted node", "[extract]") {
  Graph g;
  Symbol x  = g.variable();
  Symbol ex = exp(x);
  Program p = compile1(g, x, ex, {.optimization_passes = 0});
  REQUIRE(p.output_nodes.size() == 1);
  REQUIRE(p.nodes[p.output_nodes[0]].operation == Op::EXP);
}

// ── Topological ordering ──────────────────────────────────────────────────────

TEST_CASE("compile: children always precede parents in output", "[topo]") {
  Graph g;
  Symbol x   = g.variable();
  Symbol ex  = exp(x);
  Symbol lex = log(ex);
  Program p  = compile1(g, x, lex, {.optimization_passes = 0});

  auto pos = [&](Op op) -> uint32_t {
    for (uint32_t i = 0; i < p.nodes.size(); i++)
      if (p.nodes[i].operation == op) return i;
    return UINT32_MAX;
  };

  REQUIRE(pos(Op::VARIABLE) < pos(Op::EXP));
  REQUIRE(pos(Op::EXP) < pos(Op::LOG));
}

// ── Canonicalize pass ─────────────────────────────────────────────────────────

TEST_CASE("canonicalize: NEG is rewritten to MULTIPLY", "[canonicalize]") {
  Graph g;
  Symbol x = g.variable();
  // Canonicalize converts NEG→MULTIPLY; run just 1 pass so we see the change
  // before any further simplification.
  Program p = compile1(g, x, -x, {.optimization_passes = 1,
                                   .enable_constant_folding = false});
  REQUIRE(!has_op(p, Op::NEG));
  REQUIRE(has_op(p, Op::MULTIPLY));
}

TEST_CASE("canonicalize: SUB is rewritten to ADD + MULTIPLY", "[canonicalize]") {
  Graph g;
  Symbol x = g.variable();
  Symbol y = g.variable();
  Program p = Compiler::compile(g, {x, y}, {x - y},
                                {.optimization_passes = 1,
                                 .enable_constant_folding = false});
  REQUIRE(!has_op(p, Op::SUB));
  REQUIRE(has_op(p, Op::ADD));
  REQUIRE(has_op(p, Op::MULTIPLY));
}

TEST_CASE("canonicalize: SQRT is rewritten to POWER", "[canonicalize]") {
  Graph g;
  Symbol x = g.variable();
  Program p = compile1(g, x, sqrt(x),
                       {.optimization_passes = 1,
                        .enable_constant_folding = false});
  REQUIRE(!has_op(p, Op::SQRT));
  REQUIRE(has_op(p, Op::POWER));
}

// ── Addition / multiplication folding ────────────────────────────────────────

TEST_CASE("fold: (a+b)+c flattens to a single ADD with 3 inputs", "[fold]") {
  Graph g;
  Symbol a = g.variable();
  Symbol b = g.variable();
  Symbol c = g.variable();
  Symbol z = (a + b) + c;
  Program p = Compiler::compile(g, {a, b, c}, {z},
                                {.optimization_passes = 2,
                                 .enable_constant_folding = false});
  // There should be exactly one ADD node with input_count == 3.
  bool found = false;
  for (const Node &n : p.nodes)
    if (n.operation == Op::ADD && n.input_count == 3) found = true;
  REQUIRE(found);
}

TEST_CASE("fold: (a*b)*c flattens to a single MULTIPLY with 3 inputs", "[fold]") {
  Graph g;
  Symbol a = g.variable();
  Symbol b = g.variable();
  Symbol c = g.variable();
  Symbol z = (a * b) * c;
  Program p = Compiler::compile(g, {a, b, c}, {z},
                                {.optimization_passes = 2,
                                 .enable_constant_folding = false});
  bool found = false;
  for (const Node &n : p.nodes)
    if (n.operation == Op::MULTIPLY && n.input_count == 3) found = true;
  REQUIRE(found);
}

// ── Constant folding ──────────────────────────────────────────────────────────

TEST_CASE("constant_fold: 2 + 3 folds to CONSTANT(5)", "[constant_fold]") {
  Graph g;
  Symbol z = g.constant(2.0f) + g.constant(3.0f);
  // Use a placeholder as dummy input so compile1 accepts any symbol as input.
  Program p = Compiler::compile(g, {}, {z});
  REQUIRE(sole_constant(p) == Catch::Approx(5.0f));
}

TEST_CASE("constant_fold: 4 * 3 folds to CONSTANT(12)", "[constant_fold]") {
  Graph g;
  Symbol z = g.constant(4.0f) * g.constant(3.0f);
  Program p = Compiler::compile(g, {}, {z});
  REQUIRE(sole_constant(p) == Catch::Approx(12.0f));
}

TEST_CASE("constant_fold: 2^10 folds to CONSTANT(1024)", "[constant_fold]") {
  Graph g;
  Symbol z = pow(g.constant(2.0f), g.constant(10.0f));
  Program p = Compiler::compile(g, {}, {z});
  REQUIRE(sole_constant(p) == Catch::Approx(1024.0f));
}

TEST_CASE("constant_fold: variable input is not folded", "[constant_fold]") {
  Graph g;
  Symbol x = g.variable();
  Symbol z = x + g.constant(2.0f);
  Program p = compile1(g, x, z);
  REQUIRE(has_op(p, Op::ADD));
  REQUIRE(has_op(p, Op::VARIABLE));
}

// ── Algebraic simplification ──────────────────────────────────────────────────

TEST_CASE("algebra: x + 0 reduces to x", "[algebra]") {
  Graph g;
  Symbol x = g.variable();
  Program p = compile1(g, x, x + 0.0f);
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].operation == Op::VARIABLE);
}

TEST_CASE("algebra: x * 1 reduces to x", "[algebra]") {
  Graph g;
  Symbol x = g.variable();
  Program p = compile1(g, x, x * 1.0f);
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].operation == Op::VARIABLE);
}

TEST_CASE("algebra: x * 0 reduces to constant zero", "[algebra]") {
  Graph g;
  Symbol x = g.variable();
  Program p = compile1(g, x, x * 0.0f);
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].operation == Op::CONSTANT);
  REQUIRE(p.values[p.nodes[0].value_index](0, 0) == Catch::Approx(0.0f));
}

TEST_CASE("algebra: x^0 reduces to constant one", "[algebra]") {
  Graph g;
  Symbol x = g.variable();
  Program p = compile1(g, x, pow(x, 0.0f));
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].operation == Op::CONSTANT);
  REQUIRE(p.values[p.nodes[0].value_index](0, 0) == Catch::Approx(1.0f));
}

TEST_CASE("algebra: x^1 reduces to x", "[algebra]") {
  Graph g;
  Symbol x = g.variable();
  Program p = compile1(g, x, pow(x, 1.0f));
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].operation == Op::VARIABLE);
}

TEST_CASE("algebra: log(exp(x)) reduces to x", "[algebra]") {
  Graph g;
  Symbol x = g.variable();
  Program p = compile1(g, x, log(exp(x)));
  REQUIRE(!has_op(p, Op::LOG));
  REQUIRE(!has_op(p, Op::EXP));
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].operation == Op::VARIABLE);
}

TEST_CASE("algebra: exp(log(x)) reduces to x", "[algebra]") {
  Graph g;
  Symbol x = g.variable();
  Program p = compile1(g, x, exp(log(x)));
  REQUIRE(!has_op(p, Op::EXP));
  REQUIRE(!has_op(p, Op::LOG));
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].operation == Op::VARIABLE);
}
