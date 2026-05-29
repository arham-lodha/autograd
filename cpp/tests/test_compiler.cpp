#include "autograd/compiler.hpp"
#include "autograd/symbol.hpp"
#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace autograd;

// ── Helpers ───────────────────────────────────────────────────────────────────

static uint32_t count_op(const IntermediateProgram &prog, Op op) {
  uint32_t n = 0;
  for (const IntermediateProgramNode &node : prog.nodes)
    if (node.op == op) n++;
  return n;
}

static uint32_t count_op(const Program &prog, Op op) {
  uint32_t n = 0;
  for (const ProgramNode &node : prog.nodes)
    if (node.op == op) n++;
  return n;
}

static bool has_op(const IntermediateProgram &prog, Op op) {
  return count_op(prog, op) > 0;
}
static bool has_op(const Program &prog, Op op) {
  return count_op(prog, op) > 0;
}

// Return the single constant value in a one-node folded program.
static float sole_constant(const Program &prog) {
  REQUIRE(prog.nodes.size() == 1);
  REQUIRE(prog.nodes[0].op == Op::CONSTANT);
  return prog.values[prog.nodes[0].value_index](0, 0);
}

// Build a Compiler IR directly from a graph output symbol.
static IntermediateProgram make_ir(Graph &g, Symbol out) {
  Compiler c;
  return c.make_ir(g, c.topological_sort(out));
}

// ── make_ir ───────────────────────────────────────────────────────────────────

TEST_CASE("make_ir preserves ops and node count", "[make_ir]") {
  Graph g;
  Symbol x = g.variable();
  Symbol y = g.variable();
  Symbol z = x + y;
  IntermediateProgram ir = make_ir(g, z);
  REQUIRE(ir.nodes.size() == 3);
  REQUIRE(has_op(ir, Op::VARIABLE));
  REQUIRE(has_op(ir, Op::ADD));
}

TEST_CASE("make_ir wires inputs correctly", "[make_ir]") {
  Graph g;
  Symbol x  = g.variable();
  Symbol ex = exp(x);
  IntermediateProgram ir = make_ir(g, ex);
  REQUIRE(ir.nodes.size() == 2);
  // EXP node (index 1) must reference VARIABLE (index 0)
  REQUIRE(ir.nodes[1].op == Op::EXP);
  REQUIRE(ir.nodes[1].inputs.size() == 1);
  REQUIRE(ir.nodes[1].inputs[0] == 0);
}

TEST_CASE("make_ir copies constant values", "[make_ir]") {
  Graph g;
  Symbol c = g.constant(7.0f);
  IntermediateProgram ir = make_ir(g, c);
  REQUIRE(ir.nodes.size() == 1);
  REQUIRE(ir.nodes[0].op == Op::CONSTANT);
  REQUIRE(ir.values[ir.nodes[0].value_index](0, 0) == Catch::Approx(7.0f));
}

// ── canonicalize ─────────────────────────────────────────────────────────────

TEST_CASE("canonicalize: NEG becomes MULTIPLY with -1 constant", "[canonicalize]") {
  Graph g;
  Symbol x = g.variable();
  Symbol nx = -x;
  Compiler c;
  IntermediateProgram ir   = make_ir(g, nx);
  IntermediateProgram can  = c.canonicalize(std::move(ir));
  REQUIRE(can.changed == true);
  REQUIRE(!has_op(can, Op::NEG));
  REQUIRE(has_op(can, Op::MULTIPLY));
  // The inserted -1 constant must be in the values pool
  bool found_neg_one = false;
  for (const Eigen::MatrixXf &v : can.values)
    if (v(0, 0) == Catch::Approx(-1.0f)) found_neg_one = true;
  REQUIRE(found_neg_one);
}

TEST_CASE("canonicalize: SUB becomes ADD + MULTIPLY(-1)", "[canonicalize]") {
  Graph g;
  Symbol x = g.variable();
  Symbol y = g.variable();
  Symbol s = x - y;
  Compiler c;
  IntermediateProgram can = c.canonicalize(make_ir(g, s));
  REQUIRE(can.changed == true);
  REQUIRE(!has_op(can, Op::SUB));
  REQUIRE(has_op(can, Op::ADD));
  REQUIRE(has_op(can, Op::MULTIPLY));
}

TEST_CASE("canonicalize: SQRT becomes POWER(x, 0.5)", "[canonicalize]") {
  Graph g;
  Symbol x = g.variable();
  Symbol s = sqrt(x);
  Compiler c;
  IntermediateProgram can = c.canonicalize(make_ir(g, s));
  REQUIRE(can.changed == true);
  REQUIRE(!has_op(can, Op::SQRT));
  REQUIRE(has_op(can, Op::POWER));
  bool found_half = false;
  for (const Eigen::MatrixXf &v : can.values)
    if (v(0, 0) == Catch::Approx(0.5f)) found_half = true;
  REQUIRE(found_half);
}

TEST_CASE("canonicalize: non-canonical ops are unchanged", "[canonicalize]") {
  Graph g;
  Symbol x = g.variable();
  Symbol y = g.variable();
  Symbol z = x + y;
  Compiler c;
  IntermediateProgram can = c.canonicalize(make_ir(g, z));
  // ADD is not canonicalized, changed should be false
  REQUIRE(can.changed == false);
  REQUIRE(has_op(can, Op::ADD));
}

// ── addition_folding ──────────────────────────────────────────────────────────

TEST_CASE("addition_folding: (a+b)+c flattens to ADD with 3 inputs", "[addition_folding]") {
  Graph g;
  Symbol a = g.variable();
  Symbol b = g.variable();
  Symbol cc = g.variable();
  Symbol z = (a + b) + cc;
  Compiler c;
  IntermediateProgram ir     = make_ir(g, z);
  IntermediateProgram folded = c.addition_folding(std::move(ir));
  REQUIRE(folded.changed == true);
  // Find the ADD node with 3 inputs
  bool found = false;
  for (const IntermediateProgramNode &node : folded.nodes)
    if (node.op == Op::ADD && node.inputs.size() == 3) found = true;
  REQUIRE(found);
}

TEST_CASE("addition_folding: single ADD is unchanged", "[addition_folding]") {
  Graph g;
  Symbol a = g.variable();
  Symbol b = g.variable();
  Symbol z = a + b;
  Compiler c;
  IntermediateProgram folded = c.addition_folding(make_ir(g, z));
  REQUIRE(folded.changed == false);
}

// ── multiplication_folding ────────────────────────────────────────────────────

TEST_CASE("multiplication_folding: (a*b)*c flattens to MULTIPLY with 3 inputs",
          "[multiplication_folding]") {
  Graph g;
  Symbol a = g.variable();
  Symbol b = g.variable();
  Symbol cc = g.variable();
  Symbol z = (a * b) * cc;
  Compiler c;
  IntermediateProgram folded = c.multiplication_folding(make_ir(g, z));
  REQUIRE(folded.changed == true);
  bool found = false;
  for (const IntermediateProgramNode &node : folded.nodes)
    if (node.op == Op::MULTIPLY && node.inputs.size() == 3) found = true;
  REQUIRE(found);
}

// ── constant_fold ─────────────────────────────────────────────────────────────

TEST_CASE("constant_fold: ADD of two scalars folds to correct value", "[constant_fold]") {
  Graph g;
  Symbol a = g.constant(2.0f);
  Symbol b = g.constant(3.0f);
  Symbol s = a + b;
  Compiler c;
  IntermediateProgram ir     = make_ir(g, s);
  IntermediateProgram folded = c.constant_fold(std::move(ir));
  REQUIRE(folded.changed == true);
  // Root node becomes CONSTANT(5)
  REQUIRE(folded.nodes.back().op == Op::CONSTANT);
  uint32_t vi = folded.nodes.back().value_index;
  REQUIRE(folded.values[vi](0, 0) == Catch::Approx(5.0f));
}

TEST_CASE("constant_fold: MULTIPLY of two scalars folds correctly", "[constant_fold]") {
  Graph g;
  Symbol a = g.constant(4.0f);
  Symbol b = g.constant(3.0f);
  Compiler c;
  IntermediateProgram folded = c.constant_fold(make_ir(g, a * b));
  REQUIRE(folded.nodes.back().op == Op::CONSTANT);
  REQUIRE(folded.values[folded.nodes.back().value_index](0, 0) == Catch::Approx(12.0f));
}

TEST_CASE("constant_fold: POWER of two constants folds correctly", "[constant_fold]") {
  Graph g;
  Compiler c;
  // 2^3 = 8; but SQRT is canonicalized to POWER, so use pow directly
  Symbol base = g.constant(2.0f);
  Symbol ep   = g.constant(3.0f);
  Symbol pw   = pow(base, ep);
  IntermediateProgram folded = c.constant_fold(make_ir(g, pw));
  REQUIRE(folded.nodes.back().op == Op::CONSTANT);
  REQUIRE(folded.values[folded.nodes.back().value_index](0, 0) == Catch::Approx(8.0f));
}

TEST_CASE("constant_fold: variable input is not folded", "[constant_fold]") {
  Graph g;
  Symbol x = g.variable();
  Symbol c_sym = g.constant(2.0f);
  Compiler c;
  IntermediateProgram folded = c.constant_fold(make_ir(g, x + c_sym));
  // Root must remain ADD (x is not constant)
  REQUIRE(folded.nodes.back().op == Op::ADD);
  REQUIRE(folded.changed == false);
}

TEST_CASE("constant_fold: two constants in n-ary ADD fold, variable stays",
          "[constant_fold]") {
  Graph g;
  Symbol x  = g.variable();
  Symbol c1 = g.constant(1.0f);
  Symbol c2 = g.constant(2.0f);
  // Build x + c1 + c2 without losing x
  Symbol s  = (x + c1) + c2;
  Compiler c;
  // Flatten first so ADD sees all 3 inputs
  IntermediateProgram ir     = make_ir(g, s);
  IntermediateProgram flat   = c.addition_folding(std::move(ir));
  IntermediateProgram folded = c.constant_fold(std::move(flat));
  REQUIRE(folded.changed == true);
  // Root is ADD(x, CONSTANT(3))
  REQUIRE(folded.nodes.back().op == Op::ADD);
  REQUIRE(folded.nodes.back().inputs.size() == 2);
}

// ── algebraic_simplification ──────────────────────────────────────────────────

TEST_CASE("algebraic_simplification: x + 0 reduces to x", "[algebraic_simplification]") {
  Graph g;
  Symbol x = g.variable();
  Symbol z = x + 0.0f;
  Program p = Compiler().compile(z);
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].op == Op::VARIABLE);
}

TEST_CASE("algebraic_simplification: x * 1 reduces to x", "[algebraic_simplification]") {
  Graph g;
  Symbol x = g.variable();
  Program p = Compiler().compile(x * 1.0f);
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].op == Op::VARIABLE);
}

TEST_CASE("algebraic_simplification: x * 0 reduces to constant zero",
          "[algebraic_simplification]") {
  Graph g;
  Symbol x = g.variable();
  Program p = Compiler().compile(x * 0.0f);
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].op == Op::CONSTANT);
  REQUIRE(p.values[p.nodes[0].value_index](0, 0) == Catch::Approx(0.0f));
}

TEST_CASE("algebraic_simplification: x^0 reduces to constant one",
          "[algebraic_simplification]") {
  Graph g;
  Symbol x = g.variable();
  Program p = Compiler().compile(pow(x, 0.0f));
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].op == Op::CONSTANT);
  REQUIRE(p.values[p.nodes[0].value_index](0, 0) == Catch::Approx(1.0f));
}

TEST_CASE("algebraic_simplification: x^1 reduces to x", "[algebraic_simplification]") {
  Graph g;
  Symbol x = g.variable();
  Program p = Compiler().compile(pow(x, 1.0f));
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].op == Op::VARIABLE);
}

TEST_CASE("algebraic_simplification: log(exp(x)) reduces to x",
          "[algebraic_simplification]") {
  Graph g;
  Symbol x = g.variable();
  Program p = Compiler().compile(log(exp(x)));
  REQUIRE(!has_op(p, Op::LOG));
  REQUIRE(!has_op(p, Op::EXP));
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].op == Op::VARIABLE);
}

TEST_CASE("algebraic_simplification: exp(log(x)) reduces to x",
          "[algebraic_simplification]") {
  Graph g;
  Symbol x = g.variable();
  Program p = Compiler().compile(exp(log(x)));
  REQUIRE(!has_op(p, Op::EXP));
  REQUIRE(!has_op(p, Op::LOG));
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].op == Op::VARIABLE);
}

TEST_CASE("algebraic_simplification: T(T(x)) reduces to x",
          "[algebraic_simplification]") {
  Graph g;
  Symbol x = g.variable();
  Program p = Compiler().compile(transpose(transpose(x)));
  REQUIRE(!has_op(p, Op::TRANSPOSE));
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].op == Op::VARIABLE);
}

TEST_CASE("algebraic_simplification: relu(relu(x)) reduces to one relu",
          "[algebraic_simplification]") {
  Graph g;
  Symbol x = g.variable();
  Program p = Compiler().compile(relu(relu(x)));
  REQUIRE(count_op(p, Op::RELU) == 1);
}

// ── decanonicalize ────────────────────────────────────────────────────────────

TEST_CASE("decanonicalize: MULTIPLY(x,-1) becomes NEG after full compile",
          "[decanonicalize]") {
  Graph g;
  Symbol x = g.variable();
  Program p = Compiler().compile(-x);
  REQUIRE(!has_op(p, Op::MULTIPLY));
  REQUIRE(has_op(p, Op::NEG));
}

TEST_CASE("decanonicalize: sqrt(x) roundtrips to SQRT not POWER",
          "[decanonicalize]") {
  Graph g;
  Symbol x = g.variable();
  Program p = Compiler().compile(sqrt(x));
  REQUIRE(has_op(p, Op::SQRT));
  REQUIRE(!has_op(p, Op::POWER));
}

// ── dead_code_elimination ─────────────────────────────────────────────────────

TEST_CASE("dead_code_elimination: unreachable nodes removed", "[dce]") {
  // algebraic_simplification on x+0 replaces the ADD with x, leaving CONSTANT(0) dead
  Graph g;
  Symbol x = g.variable();
  Symbol z = x + 0.0f;
  Compiler c;
  IntermediateProgram ir   = make_ir(g, z);
  IntermediateProgram simp = c.algebraic_simplification(std::move(ir));
  IntermediateProgram dce  = c.dead_code_elimination(std::move(simp));
  REQUIRE(!has_op(dce, Op::CONSTANT));
  REQUIRE(dce.changed == true);
}

TEST_CASE("dead_code_elimination: values pool is compacted", "[dce]") {
  // After simplification the CONSTANT(0) node is dead; its value should be removed
  Graph g;
  Symbol x = g.variable();
  Symbol z = x + 0.0f;
  Compiler c;
  IntermediateProgram ir   = make_ir(g, z);
  IntermediateProgram simp = c.algebraic_simplification(std::move(ir));
  IntermediateProgram dce  = c.dead_code_elimination(std::move(simp));
  for (const Eigen::MatrixXf &v : dce.values)
    REQUIRE(v(0, 0) != Catch::Approx(0.0f));
}

TEST_CASE("dead_code_elimination: all-live graph is unchanged", "[dce]") {
  Graph g;
  Symbol x = g.variable();
  Symbol y = exp(x);
  Compiler c;
  IntermediateProgram ir  = make_ir(g, y);
  IntermediateProgram dce = c.dead_code_elimination(std::move(ir));
  REQUIRE(dce.changed == false);
  REQUIRE(dce.nodes.size() == 2);
}

// ── full compile pipeline ─────────────────────────────────────────────────────

TEST_CASE("compile: 2 + 3 folds to single CONSTANT(5)", "[compile]") {
  Graph g;
  Program p = Compiler().compile(g.constant(2.0f) + g.constant(3.0f));
  REQUIRE(sole_constant(p) == Catch::Approx(5.0f));
}

TEST_CASE("compile: 4 * 3 folds to single CONSTANT(12)", "[compile]") {
  Graph g;
  Program p = Compiler().compile(g.constant(4.0f) * g.constant(3.0f));
  REQUIRE(sole_constant(p) == Catch::Approx(12.0f));
}

TEST_CASE("compile: 2^10 folds to single CONSTANT(1024)", "[compile]") {
  Graph g;
  Program p = Compiler().compile(pow(g.constant(2.0f), g.constant(10.0f)));
  REQUIRE(sole_constant(p) == Catch::Approx(1024.0f));
}

TEST_CASE("compile: skip_optimization preserves full graph", "[compile]") {
  Graph g;
  Symbol x = g.variable();
  Symbol z = x + 0.0f;
  Program opt   = Compiler().compile(z);
  Program noopt = Compiler().compile(z, 10, /*skip_optimization=*/true);
  REQUIRE(opt.nodes.size() < noopt.nodes.size());
}

TEST_CASE("compile: a plain variable compiles to a single VARIABLE node", "[compile]") {
  Graph g;
  Symbol x = g.variable();
  Program p = Compiler().compile(x);
  REQUIRE(p.nodes.size() == 1);
  REQUIRE(p.nodes[0].op == Op::VARIABLE);
}

TEST_CASE("compile: requires_grad propagated into Program nodes", "[compile]") {
  Graph g;
  Symbol x = g.variable();
  Symbol c = g.constant(2.0f);
  Program p = Compiler().compile(x + c);
  bool any_grad = false;
  for (const ProgramNode &n : p.nodes)
    if (n.requires_grad) any_grad = true;
  REQUIRE(any_grad);
}
