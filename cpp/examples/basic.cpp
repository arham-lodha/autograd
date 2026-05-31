#include "autograd/backward.hpp"
#include "autograd/compiler.hpp"
#include "autograd/executor.hpp"
#include "autograd/graph.hpp"
#include "autograd/symbol.hpp"
#include "autograd/tensor.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <span>

using namespace autograd;

// ── infrastructure ────────────────────────────────────────────────────────────

static int failures = 0;

static bool near(float a, float b, float tol = 1e-3f) {
  return std::abs(a - b) < tol;
}

#define CHECK(cond, msg)                          \
  do {                                            \
    if (!(cond)) {                                \
      std::printf("  FAIL  %s\n", (msg));         \
      failures++;                                 \
    } else {                                      \
      std::printf("  pass  %s\n", (msg));         \
    }                                             \
  } while (0)

static Tensor run1(const Program &prog, const Tensor &feed) {
  Executor ex;
  return ex.forward(prog, std::span<const Tensor>(&feed, 1))[0];
}

static Tensor run0(const Program &prog) {
  Executor ex;
  return ex.forward(prog, std::span<const Tensor>{})[0];
}

// ── 1. Tensor SBO ─────────────────────────────────────────────────────────────

static void test_tensor_sbo() {
  std::printf("\n[1] Tensor SBO\n");

  auto is_inline = [](const Tensor &t) {
    const void *obj  = &t;
    const void *data = t.data();
    return data >= obj &&
           data < static_cast<const char *>(obj) + sizeof(Tensor);
  };

  Tensor scalar(3.14f);
  CHECK(is_inline(scalar),         "scalar stored inline");
  CHECK(scalar.is_scalar(),        "scalar.is_scalar()");
  CHECK(near(scalar(0,0), 3.14f),  "scalar value correct");

  Tensor edge(4, 6);   // 24 floats — exactly fills kInline
  edge.map().fill(7.0f);
  CHECK(is_inline(edge),           "4×6 tensor inline (kInline=24)");

  Tensor large(5, 5);  // 25 floats — must heap-allocate
  CHECK(!is_inline(large),         "5×5 tensor heap-allocated");

  Tensor copy = edge;
  copy.map().fill(99.0f);
  CHECK(near(copy(0,0), 99.0f),    "copy has new value");
  CHECK(near(edge(0,0),  7.0f),    "original unchanged after copy");

  Tensor moved = std::move(copy);
  CHECK(near(moved(0,0), 99.0f),   "moved tensor retains value");

  uint32_t sh[] = {2, 3, 4};
  Tensor nd(std::span<const uint32_t>(sh, 3));
  CHECK(nd.ndim()       == 3,  "ndim == 3");
  CHECK(nd.batch_size() == 2,  "batch_size == 2");
  CHECK(nd.rows()       == 3,  "rows == 3");
  CHECK(nd.cols()       == 4,  "cols == 4");
  CHECK(nd.size()       == 24, "size == 24");
}

// ── 2. Scalar forward pass ────────────────────────────────────────────────────

static void test_scalar_forward() {
  std::printf("\n[2] Scalar forward: f(x) = exp(x*2) + 1\n");

  Graph g;
  Symbol x = g.variable();
  Symbol y = exp(x * 2.0f) + 1.0f;
  Program prog = Compiler::compile(g, {x}, {y});

  CHECK(near(run1(prog, Tensor( 0.0f))(0,0), std::exp(0.0f) + 1.0f), "f(0)  = 2");
  CHECK(near(run1(prog, Tensor( 1.0f))(0,0), std::exp(2.0f) + 1.0f), "f(1)  = exp(2)+1");
  CHECK(near(run1(prog, Tensor(-1.0f))(0,0), std::exp(-2.0f)+ 1.0f), "f(-1) = exp(-2)+1");
}

// ── 3. Linear layer ───────────────────────────────────────────────────────────

static void test_linear_layer() {
  std::printf("\n[3] Linear layer: y = x @ W + b  (bias broadcast)\n");

  Eigen::MatrixXf W(3, 2);
  W << 1, 0,
       0, 1,
       1, 1;

  Eigen::MatrixXf b(1, 2);
  b << 0.5f, -0.5f;

  Eigen::MatrixXf x_val(4, 3);
  x_val << 1, 0, 0,
           0, 1, 0,
           0, 0, 1,
           1, 1, 1;

  Graph g;
  Symbol x  = g.variable();
  Symbol y  = matmul(x, g.constant(W)) + g.constant(b);
  Program prog = Compiler::compile(g, {x}, {y});

  Tensor out = run1(prog, Tensor(x_val));

  Eigen::MatrixXf expected = x_val * W;
  expected.rowwise() += b.row(0);

  CHECK(out.rows() == 4 && out.cols() == 2, "output shape (4, 2)");

  bool ok = true;
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 2; c++)
      if (!near(out(r, c), expected(r, c))) ok = false;
  CHECK(ok, "output values match x @ W + b");
}

// ── 4. Autobroadcast ─────────────────────────────────────────────────────────

static void test_autobroadcast() {
  std::printf("\n[4] Autobroadcast\n");

  Graph g;
  Symbol x     = g.variable();
  Symbol bias  = g.constant(Eigen::MatrixXf::Ones(1, 4));  // (1,4) → broadcasts to (3,4)
  Symbol scal  = g.constant(3.0f);                          // (1,1) → scalar broadcast

  Program p1 = Compiler::compile(g, {x}, {x + bias});
  Program p2 = Compiler::compile(g, {x}, {x * scal});

  Tensor zeros(Eigen::MatrixXf::Zero(3, 4));

  Tensor r1 = run1(p1, zeros);
  CHECK(r1.rows() == 3 && r1.cols() == 4, "row-bias broadcast shape (3,4)");
  bool ones = true;
  for (uint32_t r = 0; r < 3; r++)
    for (uint32_t c = 0; c < 4; c++)
      if (!near(r1(r, c), 1.0f)) ones = false;
  CHECK(ones, "zeros + ones-row = all-ones");

  Tensor r2 = run1(p2, zeros);
  CHECK(r2.rows() == 3 && r2.cols() == 4, "scalar broadcast shape (3,4)");
  bool zeroes = true;
  for (uint32_t r = 0; r < 3; r++)
    for (uint32_t c = 0; c < 4; c++)
      if (!near(r2(r, c), 0.0f)) zeroes = false;
  CHECK(zeroes, "zeros * scalar-3 = zeros");
}

// ── 5. Reductions ─────────────────────────────────────────────────────────────

static void test_reductions() {
  std::printf("\n[5] Reductions\n");

  Tensor xt(Eigen::MatrixXf::Constant(3, 4, 2.0f));  // 3×4 of 2s

  Graph g;
  Symbol x = g.variable();
  auto p = [&](Symbol out) { return Compiler::compile(g, {x}, {out}); };

  CHECK(near(run1(p(sum(x)),       xt)(0,0), 24.0f), "sum_all = 24");
  CHECK(near(run1(p(mean(x)),      xt)(0,0),  2.0f), "mean_all = 2");

  Tensor sc = run1(p(sum(x, 0)), xt);
  CHECK(sc.rows() == 1 && sc.cols() == 4, "sum(axis=0) shape (1,4)");
  bool ok = true;
  for (uint32_t c = 0; c < 4; c++)
    if (!near(sc(0, c), 6.0f)) ok = false;
  CHECK(ok, "sum(axis=0) = 6 per column");

  Tensor mr = run1(p(mean(x, 1)), xt);
  CHECK(mr.rows() == 3 && mr.cols() == 1, "mean(axis=1) shape (3,1)");
  ok = true;
  for (uint32_t r = 0; r < 3; r++)
    if (!near(mr(r, 0), 2.0f)) ok = false;
  CHECK(ok, "mean(axis=1) = 2 per row");
}

// ── 6. Gradient check ─────────────────────────────────────────────────────────

static void test_gradient_check() {
  std::printf("\n[6] Gradient check: f(x) = x^2 + exp(x),  f'(x) = 2x + exp(x)\n");

  Graph g;
  Symbol x = g.variable();
  Symbol y = pow(x, 2.0f) + exp(x);

  Program fwd = Compiler::compile(g, {x}, {y});

  auto grads = backwards(y);
  Program bwd = Compiler::compile(g, {x}, {*grads[x.node_index]});

  auto f  = [&](float v) { return run1(fwd, Tensor(v))(0, 0); };
  auto df = [&](float v) { return run1(bwd, Tensor(v))(0, 0); };

  for (float xv : {-1.5f, 0.0f, 1.0f, 2.0f}) {
    float eps        = 1e-4f;
    float numerical  = (f(xv + eps) - f(xv - eps)) / (2.0f * eps);
    float analytical = df(xv);
    float exact      = 2.0f * xv + std::exp(xv);

    char msg[96];
    std::snprintf(msg, sizeof(msg),
                  "x=% .2f  analytical=%.4f  numerical=%.4f  exact=%.4f",
                  xv, analytical, numerical, exact);
    CHECK(near(analytical, numerical, 1e-2f) && near(analytical, exact, 1e-2f), msg);
  }
}

// ── 7. MSE loss gradient ──────────────────────────────────────────────────────

static void test_mse_gradient() {
  std::printf("\n[7] MSE gradient: loss = (x*w - t)^2,  d/dw = 2*(x*w-t)*x\n");

  const float xv = 2.0f, tv = 1.0f;

  Graph g;
  Symbol w    = g.variable();
  Symbol pred = g.constant(xv) * w;
  Symbol diff = pred - g.constant(tv);
  Symbol loss = pow(diff, 2.0f);

  Program fwd = Compiler::compile(g, {w}, {loss});

  auto grads = backwards(loss);
  Program bwd = Compiler::compile(g, {w}, {*grads[w.node_index]});

  auto check_at = [&](float wv) {
    float loss_out = run1(fwd, Tensor(wv))(0, 0);
    float grad_out = run1(bwd, Tensor(wv))(0, 0);
    float exp_loss = std::pow(xv * wv - tv, 2.0f);
    float exp_grad = 2.0f * (xv * wv - tv) * xv;

    char msg[64];
    std::snprintf(msg, sizeof(msg), "w=%.1f  loss=%.3f(exp %.3f)  grad=%.3f(exp %.3f)",
                  wv, loss_out, exp_loss, grad_out, exp_grad);
    CHECK(near(loss_out, exp_loss) && near(grad_out, exp_grad), msg);
  };

  check_at(0.0f);   // grad = -4
  check_at(1.0f);   // grad = +4
  check_at(0.5f);   // grad = 0 (minimum)
}

// ── 8. Compiler optimisations ─────────────────────────────────────────────────

static void test_compiler_opts() {
  std::printf("\n[8] Compiler optimisations\n");

  {
    Graph g;
    Program p = Compiler::compile(g, {}, {g.constant(2.0f) + g.constant(3.0f)});
    CHECK(p.nodes.size() == 1 && p.nodes[0].operation == Op::CONSTANT,
          "2+3 constant-folds to single CONSTANT node");
    CHECK(near(run0(p)(0,0), 5.0f), "folded value = 5");
  }
  {
    Graph g;
    Symbol x = g.variable();
    Program p = Compiler::compile(g, {x}, {x + 0.0f});
    CHECK(p.nodes.size() == 1 && p.nodes[0].operation == Op::VARIABLE,
          "x + 0 simplifies to x");
  }
  {
    Graph g;
    Symbol x = g.variable();
    Program p = Compiler::compile(g, {x}, {log(exp(x))});
    CHECK(p.nodes.size() == 1 && p.nodes[0].operation == Op::VARIABLE,
          "log(exp(x)) simplifies to x");
  }
  {
    Graph g;
    Symbol x = g.variable();
    Program p = Compiler::compile(g, {x}, {pow(x, 0.0f)});
    CHECK(p.nodes.size() == 1 && p.nodes[0].operation == Op::CONSTANT,
          "x^0 simplifies to constant 1");
  }
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
  std::printf("=== autograd end-to-end ===\n");

  test_tensor_sbo();
  test_scalar_forward();
  test_linear_layer();
  test_autobroadcast();
  test_reductions();
  test_gradient_check();
  test_mse_gradient();
  test_compiler_opts();

  std::printf("\n");
  if (failures == 0) {
    std::printf("All tests passed.\n");
    return 0;
  }
  std::printf("%d test(s) FAILED.\n", failures);
  return 1;
}
