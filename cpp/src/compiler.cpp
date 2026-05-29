#include "autograd/compiler.hpp"
#include "autograd/symbol.hpp"
#include <cstdint>
#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace autograd {

// ── Static helpers
// ────────────────────────────────────────────────────────────

static std::string shape_str(const Eigen::MatrixXf &m) {
  return "(" + std::to_string(m.rows()) + "×" + std::to_string(m.cols()) + ")";
}

// Broadcast src to (rows, cols) using NumPy rules: each dimension of src must
// either match the target or be 1.  Expansion is done with replicate().
static Eigen::MatrixXf broadcast_to(const Eigen::MatrixXf &src,
                                    Eigen::Index rows, Eigen::Index cols) {
  bool row_ok = src.rows() == rows || src.rows() == 1;
  bool col_ok = src.cols() == cols || src.cols() == 1;
  if (!row_ok || !col_ok)
    throw std::runtime_error("broadcast_to: cannot broadcast " +
                             shape_str(src) + " to (" + std::to_string(rows) +
                             "×" + std::to_string(cols) + ")");

  Eigen::Index row_reps = (src.rows() == 1) ? rows : 1;
  Eigen::Index col_reps = (src.cols() == 1) ? cols : 1;

  if (row_reps == 1 && col_reps == 1)
    return src;

  return src.replicate(row_reps, col_reps).eval();
}

// Broadcast a and b to a common shape (elementwise-compatible shapes only).
static std::pair<Eigen::MatrixXf, Eigen::MatrixXf>
align(const Eigen::MatrixXf &a, const Eigen::MatrixXf &b) {
  bool row_ok = a.rows() == b.rows() || a.rows() == 1 || b.rows() == 1;
  bool col_ok = a.cols() == b.cols() || a.cols() == 1 || b.cols() == 1;
  if (!row_ok || !col_ok)
    throw std::runtime_error("align: shapes " + shape_str(a) + " and " +
                             shape_str(b) + " are not broadcastable");

  Eigen::Index rows = std::max(a.rows(), b.rows());
  Eigen::Index cols = std::max(a.cols(), b.cols());

  return {broadcast_to(a, rows, cols), broadcast_to(b, rows, cols)};
}

// Returns true for ops that eval_op knows how to evaluate.
static bool is_foldable(Op op) {
  switch (op) {
  case Op::ADD:
  case Op::MULTIPLY:
  case Op::POWER:
  case Op::MATMUL:
  case Op::LOG:
  case Op::EXP:
  case Op::NEG:
  case Op::SQRT:
  case Op::ABS:
  case Op::SIGN:
  case Op::RELU:
  case Op::TRANSPOSE:
  case Op::IDENTITY:
    return true;
  default:
    return false;
  }
}

// Evaluate op over concrete matrix inputs.  Throws for incompatible shapes.
// Only call this after checking is_foldable().
static Eigen::MatrixXf eval_op(Op op, const std::vector<Eigen::MatrixXf> &ins) {
  switch (op) {
  case Op::ADD: {
    Eigen::MatrixXf acc = ins[0];
    for (size_t k = 1; k < ins.size(); k++) {
      std::pair<Eigen::MatrixXf, Eigen::MatrixXf> p = align(acc, ins[k]);
      acc = (p.first.array() + p.second.array()).matrix();
    }
    return acc;
  }
  case Op::MULTIPLY: {
    Eigen::MatrixXf acc = ins[0];
    for (size_t k = 1; k < ins.size(); k++) {
      std::pair<Eigen::MatrixXf, Eigen::MatrixXf> p = align(acc, ins[k]);
      acc = (p.first.array() * p.second.array()).matrix();
    }
    return acc;
  }
  case Op::POWER: {
    std::pair<Eigen::MatrixXf, Eigen::MatrixXf> p = align(ins[0], ins[1]);
    return p.first.array().pow(p.second.array()).matrix();
  }
  case Op::MATMUL:
    if (ins[0].cols() != ins[1].rows())
      throw std::runtime_error("matmul: inner dimensions must match, got " +
                               shape_str(ins[0]) + " and " + shape_str(ins[1]));
    return (ins[0] * ins[1]).eval();
  case Op::LOG:
    return ins[0].array().log().matrix();
  case Op::EXP:
    return ins[0].array().exp().matrix();
  case Op::NEG:
    return (-ins[0]).eval();
  case Op::SQRT:
    return ins[0].array().sqrt().matrix();
  case Op::ABS:
    return ins[0].array().abs().matrix();
  case Op::SIGN:
    return ins[0].array().sign().matrix();
  case Op::RELU:
    return ins[0].cwiseMax(0.0f);
  case Op::TRANSPOSE:
    return ins[0].transpose().eval();
  case Op::IDENTITY:
    return ins[0];
  default:
    throw std::runtime_error("eval_op: op is not foldable");
  }
}

// Append an arbitrary constant matrix to the IR being built; return its index.
static uint32_t push_const(IntermediateProgram &prog,
                           std::vector<IntermediateProgramNode> &new_nodes,
                           Eigen::MatrixXf val) {
  uint32_t val_idx = prog.values.size();
  prog.values.push_back(std::move(val));
  uint32_t idx = new_nodes.size();
  IntermediateProgramNode cn;
  cn.op = Op::CONSTANT;
  cn.value_index = val_idx;
  new_nodes.push_back(std::move(cn));
  return idx;
}

// Shorthand: push a 1×1 scalar constant.
static uint32_t push_scalar(IntermediateProgram &prog,
                            std::vector<IntermediateProgramNode> &new_nodes,
                            float v) {
  return push_const(prog, new_nodes, Eigen::MatrixXf::Constant(1, 1, v));
}

// True if the node at new-space idx is a CONSTANT with every element equal to
// v.
static bool is_const_val(const std::vector<IntermediateProgramNode> &nodes,
                         const std::vector<Eigen::MatrixXf> &values,
                         uint32_t idx, float v) {
  const IntermediateProgramNode &n = nodes[idx];
  if (n.op != Op::CONSTANT)
    return false;
  return (values[n.value_index].array() == v).all();
}

// Replace node i with an already-emitted node at replacement_new_idx.
// If node.retain, emits it as IDENTITY instead of truly skipping it.
// Caller must `continue` after this call either way.
static void try_replace(uint32_t i, IntermediateProgramNode &node,
                        uint32_t replacement_new_idx,
                        std::vector<IntermediateProgramNode> &new_nodes,
                        std::unordered_map<uint32_t, uint32_t> &remap) {
  if (node.retain) {
    node.op = Op::IDENTITY;
    node.inputs = {replacement_new_idx};
    node.value_index = UINT32_MAX;
    remap[i] = new_nodes.size();
    new_nodes.push_back(std::move(node));
  } else {
    remap[i] = replacement_new_idx;
  }
}

// ── Topological sort ─────────────────────────────────────────────────────────

std::vector<uint32_t> Compiler::topological_sort(const Symbol &output) {
  std::vector<uint32_t> sorted;
  std::unordered_set<uint32_t> visited;
  std::stack<uint32_t> stk;
  const Graph *graph = output.graph;
  stk.push(output.node_index);

  while (!stk.empty()) {
    uint32_t node_id = stk.top();

    if (visited.count(node_id)) {
      stk.pop();
      continue;
    }

    const Node &node = graph->nodes[node_id];
    bool all_visited = true;

    for (uint32_t i = 0; i < node.input_count; i++) {
      uint32_t input_id = graph->inputs[node.input_offset + i];
      if (!visited.count(input_id)) {
        stk.push(input_id);
        all_visited = false;
      }
    }

    if (all_visited) {
      visited.insert(node_id);
      sorted.push_back(node_id);
      stk.pop();
    }
  }

  return sorted;
}

// ── Graph → IR ───────────────────────────────────────────────────────────────

IntermediateProgram Compiler::make_ir(const Graph &graph,
                                      const std::vector<uint32_t> &topo) {
  IntermediateProgram prog;
  prog.values = graph.values;
  std::unordered_map<uint32_t, uint32_t> graph_to_ir;

  for (uint32_t graph_idx : topo) {
    const Node &n = graph.nodes[graph_idx];

    IntermediateProgramNode ir;
    ir.op = n.operation;
    ir.value_index = n.value_index;
    ir.axis = n.axis;
    ir.axis2 = n.axis2;
    ir.keepdims = n.keepdims;
    ir.retain = n.retain;
    ir.requires_grad = n.requires_grad;

    for (uint32_t i = 0; i < n.input_count; i++)
      ir.inputs.push_back(graph_to_ir.at(graph.inputs[n.input_offset + i]));

    graph_to_ir[graph_idx] = prog.nodes.size();
    prog.nodes.push_back(std::move(ir));
  }

  prog.output_idx = prog.nodes.size() - 1;
  return prog;
}

// ── IR → Program (freeze to CSR) ─────────────────────────────────────────────

Program Compiler::freeze(const IntermediateProgram &prog) {
  Program p;
  p.values = prog.values;

  for (const IntermediateProgramNode &ir : prog.nodes) {
    ProgramNode pn;
    pn.op = ir.op;
    pn.value_index = ir.value_index;
    pn.axis = ir.axis;
    pn.axis2 = ir.axis2;
    pn.keepdims = ir.keepdims;
    pn.requires_grad = ir.requires_grad;
    pn.inputs_offset = p.inputs.size();
    pn.inputs_count = ir.inputs.size();

    for (uint32_t inp : ir.inputs)
      p.inputs.push_back(inp);

    p.nodes.push_back(pn);
  }

  return p;
}

// ── compile
// ───────────────────────────────────────────────────────────────────

Program Compiler::compile(const Symbol &output, bool backwards,
                          uint optimization_passes, bool skip_optimization) {
  std::vector<uint32_t> topo = topological_sort(output);
  IntermediateProgram prog = make_ir(*output.graph, topo);

  if (!skip_optimization) {
    for (uint i = 0; i < optimization_passes; i++) {
      prog = optimize(std::move(prog));
      if (!prog.changed)
        break;
    }
    prog = decanonicalize(std::move(prog));
  }

  return freeze(prog);
}

// ── optimize
// ──────────────────────────────────────────────────────────────────

IntermediateProgram Compiler::optimize(IntermediateProgram prog) {
  bool changed = false;

  prog = canonicalize(std::move(prog));
  changed |= prog.changed;
  prog = addition_folding(std::move(prog));
  changed |= prog.changed;
  prog = multiplication_folding(std::move(prog));
  changed |= prog.changed;
  prog = constant_fold(std::move(prog));
  changed |= prog.changed;
  prog = algebraic_simplification(std::move(prog));
  changed |= prog.changed;
  prog = dead_code_elimination(std::move(prog));
  changed |= prog.changed;

  prog.changed = changed;
  return prog;
}

// ── canonicalize ─────────────────────────────────────────────────────────────
//
// NEG x        →  MUL x, -1
// SUB a, b     →  ADD a, (MUL b, -1)
// SQRT x       →  POW x, 0.5

IntermediateProgram Compiler::canonicalize(IntermediateProgram prog) {
  std::vector<IntermediateProgramNode> new_nodes;
  std::unordered_map<uint32_t, uint32_t> remap;
  bool changed = false;
  uint32_t old_output_idx = prog.output_idx;

  for (uint32_t i = 0; i < prog.nodes.size(); i++) {
    IntermediateProgramNode node = prog.nodes[i];

    for (uint32_t &inp : node.inputs)
      inp = remap.at(inp);

    switch (node.op) {
    case Op::NEG: {
      uint32_t neg_one = push_scalar(prog, new_nodes, -1.0f);
      node.op = Op::MULTIPLY;
      node.inputs.push_back(neg_one);
      changed = true;
      break;
    }
    case Op::SUB: {
      uint32_t b = node.inputs[1];
      uint32_t neg_one = push_scalar(prog, new_nodes, -1.0f);
      IntermediateProgramNode mul;
      mul.op = Op::MULTIPLY;
      mul.inputs = {b, neg_one};
      mul.requires_grad = new_nodes[b].requires_grad;
      uint32_t mul_idx = new_nodes.size();
      new_nodes.push_back(std::move(mul));
      node.op = Op::ADD;
      node.inputs[1] = mul_idx;
      changed = true;
      break;
    }
    case Op::SQRT: {
      uint32_t half = push_scalar(prog, new_nodes, 0.5f);
      node.op = Op::POWER;
      node.inputs.push_back(half);
      changed = true;
      break;
    }
    default:
      break;
    }

    remap[i] = new_nodes.size();
    new_nodes.push_back(std::move(node));
  }

  prog.nodes = std::move(new_nodes);
  prog.output_idx = remap.at(old_output_idx);
  prog.changed = changed;
  return prog;
}

// ── decanonicalize
// ────────────────────────────────────────────────────────────

IntermediateProgram Compiler::decanonicalize(IntermediateProgram prog) {
  std::vector<IntermediateProgramNode> new_nodes;
  std::unordered_map<uint32_t, uint32_t> remap;
  bool changed = false;
  uint32_t old_output_idx = prog.output_idx;

  for (uint32_t i = 0; i < prog.nodes.size(); i++) {
    IntermediateProgramNode node = prog.nodes[i];

    for (uint32_t &inp : node.inputs)
      inp = remap.at(inp);

    // ── MULTIPLY: absorb -1 constants ────────────────────────────────────────
    if (node.op == Op::MULTIPLY) {
      std::vector<uint32_t> neg_ones, operands;
      for (uint32_t inp : node.inputs)
        (is_const_val(new_nodes, prog.values, inp, -1.0f) ? neg_ones : operands)
            .push_back(inp);

      if (!neg_ones.empty()) {
        changed = true;
        bool odd = (neg_ones.size() % 2) == 1;

        if (odd) {
          if (operands.empty()) {
            // (-1)*(-1)*... (odd) → constant -1
            uint32_t val_idx = prog.values.size();
            prog.values.push_back(Eigen::MatrixXf::Constant(1, 1, -1.0f));
            node.op = Op::CONSTANT;
            node.value_index = val_idx;
            node.inputs.clear();
            node.requires_grad = false;
          } else if (operands.size() == 1) {
            // x * (-1) → NEG x
            node.op = Op::NEG;
            node.inputs = operands;
          } else {
            // multiple operands, odd -1s → MULTIPLY(operands..., -1)
            uint32_t neg_one = push_scalar(prog, new_nodes, -1.0f);
            operands.push_back(neg_one);
            node.inputs = std::move(operands);
          }
        } else {
          if (operands.empty()) {
            // (-1)*(-1)*... (even) → constant 1
            uint32_t val_idx = prog.values.size();
            prog.values.push_back(Eigen::MatrixXf::Constant(1, 1, 1.0f));
            node.op = Op::CONSTANT;
            node.value_index = val_idx;
            node.inputs.clear();
            node.requires_grad = false;
          } else if (operands.size() == 1) {
            // x * (-1)*(-1) → IDENTITY x
            node.op = Op::IDENTITY;
            node.inputs = operands;
          } else {
            // strip the even -1s, keep the rest
            node.inputs = std::move(operands);
          }
        }
      }
    }

    // ── POWER(x, 0.5) → SQRT(x) ──────────────────────────────────────────────
    if (node.op == Op::POWER &&
        is_const_val(new_nodes, prog.values, node.inputs[1], 0.5f)) {
      node.op = Op::SQRT;
      node.inputs = {node.inputs[0]};
      changed = true;
    }

    remap[i] = new_nodes.size();
    new_nodes.push_back(std::move(node));
  }

  prog.nodes = std::move(new_nodes);
  prog.output_idx = remap.at(old_output_idx);
  prog.changed = changed;
  return prog;
}

// ── constant_fold
// ─────────────────────────────────────────────────────────────

IntermediateProgram Compiler::constant_fold(IntermediateProgram prog) {
  std::vector<IntermediateProgramNode> new_nodes;
  std::unordered_map<uint32_t, uint32_t> remap;
  bool changed = false;
  uint32_t old_output_idx = prog.output_idx;

  for (uint32_t i = 0; i < prog.nodes.size(); i++) {
    IntermediateProgramNode node = prog.nodes[i];
    for (uint32_t &inp : node.inputs)
      inp = remap.at(inp);

    bool is_nary = node.op == Op::ADD || node.op == Op::MULTIPLY;

    if (!node.inputs.empty() && !is_nary && is_foldable(node.op)) {
      // Non n-ary: if every input is a constant, evaluate and replace.
      bool all_const = true;
      for (uint32_t inp : node.inputs)
        if (new_nodes[inp].op != Op::CONSTANT) {
          all_const = false;
          break;
        }

      if (all_const) {
        std::vector<Eigen::MatrixXf> ins;
        for (uint32_t inp : node.inputs)
          ins.push_back(prog.values[new_nodes[inp].value_index]);

        Eigen::MatrixXf res = eval_op(node.op, ins);
        uint32_t val_idx = prog.values.size();
        prog.values.push_back(std::move(res));
        node.op = Op::CONSTANT;
        node.value_index = val_idx;
        node.inputs.clear();
        node.requires_grad = false;
        changed = true;
      }
    } else if (is_nary) {
      // n-ary ADD/MULTIPLY: fold any group of 2+ constant inputs into one.
      std::vector<uint32_t> consts, non_consts;
      for (uint32_t inp : node.inputs)
        (new_nodes[inp].op == Op::CONSTANT ? consts : non_consts)
            .push_back(inp);

      if (consts.size() >= 2) {
        std::vector<Eigen::MatrixXf> ins;
        for (uint32_t inp : consts)
          ins.push_back(prog.values[new_nodes[inp].value_index]);

        Eigen::MatrixXf res = eval_op(node.op, ins);
        if (non_consts.empty()) {
          uint32_t val_idx = prog.values.size();
          prog.values.push_back(std::move(res));
          node.op = Op::CONSTANT;
          node.value_index = val_idx;
          node.inputs.clear();
          node.requires_grad = false;
        } else {
          uint32_t folded_idx = push_const(prog, new_nodes, std::move(res));
          non_consts.push_back(folded_idx);
          node.inputs = std::move(non_consts);
        }
        changed = true;
      }
    }

    remap[i] = new_nodes.size();
    new_nodes.push_back(std::move(node));
  }

  prog.nodes = std::move(new_nodes);
  prog.output_idx = remap.at(old_output_idx);
  prog.changed = changed;
  return prog;
}

// ── addition_folding
// ──────────────────────────────────────────────────────────

IntermediateProgram Compiler::addition_folding(IntermediateProgram prog) {
  bool changed = false;

  for (IntermediateProgramNode &node : prog.nodes) {
    if (node.op != Op::ADD)
      continue;

    bool has_add_input = false;
    for (uint32_t inp : node.inputs) {
      if (prog.nodes[inp].op == Op::ADD && !prog.nodes[inp].retain) {
        has_add_input = true;
        break;
      }
    }

    if (!has_add_input)
      continue;

    std::vector<uint32_t> flat;
    for (uint32_t inp : node.inputs) {
      const IntermediateProgramNode &parent = prog.nodes[inp];
      if (parent.op == Op::ADD && !parent.retain)
        flat.insert(flat.end(), parent.inputs.begin(), parent.inputs.end());
      else
        flat.push_back(inp);
    }
    node.inputs = std::move(flat);
    changed = true;
  }

  prog.changed = changed;
  return prog;
}

// ── multiplication_folding
// ────────────────────────────────────────────────────

IntermediateProgram Compiler::multiplication_folding(IntermediateProgram prog) {
  bool changed = false;

  for (IntermediateProgramNode &node : prog.nodes) {
    if (node.op != Op::MULTIPLY)
      continue;

    bool has_mul_input = false;
    for (uint32_t inp : node.inputs) {
      if (prog.nodes[inp].op == Op::MULTIPLY && !prog.nodes[inp].retain) {
        has_mul_input = true;
        break;
      }
    }

    if (!has_mul_input)
      continue;

    std::vector<uint32_t> flat;
    for (uint32_t inp : node.inputs) {
      const IntermediateProgramNode &parent = prog.nodes[inp];
      if (parent.op == Op::MULTIPLY && !parent.retain)
        flat.insert(flat.end(), parent.inputs.begin(), parent.inputs.end());
      else
        flat.push_back(inp);
    }
    node.inputs = std::move(flat);
    changed = true;
  }

  prog.changed = changed;
  return prog;
}

// ── Pass stubs
// ────────────────────────────────────────────────────────────────

IntermediateProgram
Compiler::algebraic_simplification(IntermediateProgram prog) {
  std::vector<IntermediateProgramNode> new_nodes;
  std::unordered_map<uint32_t, uint32_t> remap;
  bool changed = false;
  uint32_t old_output_idx = prog.output_idx;

  for (uint32_t i = 0; i < prog.nodes.size(); i++) {
    IntermediateProgramNode node = prog.nodes[i];

    for (uint32_t &inp : node.inputs)
      inp = remap.at(inp);

    switch (node.op) {

    case Op::ADD: {
      // ── Remove zero inputs ────────────────────────────────────────────────
      std::vector<uint32_t> zeros, non_zeros;
      for (uint32_t inp : node.inputs)
        (is_const_val(new_nodes, prog.values, inp, 0.0f) ? zeros : non_zeros)
            .push_back(inp);

      if (!zeros.empty()) {
        changed = true;
        if (non_zeros.empty()) {
          uint32_t z = push_scalar(prog, new_nodes, 0.0f);
          try_replace(i, node, z, new_nodes, remap);
          continue;
        }
        if (non_zeros.size() == 1) {
          try_replace(i, node, non_zeros[0], new_nodes, remap);
          continue;
        }
        node.inputs = non_zeros;
      }

      // ── Deduplicate: x+x+x → 3*x ─────────────────────────────────────────
      std::unordered_map<uint32_t, uint32_t> counts;
      for (uint32_t inp : node.inputs)
        counts[inp]++;

      bool has_dups = false;
      for (const std::pair<const uint32_t, uint32_t> &kv : counts)
        if (kv.second > 1) {
          has_dups = true;
          break;
        }

      if (has_dups) {
        std::vector<uint32_t> new_inputs;
        std::unordered_set<uint32_t> seen;
        for (uint32_t inp : node.inputs) {
          if (seen.count(inp))
            continue;
          seen.insert(inp);
          uint32_t cnt = counts[inp];
          if (cnt > 1) {
            uint32_t c = push_scalar(prog, new_nodes, float(cnt));
            IntermediateProgramNode mul;
            mul.op = Op::MULTIPLY;
            mul.inputs = {c, inp};
            mul.requires_grad = new_nodes[inp].requires_grad;
            uint32_t mul_idx = new_nodes.size();
            new_nodes.push_back(std::move(mul));
            new_inputs.push_back(mul_idx);
          } else {
            new_inputs.push_back(inp);
          }
        }
        node.inputs = std::move(new_inputs);
        changed = true;
      }
      break;
    }

    case Op::MULTIPLY: {
      // ── Any input is zero → replace with zero ─────────────────────────────
      bool has_zero = false;
      for (uint32_t inp : node.inputs)
        if (is_const_val(new_nodes, prog.values, inp, 0.0f)) {
          has_zero = true;
          break;
        }

      if (has_zero) {
        uint32_t z = push_scalar(prog, new_nodes, 0.0f);
        changed = true;
        try_replace(i, node, z, new_nodes, remap);
        continue;
      }

      // ── Remove ones ───────────────────────────────────────────────────────
      std::vector<uint32_t> ones, non_ones;
      for (uint32_t inp : node.inputs)
        (is_const_val(new_nodes, prog.values, inp, 1.0f) ? ones : non_ones)
            .push_back(inp);

      if (!ones.empty()) {
        changed = true;
        if (non_ones.empty()) {
          uint32_t one = push_scalar(prog, new_nodes, 1.0f);
          try_replace(i, node, one, new_nodes, remap);
          continue;
        }
        if (non_ones.size() == 1) {
          try_replace(i, node, non_ones[0], new_nodes, remap);
          continue;
        }
        node.inputs = non_ones;
      }
      break;
    }

    case Op::POWER: {
      uint32_t base_idx = node.inputs[0];
      uint32_t exp_idx = node.inputs[1];

      // x^0 = 1
      if (is_const_val(new_nodes, prog.values, exp_idx, 0.0f)) {
        uint32_t one = push_scalar(prog, new_nodes, 1.0f);
        changed = true;
        try_replace(i, node, one, new_nodes, remap);
        continue;
      }
      // x^1 = x
      if (is_const_val(new_nodes, prog.values, exp_idx, 1.0f)) {
        changed = true;
        try_replace(i, node, base_idx, new_nodes, remap);
        continue;
      }
      // 1^x = 1
      if (is_const_val(new_nodes, prog.values, base_idx, 1.0f)) {
        uint32_t one = push_scalar(prog, new_nodes, 1.0f);
        changed = true;
        try_replace(i, node, one, new_nodes, remap);
        continue;
      }
      // (x^a)^b → x^(a*b)
      if (new_nodes[base_idx].op == Op::POWER) {
        uint32_t inner_base = new_nodes[base_idx].inputs[0];
        uint32_t inner_exp = new_nodes[base_idx].inputs[1];
        IntermediateProgramNode mul;
        mul.op = Op::MULTIPLY;
        mul.inputs = {inner_exp, exp_idx};
        uint32_t new_exp_idx = new_nodes.size();
        new_nodes.push_back(std::move(mul));
        node.inputs[0] = inner_base;
        node.inputs[1] = new_exp_idx;
        base_idx = inner_base;
        exp_idx = new_exp_idx;
        changed = true;
      }
      // x^n for integer n in [2,4] → n-ary MULTIPLY(x, x, ..., x)
      if (new_nodes[exp_idx].op == Op::CONSTANT) {
        const Eigen::MatrixXf &e = prog.values[new_nodes[exp_idx].value_index];
        if (e.size() == 1) {
          int n = int(e(0, 0));
          if (float(n) == e(0, 0) && n >= 2 && n <= 4) {
            node.op = Op::MULTIPLY;
            node.inputs.assign(n, base_idx);
            changed = true;
          }
        }
      }
      break;
    }

    case Op::LOG: {
      uint32_t arg = node.inputs[0];
      // log(exp(x)) = x
      if (new_nodes[arg].op == Op::EXP) {
        changed = true;
        try_replace(i, node, new_nodes[arg].inputs[0], new_nodes, remap);
        continue;
      }
      // log(x^a) = a * log(|x|)
      if (new_nodes[arg].op == Op::POWER) {
        uint32_t pow_base = new_nodes[arg].inputs[0];
        uint32_t pow_exp = new_nodes[arg].inputs[1];

        IntermediateProgramNode abs_node;
        abs_node.op = Op::ABS;
        abs_node.inputs = {pow_base};
        abs_node.requires_grad = new_nodes[pow_base].requires_grad;
        uint32_t abs_idx = new_nodes.size();
        new_nodes.push_back(std::move(abs_node));

        IntermediateProgramNode log_inner;
        log_inner.op = Op::LOG;
        log_inner.inputs = {abs_idx};
        log_inner.requires_grad = new_nodes[abs_idx].requires_grad;
        uint32_t log_idx = new_nodes.size();
        new_nodes.push_back(std::move(log_inner));

        node.op = Op::MULTIPLY;
        node.inputs = {pow_exp, log_idx};
        changed = true;
      }
      break;
    }

    case Op::EXP: {
      // exp(log(x)) = x
      if (new_nodes[node.inputs[0]].op == Op::LOG) {
        changed = true;
        try_replace(i, node, new_nodes[node.inputs[0]].inputs[0], new_nodes,
                    remap);
        continue;
      }
      break;
    }

    case Op::TRANSPOSE: {
      // T(T(x)) = x
      if (new_nodes[node.inputs[0]].op == Op::TRANSPOSE) {
        changed = true;
        try_replace(i, node, new_nodes[node.inputs[0]].inputs[0], new_nodes,
                    remap);
        continue;
      }
      break;
    }

    case Op::RELU: {
      // relu(relu(x)) = relu(x)
      uint32_t arg = node.inputs[0];
      if (new_nodes[arg].op == Op::RELU && !new_nodes[arg].retain) {
        changed = true;
        try_replace(i, node, arg, new_nodes, remap);
        continue;
      }
      break;
    }

    default:
      break;
    }

    remap[i] = new_nodes.size();
    new_nodes.push_back(std::move(node));
  }

  prog.nodes = std::move(new_nodes);
  prog.output_idx = remap.at(old_output_idx);
  prog.changed = changed;
  return prog;
}

IntermediateProgram Compiler::dead_code_elimination(IntermediateProgram prog) {
  if (prog.nodes.empty()) {
    prog.changed = false;
    return prog;
  }

  uint32_t old_output_idx = prog.output_idx;
  std::unordered_set<uint32_t> live;
  live.insert(old_output_idx);

  for (int i = int(prog.nodes.size()) - 1; i >= 0; i--) {
    if (!live.count(uint32_t(i)))
      continue;
    for (uint32_t inp : prog.nodes[i].inputs)
      live.insert(inp);
  }

  if (live.size() == prog.nodes.size()) {
    prog.changed = false;
    return prog;
  }

  // Rebuild keeping only live nodes; remap compacts node and value indices.
  std::vector<IntermediateProgramNode> live_nodes;
  std::vector<Eigen::MatrixXf> live_values;
  std::unordered_map<uint32_t, uint32_t> node_remap;
  std::unordered_map<uint32_t, uint32_t> value_remap;

  for (uint32_t i = 0; i < prog.nodes.size(); i++) {
    if (!live.count(i))
      continue;
    IntermediateProgramNode node = prog.nodes[i];
    for (uint32_t &inp : node.inputs)
      inp = node_remap.at(inp);
    if (node.value_index != UINT32_MAX) {
      if (!value_remap.count(node.value_index)) {
        value_remap[node.value_index] = live_values.size();
        live_values.push_back(prog.values[node.value_index]);
      }
      node.value_index = value_remap.at(node.value_index);
    }
    node_remap[i] = live_nodes.size();
    live_nodes.push_back(std::move(node));
  }

  prog.nodes = std::move(live_nodes);
  prog.values = std::move(live_values);
  prog.output_idx = node_remap.at(old_output_idx);
  prog.changed = true;
  return prog;
}

// ── Backwards (stub)
// ──────────────────────────────────────────────────────────

Program Compiler::compile_backwards(const Symbol &) { return Program{}; }

Symbol Compiler::make_constant(Graph &g, float value) {
  return g.constant(value);
}

Symbol Compiler::accumulate(const Symbol &existing, const Symbol &grad) {
  return existing + grad;
}

void Compiler::backwards_node(Graph &, uint32_t,
                              std::unordered_map<uint32_t, Symbol> &) {}

} // namespace autograd
