#include "autograd/backward.hpp"
#include <stack>
#include <stdexcept>
#include <unordered_map>

namespace autograd {

static void backwards_node(Graph &graph, uint32_t index,
                           std::unordered_map<uint32_t, Symbol> &grads) {
  Node &node = graph.nodes[index];

  if (node.operation == Op::CONSTANT || node.operation == Op::PLACEHOLDER ||
      node.operation == Op::VARIABLE)
    return;

  if (!node.requires_grad)
    return;

  if (grads.find(index) == grads.end())
    return;

  Symbol grad = grads[index];

  std::vector<Symbol> prev;
  for (uint32_t i = 0; i < node.input_count; i++) {
    uint32_t inp =
        i < 2 ? node.inputs[i] : graph.inputs[node.input_pool_offset + (i - 2)];
    prev.push_back(Symbol{.node_index = inp, .graph = &graph});
  }

  auto has_axis = [&]() { return node.axes.axis >= 0; };
  auto ax_opt = [&]() -> std::optional<int> {
    return has_axis() ? std::optional<int>(node.axes.axis) : std::nullopt;
  };
  auto needs_grad = [&](const Symbol &s) {
    return graph.nodes[s.node_index].requires_grad;
  };

  switch (node.operation) {
  case Op::ADD:
    for (uint32_t i = 0; i < node.input_count; i++)
      if (needs_grad(prev[i]))
        grads[prev[i].node_index] =
            grads[prev[i].node_index] + unbroadcast(grad, prev[i]);
    break;

  case Op::MULTIPLY:
    for (uint32_t i = 0; i < node.input_count; i++) {
      if (needs_grad(prev[i])) {
        Symbol g = grad;
        for (uint32_t j = 0; j < node.input_count; j++)
          if (i != j)
            g = g * prev[j];
        grads[prev[i].node_index] =
            grads[prev[i].node_index] + unbroadcast(g, prev[i]);
      }
    }
    break;

  case Op::SUB: {
    Symbol a = prev[0], b = prev[1];
    if (needs_grad(a))
      grads[a.node_index] = grads[a.node_index] + unbroadcast(grad, a);
    if (needs_grad(b))
      grads[b.node_index] = grads[b.node_index] + unbroadcast(-grad, b);
    break;
  }

  case Op::NEG:
    if (needs_grad(prev[0]))
      grads[prev[0].node_index] = grads[prev[0].node_index] + (-grad);
    break;

  case Op::POWER: {
    Symbol base = prev[0], exp_sym = prev[1];
    Symbol output{.node_index = index, .graph = &graph};
    if (needs_grad(base))
      grads[base.node_index] = grads[base.node_index] +
          unbroadcast(exp_sym * pow(base, exp_sym - 1.0f) * grad, base);
    if (needs_grad(exp_sym))
      grads[exp_sym.node_index] = grads[exp_sym.node_index] +
          unbroadcast(output * log(base) * grad, exp_sym);
    break;
  }

  case Op::EXP: {
    Symbol output{.node_index = index, .graph = &graph};
    if (needs_grad(prev[0]))
      grads[prev[0].node_index] = grads[prev[0].node_index] + output * grad;
    break;
  }

  case Op::LOG:
    if (needs_grad(prev[0]))
      grads[prev[0].node_index] = grads[prev[0].node_index] + grad / prev[0];
    break;

  case Op::MATMUL: {
    Symbol a = prev[0], b = prev[1];
    if (needs_grad(a))
      grads[a.node_index] = grads[a.node_index] + matmul(grad, transpose(b));
    if (needs_grad(b))
      grads[b.node_index] = grads[b.node_index] + matmul(transpose(a), grad);
    break;
  }

  case Op::IDENTITY:
    if (needs_grad(prev[0]))
      grads[prev[0].node_index] = grads[prev[0].node_index] + grad;
    break;

  case Op::TRANSPOSE:
    if (needs_grad(prev[0]))
      grads[prev[0].node_index] = grads[prev[0].node_index] + transpose(grad);
    break;

  case Op::SQRT: {
    Symbol output{.node_index = index, .graph = &graph};
    if (needs_grad(prev[0]))
      grads[prev[0].node_index] =
          grads[prev[0].node_index] + grad / (2.0f * output);
    break;
  }

  case Op::ABS:
    if (needs_grad(prev[0]))
      grads[prev[0].node_index] =
          grads[prev[0].node_index] + grad * sign(prev[0]);
    break;

  case Op::RELU:
    if (needs_grad(prev[0]))
      grads[prev[0].node_index] =
          grads[prev[0].node_index] + grad * (prev[0] > 0.0f);
    break;

  case Op::SUM: {
    Symbol inp = prev[0];
    if (needs_grad(inp)) {
      Symbol g = (!node.keepdims && has_axis())
                     ? expand_dims(grad, node.axes.axis)
                     : grad;
      grads[inp.node_index] =
          grads[inp.node_index] + broadcast_to_match(g, inp);
    }
    break;
  }

  case Op::MEAN: {
    Symbol inp = prev[0];
    if (needs_grad(inp)) {
      Symbol g = (!node.keepdims && has_axis())
                     ? expand_dims(grad, node.axes.axis)
                     : grad;
      Symbol n = size(inp, ax_opt());
      grads[inp.node_index] =
          grads[inp.node_index] + broadcast_to_match(g, inp) / n;
    }
    break;
  }

  case Op::VARIANCE: {
    Symbol inp = prev[0];
    if (needs_grad(inp)) {
      Symbol g = (!node.keepdims && has_axis())
                     ? expand_dims(grad, node.axes.axis)
                     : grad;
      Symbol n = size(inp, ax_opt());
      Symbol mu = mean(inp, ax_opt(), true);
      grads[inp.node_index] = grads[inp.node_index] +
          broadcast_to_match(g, inp) * 2.0f * (inp - mu) / n;
    }
    break;
  }

  case Op::SOFTMAX: {
    Symbol inp = prev[0];
    Symbol output{.node_index = index, .graph = &graph};
    if (needs_grad(inp)) {
      Symbol ntg = output * grad;
      grads[inp.node_index] = grads[inp.node_index] +
          ntg - sum(ntg, ax_opt(), true) * output;
    }
    break;
  }

  case Op::BROADCAST_TO:
  case Op::BROADCAST_TO_MATCH: {
    Symbol inp = prev[0];
    if (needs_grad(inp))
      grads[inp.node_index] = grads[inp.node_index] + unbroadcast(grad, inp);
    break;
  }

  case Op::UNBROADCAST: {
    Symbol inp = prev[0];
    if (needs_grad(inp))
      grads[inp.node_index] =
          grads[inp.node_index] + broadcast_to_match(grad, inp);
    break;
  }

  case Op::RESHAPE:
  case Op::RESHAPE_LIKE: {
    Symbol inp = prev[0];
    if (needs_grad(inp))
      grads[inp.node_index] = grads[inp.node_index] + reshape_like(grad, inp);
    break;
  }

  case Op::EXPAND_DIMS: {
    Symbol inp = prev[0];
    if (needs_grad(inp))
      grads[inp.node_index] =
          grads[inp.node_index] + squeeze(grad, ax_opt());
    break;
  }

  case Op::SQUEEZE: {
    Symbol inp = prev[0];
    if (needs_grad(inp)) {
      Symbol g = has_axis() ? expand_dims(grad, node.axes.axis)
                            : reshape_like(grad, inp);
      grads[inp.node_index] = grads[inp.node_index] + g;
    }
    break;
  }

  case Op::SWAP_AXIS: {
    Symbol inp = prev[0];
    if (needs_grad(inp))
      grads[inp.node_index] = grads[inp.node_index] +
          swap_axis(grad, node.axes.axis, node.axes.axis2);
    break;
  }

  case Op::VECTOR:
    for (uint32_t i = 0; i < node.input_count; i++)
      if (needs_grad(prev[i]))
        grads[prev[i].node_index] =
            grads[prev[i].node_index] + get_item(grad, (int)i);
    break;

  case Op::GET_ITEM: {
    Symbol inp = prev[0];
    if (needs_grad(inp))
      grads[inp.node_index] =
          grads[inp.node_index] + scatter_like(grad, inp, node.axes.axis);
    break;
  }

  case Op::SCATTER_LIKE: {
    Symbol inp = prev[0];
    if (needs_grad(inp))
      grads[inp.node_index] =
          grads[inp.node_index] + get_item(grad, node.axes.axis);
    break;
  }

  default:
    break;
  }
}

std::unordered_map<uint32_t, Symbol> backwards(const Symbol &symbol) {
  std::unordered_map<uint32_t, Symbol> grads;

  if (!symbol.graph)
    throw std::runtime_error("backwards: symbol has no graph");

  if (symbol.node_index >= symbol.graph->nodes.size())
    throw std::runtime_error("backwards: symbol node index out of range");

  if (!symbol.graph->nodes[symbol.node_index].requires_grad)
    throw std::runtime_error(
        "backwards: cannot backprop from a node that does not require grad");

  if (symbol.graph->nodes[symbol.node_index].operation == Op::VECTOR) {
    uint16_t num_inputs = symbol.graph->nodes[symbol.node_index].input_count;
    grads[symbol.node_index] =
        symbol.graph->constant(Eigen::MatrixXf::Ones(num_inputs, 1));
  } else {
    grads[symbol.node_index] =
        symbol.graph->constant(Eigen::MatrixXf::Ones(1, 1));
  }

  std::stack<Symbol> stack;
  stack.push(symbol);

  while (!stack.empty()) {
    Symbol current = stack.top();
    stack.pop();

    const Node &node = current.graph->nodes[current.node_index];

    if (node.requires_grad)
      backwards_node(*current.graph, current.node_index, grads);

    for (uint32_t i = 0; i < node.input_count; i++) {
      uint32_t input_id =
          i < 2 ? node.inputs[i]
                : current.graph->inputs[node.input_pool_offset + (i - 2)];
      if (!grads.count(input_id) &&
          current.graph->nodes[input_id].requires_grad) {
        grads[input_id] =
            current.graph->constant(Eigen::MatrixXf::Zero(1, 1));
        stack.push(Symbol{.node_index = input_id, .graph = current.graph});
      }
    }
  }

  return grads;
}

} // namespace autograd
