#include "autograd/backward.hpp"
#include "autograd/graph.hpp"
#include "autograd/tensor.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stack>
#include <stdexcept>
#include <vector>

namespace autograd {

static void backwards_node(Graph &graph, uint32_t index,
                           std::vector<std::optional<Symbol>> &grads) {
  // Copy: any graph.add_node() call below may reallocate graph.nodes.
  const Node node = graph.nodes[index];

  if (node.operation == Op::CONSTANT || node.operation == Op::PLACEHOLDER ||
      node.operation == Op::VARIABLE)
    return;

  if (!node.requires_grad)
    return;

  if (!grads[index].has_value())
    return;

  Symbol grad = *grads[index];

  auto accum = [&](uint32_t idx, Symbol g) {
    grads[idx] = grads[idx].has_value() ? *grads[idx] + g : g;
  };
  auto has_axis = [&]() { return node.axes.axis >= 0; };
  auto ax_opt = [&]() -> std::optional<int> {
    return has_axis() ? std::optional<int>(node.axes.axis) : std::nullopt;
  };
  auto needs_grad = [&](uint32_t index) {
    return graph.nodes[index].requires_grad;
  };
  auto symbol = [&](uint32_t index) -> Symbol {
    return Symbol{.node_index = index, .graph = &graph};
  };

  switch (node.operation) {
  case Op::ADD:
    for (uint32_t i = 0; i < node.input_count; i++) {
      uint32_t prev_idx = i < 2
                              ? node.inputs[i]
                              : graph.inputs[node.input_pool_offset + (i - 2)];
      if (needs_grad(prev_idx))
        accum(prev_idx, unbroadcast(grad, symbol(prev_idx)));
    }
    break;

  case Op::MULTIPLY:
    for (uint32_t i = 0; i < node.input_count; i++) {
      uint32_t prev_idx = i < 2
                              ? node.inputs[i]
                              : graph.inputs[node.input_pool_offset + (i - 2)];
      if (needs_grad(prev_idx)) {
        Symbol g = grad;
        for (uint32_t j = 0; j < node.input_count; j++)
          if (i != j)
            g = g *
                symbol(j < 2 ? node.inputs[j]
                             : graph.inputs[node.input_pool_offset + (j - 2)]);
        accum(prev_idx, unbroadcast(g, symbol(prev_idx)));
      }
    }
    break;

  case Op::SUB: {
    uint32_t a = node.inputs[0], b = node.inputs[1];
    if (needs_grad(a))
      accum(a, unbroadcast(grad, symbol(a)));
    if (needs_grad(b))
      accum(b, unbroadcast(-grad, symbol(b)));
    break;
  }

  case Op::NEG: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp))
      accum(inp, -grad);
    break;
  }

  case Op::POWER: {
    uint32_t base = node.inputs[0], exp_sym = node.inputs[1];
    if (needs_grad(base))
      accum(base, unbroadcast(symbol(exp_sym) * pow(symbol(base), symbol(exp_sym) - 1.0f) * grad,
                              symbol(base)));
    if (needs_grad(exp_sym))
      accum(exp_sym, unbroadcast(symbol(index) * log(symbol(base)) * grad, symbol(exp_sym)));
    break;
  }

  case Op::EXP: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp))
      accum(inp, symbol(index) * grad);
    break;
  }

  case Op::LOG: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp))
      accum(inp, grad / symbol(inp));
    break;
  }

  case Op::MATMUL: {
    uint32_t a = node.inputs[0], b = node.inputs[1];
    if (needs_grad(a))
      accum(a, matmul(grad, transpose(symbol(b))));
    if (needs_grad(b))
      accum(b, matmul(transpose(symbol(a)), grad));
    break;
  }

  case Op::IDENTITY: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp))
      accum(inp, grad);
    break;
  }

  case Op::TRANSPOSE: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp))
      accum(inp, transpose(grad));
    break;
  }

  case Op::SQRT: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp))
      accum(inp, grad / (2.0f * symbol(index)));
    break;
  }

  case Op::ABS: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp))
      accum(inp, grad * sign(symbol(inp)));
    break;
  }

  case Op::RELU: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp))
      accum(inp, grad * (symbol(inp) > 0.0f));
    break;
  }

  case Op::SUM: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp)) {
      Symbol g = (!node.keepdims && has_axis())
                     ? expand_dims(grad, node.axes.axis)
                     : grad;
      accum(inp, broadcast_to_match(g, symbol(inp)));
    }
    break;
  }

  case Op::MEAN: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp)) {
      Symbol g = (!node.keepdims && has_axis())
                     ? expand_dims(grad, node.axes.axis)
                     : grad;
      accum(inp, broadcast_to_match(g, symbol(inp)) / size(symbol(inp), ax_opt()));
    }
    break;
  }

  case Op::VARIANCE: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp)) {
      Symbol g = (!node.keepdims && has_axis())
                     ? expand_dims(grad, node.axes.axis)
                     : grad;
      Symbol mu = mean(symbol(inp), ax_opt(), true);
      accum(inp, broadcast_to_match(g, symbol(inp)) * 2.0f *
                     (symbol(inp) - mu) / size(symbol(inp), ax_opt()));
    }
    break;
  }

  case Op::SOFTMAX: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp)) {
      Symbol output = symbol(index);
      Symbol ntg    = output * grad;
      accum(inp, ntg - sum(ntg, ax_opt(), true) * output);
    }
    break;
  }

  case Op::BROADCAST_TO:
  case Op::BROADCAST_TO_MATCH: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp))
      accum(inp, unbroadcast(grad, symbol(inp)));
    break;
  }

  case Op::UNBROADCAST: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp))
      accum(inp, broadcast_to_match(grad, symbol(inp)));
    break;
  }

  case Op::RESHAPE:
  case Op::RESHAPE_LIKE: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp))
      accum(inp, reshape_like(grad, symbol(inp)));
    break;
  }

  case Op::EXPAND_DIMS: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp))
      accum(inp, squeeze(grad, ax_opt()));
    break;
  }

  case Op::SQUEEZE: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp)) {
      Symbol g = has_axis() ? expand_dims(grad, node.axes.axis)
                            : reshape_like(grad, symbol(inp));
      accum(inp, g);
    }
    break;
  }

  case Op::SWAP_AXIS: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp))
      accum(inp, swap_axis(grad, node.axes.axis, node.axes.axis2));
    break;
  }

  case Op::VECTOR:
    for (uint32_t i = 0; i < node.input_count; i++) {
      uint32_t inp = i < 2 ? node.inputs[i]
                           : graph.inputs[node.input_pool_offset + (i - 2)];
      if (needs_grad(inp))
        accum(inp, get_item(grad, (int)i));
    }
    break;

  case Op::GET_ITEM: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp))
      accum(inp, scatter_like(grad, symbol(inp), node.axes.axis));
    break;
  }

  case Op::SCATTER_LIKE: {
    uint32_t inp = node.inputs[0];
    if (needs_grad(inp))
      accum(inp, get_item(grad, node.axes.axis));
    break;
  }

  default:
    break;
  }
}

std::vector<std::optional<Symbol>> backwards(const Symbol &symbol) {
  std::vector<std::optional<Symbol>> grads(symbol.graph->nodes.size());

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
        symbol.graph->constant(Tensor::ones(num_inputs, 1));
  } else {
    grads[symbol.node_index] =
        symbol.graph->constant(Tensor(1.0f));
  }

  std::vector<bool> pushed(symbol.graph->nodes.size(), false);
  pushed[symbol.node_index] = true;

  std::stack<Symbol> stack;
  stack.push(symbol);

  while (!stack.empty()) {
    Symbol current = stack.top();
    stack.pop();

    // Copy the node: backwards_node may push_back to graph.nodes, which
    // can reallocate the vector and invalidate any reference into it.
    const Node node = current.graph->nodes[current.node_index];

    if (node.requires_grad)
      backwards_node(*current.graph, current.node_index, grads);

    for (uint32_t i = 0; i < node.input_count; i++) {
      uint32_t input_id =
          i < 2 ? node.inputs[i]
                : current.graph->inputs[node.input_pool_offset + (i - 2)];
      if (!pushed[input_id] && current.graph->nodes[input_id].requires_grad) {
        pushed[input_id] = true;
        stack.push(Symbol{.node_index = input_id, .graph = current.graph});
      }
    }
  }

  return grads;
}

} // namespace autograd
