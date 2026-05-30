#pragma once
#include "autograd/graph.hpp"
#include "autograd/symbol.hpp"
#include <optional>
#include <vector>

namespace autograd {

std::vector<std::optional<Symbol>> backwards(const Symbol &output);

} // namespace autograd
