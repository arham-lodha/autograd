#pragma once
#include "autograd/graph.hpp"
#include "autograd/symbol.hpp"
#include <unordered_map>

namespace autograd {

std::unordered_map<uint32_t, Symbol> backwards(const Symbol &output);

} // namespace autograd
