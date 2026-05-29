#pragma once

namespace autograd {

// Mirrors python/autograd/ops.py :: Operation
enum class Op {
    CONSTANT        = 0,
    VARIABLE        = 1,
    PLACEHOLDER     = 2,
    ADD             = 3,
    MULTIPLY        = 4,
    POWER           = 5,
    LOG             = 6,
    MATMUL          = 7,
    RMATMUL         = 8,
    TRANSPOSE       = 9,
    SUM             = 10,
    BROADCAST_TO    = 11,
    RESHAPE         = 12,
    EXPAND_DIMS     = 13,
    SQUEEZE         = 14,
    SWAP_AXIS       = 15,
    EXP             = 16,
    MEAN            = 17,
    RELU            = 18,
    SOFTMAX         = 19,
    VARIANCE        = 20,
    NEG             = 21,
    SUB             = 22,
    SQRT            = 24,
    IDENTITY        = 25,
    ABS             = 26,
    GREATER_THAN    = 27,
    SIZE            = 28,
    BROADCAST_TO_MATCH = 29,
    UNBROADCAST     = 30,
    LESS_THAN       = 31,
    RESHAPE_LIKE    = 32,
    SIGN            = 33,
    VECTOR               = 34,
    GET_ITEM             = 35,
    SCATTER_LIKE         = 36,
    EQUALS_TO            = 37,
    GREATER_THAN_OR_EQUAL = 38,
    LESS_THAN_OR_EQUAL   = 39,
};

} // namespace autograd
