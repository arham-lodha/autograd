from enum import Enum


class Operation(Enum):
    CONSTANT = 0
    VARIABLE = 1
    PLACEHOLDER = 2
    ADD = 3
    MULTIPLY = 4
    POWER = 5
    LOG = 6
    MATMUL = 7
    RMATMUL = 8
    TRANSPOSE = 9
    SUM = 10
    BROADCAST_TO = 11
    RESHAPE = 12
    EXPAND_DIMS = 13
    SQUEEZE = 14
    SWAP_AXIS = 15
    EXP = 16
    MEAN = 17
    RELU = 18
    SOFTMAX = 19
    VARIANCE = 20
    NEG = 21
    SUB = 22
    SQRT = 24
    IDENTITY = 25
    CUSTOM = -1
