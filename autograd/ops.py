from enum import Enum


class Operation(Enum):
    CONSTANT = 0
    ADD = 1
    MULTIPLY = 2
    POWER = 4
    LOG = 5
    MATMUL = 6
    RMATMUL = 7
    TRANSPOSE = 8
    SUM = 9
    BROADCAST_TO = 10
    RESHAPE = 11
    EXPAND_DIMS = 12
    SQUEEZE = 13
    SWAP_AXIS = 14
    EXP = 15
    MEAN = 16
    RELU = 17
