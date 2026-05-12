from enum import Enum


class NodeType(Enum):

    VARIABLE = 0
    LITERAL = 1
    TASK = 2
    EXPRESSION = 3
    INTERMEDIATE = 4
    COLLECTION = 5

    UNKNOWN = 999


class EdgeType(Enum):

    DEF = 0
    USE = 1
    ORDER = 2
    KEYWORD = 3
    WHEN = 4
    LOOP = 5
    NOTIFIES = 6

    PARAMETER = 7
    LOOP_SOURCE = 8

    UNKNOWN = 999