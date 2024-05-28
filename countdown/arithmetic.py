#!/usr/bin/env python3
"""
arithmetic.py
-----------------

Solver for the arithmetic puzzles on Countdown.
"""

import itertools
from typing import Union

NullableInt = Union[int, None]


def add(a: NullableInt, b: NullableInt) -> NullableInt:
    """
    Addition fn.
    """
    if a is None or b is None:
        return None
    return a + b


def sub(a: NullableInt, b: NullableInt) -> NullableInt:
    """
    Subtraction function.
    """
    if a is None or b is None:
        return None
    return a - b


def mul(a: NullableInt, b: NullableInt) -> NullableInt:
    """
    Multiplication function.
    """
    if a is None or b is None:
        return None
    return a * b


def div(a: NullableInt, b: NullableInt) -> NullableInt:
    """
    Division function.
    """
    if a is None or b is None:
        return None
    if a % b == 0:
        return a // b
    return None


CD_ARITHMETIC_OPERATIONS = (
    ("+", add),
    ("-", sub),
    ("*", mul),
    ("/", div),
)


def solve_single_ordering(target: int, inputs):
    """
    Solve a single ordering.
    """
    operator_slots = len(inputs) - 1
    for op_ordering in itertools.product(CD_ARITHMETIC_OPERATIONS, repeat=operator_slots):
        value = inputs[0]
        for i in range(operator_slots):
            value = op_ordering[i][1](value, inputs[i + 1])
        if value == target:
            return [o[0] for o in op_ordering]
    return []


def solve_cdarithmetic(target: int, inputs):
    """Solve a Countdown arithmetic problem."""
    for i in range(1, len(inputs) + 1):
        for perm in itertools.permutations(inputs, i):
            sol = solve_single_ordering(target, perm)
            if sol:
                final: list[str] = [str(perm[0])]
                for i in range(len(perm) - 1):
                    final.append(sol[i])
                    final.append(str(perm[i + 1]))
                return " ".join(final)
    return ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve Countdown arithmetic puzzles from the CLI")
    parser.add_argument("target", type=int, help="Target integer")
    parser.add_argument("inputs", type=int, nargs="+", help="Input integers")
    args = parser.parse_args()

    print(solve_cdarithmetic(args.target, args.inputs))
