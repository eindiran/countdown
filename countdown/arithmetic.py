#!/usr/bin/env python3
"""
arithmetic.py
-----------------

Solver for the arithmetic puzzles on Countdown.
"""

import itertools
import random
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


def loop_mode(loops: int):
    """
    Launch into looping over runs.
    """
    for _ in range(loops):
        target = random.randint(101, 999)
        inputs = [
            random.choice([25, 50, 75, 100]),
            random.choice([25, 50, 75, 100]),
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
        ]
        print(f"Target: {target}")
        print(f"Inputs: {inputs}")
        res = solve_cdarithmetic(target, inputs)
        print(f"Result: {res if res else 'no solution found'}\n\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve Countdown arithmetic puzzles from the CLI")
    subparsers = parser.add_subparsers()
    solve = subparsers.add_parser("solve", help="Command to run a single solution")
    solve.add_argument("target", type=int, help="Target integer")
    solve.add_argument("inputs", type=int, nargs="+", help="Input integers")
    loop = subparsers.add_parser("loop", help="Command to loop over random inputs")
    loop.add_argument("loops", type=int, help="Iniate looping n times over random runs")
    args = parser.parse_args()
    if vars(args).get("loops"):
        loop_mode(args.loops)
    else:
        print(solve_cdarithmetic(args.target, args.inputs))
