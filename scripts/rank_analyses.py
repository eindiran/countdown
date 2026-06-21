#!/usr/bin/env python3
"""Rank survey analyses by predictive power (success rate spread)."""

from __future__ import annotations

import argparse
import math
import sys

import pandas as pd
from sympy import factorint, isprime

RankResult = tuple[str, float, float, float, int]


def load_survey(path: str) -> pd.DataFrame:
    """Load survey CSV and add derived columns."""
    survey = pd.read_csv(path)
    survey["solved"] = survey["num_solutions"] > 0
    survey["inputs_list"] = (
        survey["inputs"].str.split(";").apply(lambda xs: [int(x) for x in xs])
    )
    return survey


def _spread(name: str, rates: list[float]) -> RankResult:
    """Compute success rate range as a simple measure of predictive power."""
    return (name, max(rates) - min(rates), min(rates), max(rates), len(rates))


def _bucketed_rates(
    series: pd.Series,
    solved: pd.Series,
    buckets: list[tuple[float, float]],
) -> list[float]:
    """Compute success rates for a series across value buckets."""
    rates = []
    for lo, hi in buckets:
        mask = (series >= lo) & (series <= hi)
        if mask.sum() > 0:
            rates.append(solved[mask].mean())
    return rates


def _closest_product_distance(target: int, inputs: list[int]) -> int:
    """Find the minimum distance from target to any pairwise product of inputs."""
    best = abs(target - inputs[0])
    for i in range(len(inputs)):
        best = min(best, abs(target - inputs[i]))
        for j in range(i + 1, len(inputs)):
            best = min(best, abs(target - inputs[i] * inputs[j]))
    return best


def _rank_groupby_analyses(survey: pd.DataFrame) -> list[RankResult]:
    """Analyses that use simple groupby or boolean splits."""
    results = []

    # Large number count
    rates = [g["solved"].mean() for _, g in survey.groupby("num_large")]
    results.append(_spread("Large number count", rates))

    # Individual input number
    all_nums = sorted({x for row in survey["inputs_list"] for x in row})
    rates = []
    for num in all_nums:
        target = num
        mask = survey["inputs_list"].apply(lambda xs, t=target: t in xs)
        rates.append(survey.loc[mask, "solved"].mean())
    results.append(_spread("Individual input number", rates))

    # Target parity
    rates = [
        survey.loc[survey["target"] % 2 == 0, "solved"].mean(),
        survey.loc[survey["target"] % 2 == 1, "solved"].mean(),
    ]
    results.append(_spread("Target parity", rates))

    # Target divisibility
    rates = [
        survey.loc[survey["target"] % d == 0, "solved"].mean() for d in range(2, 11)
    ]
    results.append(_spread("Target divisibility (2-10)", rates))

    # Target is prime
    prime_mask = survey["target"].apply(isprime)
    rates = [
        survey.loc[prime_mask, "solved"].mean(),
        survey.loc[~prime_mask, "solved"].mean(),
    ]
    results.append(_spread("Target is prime", rates))

    # GCD divides target
    input_gcd = survey["inputs_list"].apply(lambda xs: math.gcd(*xs))
    divides = survey["target"] % input_gcd == 0
    rates = [
        survey.loc[divides, "solved"].mean(),
        survey.loc[~divides, "solved"].mean(),
    ]
    results.append(_spread("GCD divides target", rates))

    # Distinct input count
    tmp = survey.copy()
    tmp["nd"] = tmp["inputs_list"].apply(lambda xs: len(set(xs)))
    rates = [g["solved"].mean() for _, g in tmp.groupby("nd")]
    results.append(_spread("Distinct input count", rates))

    # Shared factors with target
    tmp = survey.copy()
    tmp["shared_factors"] = tmp.apply(
        lambda row: sum(
            1 for x in row["inputs_list"] if math.gcd(x, row["target"]) > 1
        ),
        axis=1,
    )
    rates = [g["solved"].mean() for _, g in tmp.groupby("shared_factors")]
    results.append(_spread("Shared factors with target", rates))

    # First digit of target
    tmp = survey.copy()
    tmp["first_digit"] = tmp["target"] // 100
    rates = [g["solved"].mean() for _, g in tmp.groupby("first_digit")]
    results.append(_spread("First digit of target", rates))

    # Last digit of target
    tmp = survey.copy()
    tmp["last_digit"] = tmp["target"] % 10
    rates = [g["solved"].mean() for _, g in tmp.groupby("last_digit")]
    results.append(_spread("Last digit of target", rates))

    return results


def _rank_bucketed_analyses(survey: pd.DataFrame) -> list[RankResult]:
    """Analyses that bucket a derived column into ranges."""
    results = []

    # Largest prime factor
    lpf = survey["target"].apply(lambda t: max(factorint(t)))
    rates = _bucketed_rates(
        lpf,
        survey["solved"],
        [(2, 5), (7, 10), (11, 20), (21, 50), (51, 100), (101, 999)],
    )
    results.append(_spread("Largest prime factor", rates))

    # Input diversity
    diversity = survey["inputs_list"].apply(lambda xs: len(set(xs)) / len(xs))
    rates = _bucketed_rates(
        diversity,
        survey["solved"],
        [(0.0, 0.34), (0.34, 0.67), (0.67, 1.01)],
    )
    results.append(_spread("Input diversity", rates))

    # Distance from multiple of 25
    dist_25 = survey["target"].apply(lambda t: min(t % 25, 25 - t % 25))
    rates = _bucketed_rates(
        dist_25,
        survey["solved"],
        [(0, 0), (1, 3), (4, 6), (7, 12)],
    )
    results.append(_spread("Distance from multiple of 25", rates))

    # Proximity to pairwise product
    prod_dist = survey.apply(
        lambda row: _closest_product_distance(row["target"], row["inputs_list"]),
        axis=1,
    )
    rates = _bucketed_rates(
        prod_dist,
        survey["solved"],
        [(0, 0), (1, 5), (6, 15), (16, 50), (51, 150), (151, 10000)],
    )
    results.append(_spread("Proximity to pairwise product", rates))

    # Target relative to input sum
    sum_ratio = survey["target"] / survey["inputs_list"].apply(sum)
    rates = _bucketed_rates(
        sum_ratio,
        survey["solved"],
        [(0, 1), (1, 3), (3, 10), (10, 100)],
    )
    results.append(_spread("Target relative to input sum", rates))

    return results


def compute_rankings(survey: pd.DataFrame) -> list[RankResult]:
    """Compute predictive spread for all analyses."""
    results = _rank_groupby_analyses(survey) + _rank_bucketed_analyses(survey)
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def main() -> None:
    """Entry point for analysis ranking."""
    parser = argparse.ArgumentParser(
        description="Rank survey analyses by predictive power",
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="survey_results.csv",
        help="Path to survey CSV (default: survey_results.csv)",
    )
    args = parser.parse_args()

    try:
        survey = load_survey(args.csv_path)
    except FileNotFoundError:
        print(f"Error: {args.csv_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(survey)} puzzles from {args.csv_path}")
    print()

    results = compute_rankings(survey)

    print(f"{'Analysis':<35s} {'Spread':>7s}  {'Min':>7s}  {'Max':>7s}  Buckets")
    print("-" * 75)
    for name, sp, mn, mx, n in results:
        print(f"{name:<35s} {100 * sp:6.1f}%  {100 * mn:6.1f}%  {100 * mx:6.1f}%  {n}")


if __name__ == "__main__":
    main()
