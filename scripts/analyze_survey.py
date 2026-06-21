#!/usr/bin/env python3
"""
Analyze arithmetic survey CSV and report success rates.
"""

import argparse
import math
import sys

import pandas as pd
from sympy import factorint, isprime


def load_survey(path: str) -> pd.DataFrame:
    """
    Load survey CSV and add derived columns.
    """
    survey = pd.read_csv(path)
    survey["solved"] = survey["num_solutions"] > 0
    survey["inputs_list"] = (
        survey["inputs"].str.split(";").apply(lambda xs: [int(x) for x in xs])
    )
    return survey


def pct(series: pd.Series) -> str:
    """
    Format a boolean series as a percentage string.
    """
    return f"{100 * series.mean():.2f}%"


def sol_stats(series: pd.Series) -> str:
    """
    Format solution count statistics.
    """
    return f"mean={series.mean():.1f}  median={series.median():.0f}  max={series.max()}"


def print_row(label: str, group: pd.DataFrame) -> None:
    """
    Print a single analysis row with success rate and solution stats.
    """
    print(
        f"  {label}  success={pct(group['solved'])}  "
        f"{sol_stats(group['num_solutions'])}  (n={len(group)})"
    )


def by_num_large(survey: pd.DataFrame) -> None:
    """
    Report stats grouped by large number count.
    """
    print("=== By large number count ===")
    for n_large, group in survey.groupby("num_large"):
        print_row(f"{n_large} large:", group)
    print()


def by_individual_number(survey: pd.DataFrame) -> None:
    """
    Report stats grouped by each individual input number present.
    """
    print("=== By each individual input number present ===")
    all_numbers = sorted({x for row in survey["inputs_list"] for x in row})
    for num in all_numbers:
        target = num
        mask = survey["inputs_list"].apply(lambda xs, t=target: t in xs)
        subset = survey.loc[mask]
        if len(subset) == 0:
            continue
        print_row(f"contains {num:>3d}:", subset)
    print()


def by_parity(survey: pd.DataFrame) -> None:
    """
    Report stats grouped by target even/odd.
    """
    print("=== By target parity ===")
    print_row("even target:", survey[survey["target"] % 2 == 0])
    print_row(" odd target:", survey[survey["target"] % 2 == 1])
    print()


def by_divisibility(survey: pd.DataFrame) -> None:
    """
    Report stats grouped by target divisibility by small numbers.
    """
    print("=== By target divisible by small number ===")
    for d in range(1, 11):
        print_row(f"target % {d:>2d} == 0:", survey[survey["target"] % d == 0])
    print()


def by_prime(survey: pd.DataFrame) -> None:
    """
    Report stats grouped by target primality.
    """
    print("=== By target is prime ===")
    prime_mask = survey["target"].apply(isprime)
    print_row("    prime target:", survey[prime_mask])
    print_row("composite target:", survey[~prime_mask])
    print()


def by_gcd_divisibility(survey: pd.DataFrame) -> None:
    """
    Report stats by whether the GCD of all inputs divides the target.
    """
    print("=== By GCD of inputs divides target ===")
    input_gcd = survey["inputs_list"].apply(lambda xs: math.gcd(*xs))
    divides = survey["target"] % input_gcd == 0
    print_row("  GCD divides target:", survey[divides])
    print_row("GCD !divides target:", survey[~divides])
    print()


def by_num_distinct_inputs(survey: pd.DataFrame) -> None:
    """
    Report stats by number of distinct values in inputs.
    """
    print("=== By number of distinct inputs ===")
    survey = survey.copy()
    survey["num_distinct"] = survey["inputs_list"].apply(lambda xs: len(set(xs)))
    for nd, group in survey.groupby("num_distinct"):
        print_row(f"{nd} distinct:", group)
    print()


def by_target_largest_prime_factor(survey: pd.DataFrame) -> None:
    """
    Report stats by the largest prime factor of the target.
    """
    print("=== By largest prime factor of target ===")
    survey = survey.copy()
    survey["lpf"] = survey["target"].apply(lambda t: max(factorint(t)))
    buckets = [
        (2, 5, "2-5"),
        (7, 10, "7-10"),
        (11, 20, "11-20"),
        (21, 50, "21-50"),
        (51, 100, "51-100"),
        (101, 999, ">100"),
    ]
    for lo, hi, label in buckets:
        mask = (survey["lpf"] >= lo) & (survey["lpf"] <= hi)
        if mask.sum() == 0:
            continue
        print_row(f"lpf {label:>7s}:", survey[mask])
    print()


def by_input_diversity(survey: pd.DataFrame) -> None:
    """
    Report stats by ratio of distinct inputs to total inputs.
    """
    print("=== By input diversity (distinct/total) ===")
    survey = survey.copy()
    survey["diversity"] = survey["inputs_list"].apply(lambda xs: len(set(xs)) / len(xs))
    buckets = [
        (0.0, 0.34, "low (1-2 distinct)"),
        (0.34, 0.67, "medium (3-4 distinct)"),
        (0.67, 1.01, "high (5-6 distinct)"),
    ]
    for lo, hi, label in buckets:
        mask = (survey["diversity"] > lo) & (survey["diversity"] <= hi)
        if lo == 0.0:
            mask = (survey["diversity"] >= lo) & (survey["diversity"] <= hi)
        if mask.sum() == 0:
            continue
        print_row(f"{label}:", survey[mask])
    print()


def by_target_distance_from_round(survey: pd.DataFrame) -> None:
    """
    Report stats by target distance from nearest multiple of 25.
    """
    print("=== By target distance from nearest multiple of 25 ===")
    survey = survey.copy()
    survey["dist_25"] = survey["target"].apply(lambda t: min(t % 25, 25 - t % 25))
    buckets = [
        (0, 0, "exact multiple"),
        (1, 3, "1-3 away"),
        (4, 6, "4-6 away"),
        (7, 12, "7-12 away"),
    ]
    for lo, hi, label in buckets:
        mask = (survey["dist_25"] >= lo) & (survey["dist_25"] <= hi)
        if mask.sum() == 0:
            continue
        print_row(f"{label:>16s}:", survey[mask])
    print()


def by_coprimality(survey: pd.DataFrame) -> None:
    """
    Report stats by how many inputs share a common factor with the target.
    """
    print("=== By number of inputs sharing a factor with target ===")
    survey = survey.copy()
    survey["shared_factors"] = survey.apply(
        lambda row: sum(
            1 for x in row["inputs_list"] if math.gcd(x, row["target"]) > 1
        ),
        axis=1,
    )
    for count, group in survey.groupby("shared_factors"):
        print_row(f"{count} inputs share factor:", group)
    print()


def _closest_product_distance(target: int, inputs: list[int]) -> int:
    """
    Find the minimum distance from target to any pairwise product of inputs.
    """
    best = abs(target - inputs[0])
    for i in range(len(inputs)):
        best = min(best, abs(target - inputs[i]))
        for j in range(i + 1, len(inputs)):
            best = min(best, abs(target - inputs[i] * inputs[j]))
    return best


def by_proximity_to_product(survey: pd.DataFrame) -> None:
    """
    Report stats by how close the target is to a pairwise product of inputs.
    """
    print("=== By target proximity to nearest pairwise product ===")
    survey = survey.copy()
    survey["prod_dist"] = survey.apply(
        lambda row: _closest_product_distance(row["target"], row["inputs_list"]),
        axis=1,
    )
    buckets = [
        (0, 0, "exact product"),
        (1, 5, "1-5 away"),
        (6, 15, "6-15 away"),
        (16, 50, "16-50 away"),
        (51, 150, "51-150 away"),
        (151, 10000, ">150 away"),
    ]
    for lo, hi, label in buckets:
        mask = (survey["prod_dist"] >= lo) & (survey["prod_dist"] <= hi)
        if mask.sum() == 0:
            continue
        print_row(f"{label:>15s}:", survey[mask])
    print()


def by_sum_coverage(survey: pd.DataFrame) -> None:
    """
    Report stats by relationship of target to sum and product of inputs.
    """
    print("=== By target relative to input sum ===")
    survey = survey.copy()
    survey["input_sum"] = survey["inputs_list"].apply(sum)
    survey["sum_ratio"] = survey["target"] / survey["input_sum"]
    buckets = [
        (0.0, 1.0, "target <= sum"),
        (1.0, 3.0, "target 1-3x sum"),
        (3.0, 10.0, "target 3-10x sum"),
        (10.0, 100.0, "target >10x sum"),
    ]
    for lo, hi, label in buckets:
        if lo == 0.0:
            mask = (survey["sum_ratio"] >= lo) & (survey["sum_ratio"] <= hi)
        else:
            mask = (survey["sum_ratio"] > lo) & (survey["sum_ratio"] <= hi)
        if mask.sum() == 0:
            continue
        print_row(f"{label:>18s}:", survey[mask])
    print()


def by_first_digit(survey: pd.DataFrame) -> None:
    """
    Report stats grouped by the first digit of the target.
    """
    print("=== By first digit of target ===")
    survey = survey.copy()
    survey["first_digit"] = survey["target"] // 100
    for digit, group in survey.groupby("first_digit"):
        print_row(f"starts with {digit}:", group)
    print()


def by_last_digit(survey: pd.DataFrame) -> None:
    """
    Report stats grouped by the last digit of the target.
    """
    print("=== By last digit of target ===")
    survey = survey.copy()
    survey["last_digit"] = survey["target"] % 10
    for digit, group in survey.groupby("last_digit"):
        print_row(f"ends with {digit}:", group)
    print()


def by_target_extremes(survey: pd.DataFrame) -> None:
    """
    Report the 5 easiest and 5 hardest targets by success rate.
    """
    print("=== Easiest and hardest targets ===")
    target_stats = survey.groupby("target").agg(
        n=("solved", "size"),
        success_rate=("solved", "mean"),
        mean_solutions=("num_solutions", "mean"),
        median_solutions=("num_solutions", "median"),
        max_solutions=("num_solutions", "max"),
    )
    # Only consider targets with enough samples to be meaningful
    min_samples = 20
    target_stats = target_stats[target_stats["n"] >= min_samples]
    easiest = target_stats.sort_values(
        ["success_rate", "mean_solutions"], ascending=[False, False]
    ).head(5)
    hardest = target_stats.sort_values(
        ["success_rate", "mean_solutions"], ascending=[True, True]
    ).head(5)
    print(f"  (only targets with >= {min_samples} samples)")
    print()
    print("  Top 5 easiest targets:")
    for target, row in easiest.iterrows():
        print(
            f"    target={target:>3d}:  success={100 * row['success_rate']:.2f}%  "
            f"mean={row['mean_solutions']:.1f}  median={row['median_solutions']:.0f}  "
            f"max={row['max_solutions']:.0f}  (n={row['n']:.0f})"
        )
    print()
    print("  Top 5 hardest targets:")
    for target, row in hardest.iterrows():
        print(
            f"    target={target:>3d}:  success={100 * row['success_rate']:.2f}%  "
            f"mean={row['mean_solutions']:.1f}  median={row['median_solutions']:.0f}  "
            f"max={row['max_solutions']:.0f}  (n={row['n']:.0f})"
        )
    print()


def main() -> None:
    """
    Entry point for survey analysis.
    """
    parser = argparse.ArgumentParser(description="Analyze arithmetic survey results")
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
    print_row("Overall:", survey)
    print()
    by_num_large(survey)
    by_individual_number(survey)
    by_parity(survey)
    by_divisibility(survey)
    by_prime(survey)
    by_gcd_divisibility(survey)
    by_num_distinct_inputs(survey)
    by_target_largest_prime_factor(survey)
    by_input_diversity(survey)
    by_target_distance_from_round(survey)
    by_coprimality(survey)
    by_proximity_to_product(survey)
    by_sum_coverage(survey)
    by_first_digit(survey)
    by_last_digit(survey)
    by_target_extremes(survey)


if __name__ == "__main__":
    main()
