#!/usr/bin/env python3
"""
countdown_anagram.py
------------

Countdown anagram solver.
"""

import itertools
from typing import Union, Callable
from pprint import pprint


# Types:
type FilterType = Callable[[str], bool]
type Maybe[T] = Union[T, None]
# Globals:
WORD_LIST = "/usr/share/dict/words"


def filter_word_list(wfilter: FilterType) -> set[str]:
    """
    Return a list of words matching a filter,
    from WORD_LIST
    """
    words = set()
    with open(WORD_LIST, "r", encoding="utf8") as f:
        for line in f:
            w = line.strip()
            if wfilter(w):
                words.add(w.lower())
    return words


def cdlegal_filter(x: str) -> bool:
    """
    Filter to get all legal Countdown words (no
    proper nouns).
    """
    return not x[0].isupper()


def conundrum_filter(x: str) -> bool:
    """
    We only want nine letter words that are not capitalized.
    """
    return len(x) == 9 and cdlegal_filter(x)


def interstitial_filter(x: str) -> bool:
    """
    We only want 8 and 9 letter words that are not capitalized.
    """
    return len(x) in {8, 9} and cdlegal_filter(x)


def nlongest_anagrams(
    clue: str, word_set: set[str], n_longest: int = 1, lower_bound: int = 2
) -> Maybe[set[str]]:
    """
    Find the longest anagrams matching a clue.
    """
    matched: set[str] = set()
    for i in range(lower_bound, len(clue) + 1):
        for perm_chars in itertools.permutations(clue, i):
            perm = "".join(perm_chars)
            if perm in word_set:
                matched.add(perm)
    return sorted(matched, key=len, reverse=True)[:n_longest]


def all_matching_conundrums(
    clue: str,
    word_set: set[str],
    filter_: FilterType,
    len_delta: int = 0,
    n_longest: int = 1,
) -> set[str]:
    """
    Return all of the matching anagrams.

    Throws AssertionError on clue not matching the filter.
    """
    assert filter_(clue)
    lower_bound = len(clue) - len_delta
    return nlongest_anagrams(clue, word_set, n_longest, lower_bound)


def conundrum(clue: str, num: int = 1) -> None:
    """
    Print results for a final conundrum clue.
    """
    conundrum_word_set = filter_word_list(conundrum_filter)
    pprint(
        all_matching_conundrums(
            clue, conundrum_word_set, conundrum_filter, len_delta=0, n_longest=num
        )
    )


def interstitial(clue: str, num: int = 1) -> None:
    """
    Print results for an interstitial conundrum clue.

    Use len_delta since they can be 8 or 9 chars long.
    """
    interstitial_word_set = filter_word_list(interstitial_filter)
    pprint(
        all_matching_conundrums(
            clue, interstitial_word_set, interstitial_filter, len_delta=1, n_longest=num
        )
    )


def normal(clue: str, num: int = 5) -> None:
    """
    Print the results of a standard anagram.
    """
    normal_word_set = filter_word_list(cdlegal_filter)
    pprint(nlongest_anagrams(clue, normal_word_set, num))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="countdown_anagram.py",
        description="Solve Countdown conundrums from the CLI",
    )
    parser.add_argument("clue", type=str, help="Input word / clue")
    controller_group = parser.add_mutually_exclusive_group()
    controller_group.add_argument(
        "-i",
        "--interstitial",
        action="store_true",
        help="Toggle on interstitial conundrums",
    )
    controller_group.add_argument(
        "-c",
        "--conundrum",
        action="store_true",
        help="Toggle on final conundrum",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=5,
        required=False,
        help="Return a different number of anagrams (default: 5)",
    )
    args = parser.parse_args()
    clue = args.clue.lower()
    if args.conundrum:
        conundrum(clue, args.num)
    elif args.interstitial:
        interstitial(clue, args.num)
    else:
        normal(clue, args.num)
