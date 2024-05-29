#!/usr/bin/env python3
"""
countdown.py
------------

Countdown anagram and arithmetic puzzle solver.
"""

import argparse
import itertools
import random
import sys
from pprint import pprint
from typing import Callable, Union

import cv2
import easyocr  # type: ignore
import matplotlib.pyplot as plot  # type: ignore

# Types:
FilterType = Callable[[str], bool]
AnagramAnswer = list[Union[str, int]]
NullableInt = Union[int, None]
# Globals:
WORD_LIST = "/usr/share/dict/words"
## SEE HERE: http://thecountdownpage.com/letters.htm
VOWEL_WEIGHTS = {
    "a": 15,
    "e": 21,
    "i": 13,
    "o": 13,
    "u": 5,
}
CONSONANT_WEIGHTS = {
    "b": 2,
    "c": 3,
    "d": 6,
    "f": 2,
    "g": 3,
    "h": 2,
    "j": 1,
    "k": 1,
    "l": 5,
    "m": 4,
    "n": 8,
    "p": 4,
    "q": 1,
    "r": 9,
    "s": 9,
    "t": 9,
    "v": 1,
    "w": 1,
    "x": 1,
    "y": 1,
    "z": 1,
}
RED_RGB_TUPLE = (255, 0, 0)
GREEN_RGB_TUPLE = (0, 255, 0)
BLUE_RGB_TUPLE = (0, 0, 255)


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


def cd_legal_filter(x: str) -> bool:
    """
    Filter to get all legal Countdown words (no
    proper nouns).
    """
    return not x[0].isupper() and len(x) < 10


def conundrum_filter(x: str) -> bool:
    """
    We only want nine letter words that are not capitalized.
    """
    return len(x) == 9 and cd_legal_filter(x)


def interstitial_filter(x: str) -> bool:
    """
    We only want 8 and 9 letter words that are not capitalized.
    """
    return len(x) in {8, 9} and cd_legal_filter(x)


def nlongest_anagrams(
    clue: str, word_set: set[str], n_longest: int = 1, lower_bound: int = 2
) -> list[AnagramAnswer]:
    """
    Find the longest anagrams matching a clue.
    """
    matched: set[str] = set()
    for i in range(lower_bound, len(clue) + 1):
        for perm_chars in itertools.permutations(clue, i):
            perm = "".join(perm_chars)
            if perm in word_set:
                matched.add(perm)
    longest_matches = sorted(matched, key=len, reverse=True)[:n_longest]
    return sorted([[m, len(m)] for m in longest_matches], key=lambda x: (x[1], x[0]), reverse=True)


def all_matching_conundrums(
    clue: str,
    word_set: set[str],
    filter_: FilterType,
    len_delta: int = 0,
    n_longest: int = 1,
) -> list[AnagramAnswer]:
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
    normal_word_set = filter_word_list(cd_legal_filter)
    pprint(nlongest_anagrams(clue, normal_word_set, num))


def anagram_loop_mode(loops: int) -> None:
    """
    Launch into looping over runs.
    """
    normal_word_set = filter_word_list(cd_legal_filter)
    for _ in range(loops):
        num_vowels = random.choice([3, 4, 5, 6])
        num_consonants = 9 - num_vowels
        letters = []
        letters.extend(
            random.choices(
                [str(x) for x in VOWEL_WEIGHTS],
                weights=[int(x) for x in VOWEL_WEIGHTS.values()],
                k=num_vowels,
            )
        )
        letters.extend(
            random.choices(
                [str(x) for x in CONSONANT_WEIGHTS],
                weights=[int(x) for x in CONSONANT_WEIGHTS.values()],
                k=num_consonants,
            )
        )
        print(f"Letters: {' '.join(letters)}")
        pprint(nlongest_anagrams("".join(letters), normal_word_set, 3))
        print("")


def _add(a: NullableInt, b: NullableInt) -> NullableInt:
    """
    Addition fn.
    """
    if a is None or b is None:
        return None
    return a + b


def _sub(a: NullableInt, b: NullableInt) -> NullableInt:
    """
    Subtraction function.
    """
    if a is None or b is None:
        return None
    return a - b


def _mul(a: NullableInt, b: NullableInt) -> NullableInt:
    """
    Multiplication function.
    """
    if a is None or b is None:
        return None
    return a * b


def _div(a: NullableInt, b: NullableInt) -> NullableInt:
    """
    Division function.
    """
    if a is None or b is None:
        return None
    if a % b == 0:
        return a // b
    return None


ARITHMETIC_OPERATIONS = (
    ("+", _add),
    ("-", _sub),
    ("*", _mul),
    ("/", _div),
)


def solve_single_ordering(target: NullableInt, inputs):
    """
    Solve a single ordering.
    """
    if target is None:
        raise ValueError("Can't solve ordering with null target")
    operator_slots = len(inputs) - 1
    for op_ordering in itertools.product(ARITHMETIC_OPERATIONS, repeat=operator_slots):
        value = inputs[0]
        for i in range(operator_slots):
            value = op_ordering[i][1](value, inputs[i + 1])
        if value == target:
            return [o[0] for o in op_ordering]
    return []


def solve_cd_arithmetic(target: NullableInt, inputs) -> str:
    """Solve a Countdown arithmetic problem."""
    if target is None:
        raise ValueError("Can't solve ordering with null target")
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


def arithmetic_loop_mode(loops: int) -> None:
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
        res = solve_cd_arithmetic(target, inputs)
        print(f"Result: {res if res else 'no solution found'}\n\n")


def show_detected_text(img_path: str, detected, color=GREEN_RGB_TUPLE, thickness: int = 3) -> None:
    """
    Tool for showing detected text via opencv and matplotlib.
    """
    img = cv2.imread(img_path)  # pylint: disable=no-member
    for d in detected:
        top_left = tuple(d[0][0])
        bottom_right = tuple(d[0][2])
        img = cv2.rectangle(img, top_left, bottom_right, color, thickness)  # pylint: disable=no-member
    plot.imshow(img)
    plot.show()


def cd_ocr_arithmetic(
    img_path: str, debug: bool, recog_network: str = "standard", detect_network: str = "craft"
) -> None:
    """
    Given a path to an image, perform OCR with easyocr and return
    the arithmetic solution.
    """
    reader = easyocr.Reader(
        ["en"], gpu=True, recog_network=recog_network, detect_network=detect_network
    )
    detected = reader.readtext(img_path, allowlist="0123456789 |/")
    if debug:
        pprint(detected)
        show_detected_text(img_path, detected)
    target = None
    inputs: list[int] = []
    for d in detected:
        # Target first
        if not target:
            target = int(d[1])
        else:
            inputs.extend([int(_) for _ in d[1].replace("/", " ").replace("|", " ").split()])
    print(f"Detected target: {target}")
    print(f"Detected inputs: {inputs}")
    print(solve_cd_arithmetic(target, inputs))


def cd_ocr_anagram(
    img_path: str, debug: bool, recog_network: str = "standard", detect_network: str = "craft"
) -> None:
    """
    Given a path to an image, perform OCR with easyocr and return
    the anagram solution.
    """
    reader = easyocr.Reader(
        ["en"], gpu=True, recog_network=recog_network, detect_network=detect_network
    )
    detected = reader.readtext(
        img_path, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ", text_threshold=0.55, contrast_ths=0.25
    )
    if debug:
        pprint(detected)
        show_detected_text(img_path, detected)
    letters: list[str] = []
    for d in detected:
        letters.append(d[1].lower())
    clue = "".join(letters)
    print(f"Detected clue: {clue}")
    normal(clue, 5)


# pylint: disable=too-many-branches
def main() -> None:
    """
    Main function that adds parsers for CLI control.
    """
    parser = argparse.ArgumentParser(
        prog="countdown.py",
        description="Solve Countdown anagrams and arithmetic puzzles from the CLI",
    )
    subparsers = parser.add_subparsers(help="sub-command help")
    arithmetic_subcommand = subparsers.add_parser(
        "arithmetic", help="Command to run a single solution"
    )
    arithmetic_subcommand.add_argument("target", type=int, help="Target integer")
    arithmetic_subcommand.add_argument("inputs", type=int, nargs="+", help="Input integers")
    anagram_subcommand = subparsers.add_parser(
        "anagram", help="Command to run a single anagram solution"
    )
    anagram_subcommand.add_argument("clue", type=str, help="Input word / clue")
    anagram_controller_group = anagram_subcommand.add_mutually_exclusive_group()
    anagram_controller_group.add_argument(
        "-i",
        "--interstitial",
        action="store_true",
        help="Toggle on interstitial conundrums",
    )
    anagram_controller_group.add_argument(
        "-c",
        "--conundrum",
        action="store_true",
        help="Toggle on final conundrum",
    )
    anagram_subcommand.add_argument(
        "-n",
        "--num",
        type=int,
        default=5,
        required=False,
        help="Return a different number of anagrams (default: 5)",
    )
    loop_subcommand = subparsers.add_parser("loop", help="Command to loop over random inputs")
    loop_subcommand.add_argument("loops", type=int, help="Iniate looping n times over random runs")
    loop_subcommand.add_argument(
        "-t",
        "--type",
        choices=["anagram", "arithmetic"],
        default="anagram",
        required=False,
        help="Choose which puzzle type to solve",
    )
    ocr_subcommand = subparsers.add_parser(
        "ocr", help="Command for running OCR on a screenshot of Countdown"
    )
    ocr_subcommand.add_argument("image_path", type=str, help="Path to Countdown screenshot")
    ocr_subcommand.add_argument(
        "-t",
        "--type",
        choices=["anagram", "arithmetic"],
        default="anagram",
        required=False,
        help="Choose which puzzle type to solve",
    )
    ocr_subcommand.add_argument(
        "-r",
        "--recog_network",
        choices=["standard", "english_g2", "latin_g2", "latin_g1"],
        default="standard",
        required=False,
        help="Choose the recognition network for EasyOCR from the CLI",
    )
    ocr_subcommand.add_argument(
        "-e",
        "--detect_network",
        choices=["craft", "dbnet18"],
        default="craft",
        required=False,
        help="Choose the detect network for EasyOCR from the CLI",
    )
    ocr_subcommand.add_argument(
        "-d", "--debug", action="store_true", help="Show the detected text via matplotlib"
    )
    args = parser.parse_args()
    vars_args = vars(args)
    if vars_args.get("loops"):
        if args.type == "anagram":
            anagram_loop_mode(args.loops)
        elif args.type == "arithmetic":
            arithmetic_loop_mode(args.loops)
        else:
            raise ValueError(f"Unknown loop mode type: {args.type}")
    elif vars_args.get("image_path"):
        if args.type == "anagram":
            cd_ocr_anagram(
                args.image_path,
                args.debug,
                recog_network=args.recog_network,
                detect_network=args.detect_network,
            )
        elif args.type == "arithmetic":
            cd_ocr_arithmetic(
                args.image_path,
                args.debug,
                recog_network=args.recog_network,
                detect_network=args.detect_network,
            )
        else:
            raise ValueError(f"Unknown loop mode type: {args.type}")
    elif vars_args.get("num"):
        if args.conundrum:
            conundrum(args.clue.lower(), args.num)
        elif args.interstitial:
            interstitial(args.clue.lower(), args.num)
        else:
            normal(args.clue.lower(), args.num)
    elif vars_args.get("target"):
        print(solve_cd_arithmetic(args.target, args.inputs))
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
