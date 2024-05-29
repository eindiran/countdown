#!/usr/bin/env python3
"""
countdown.py
------------

Countdown anagram and arithmetic puzzle solver.
"""

import argparse
import itertools
import os
import random
import statistics
import string
import sys
import threading
import time
from pprint import pprint
from typing import Callable, Optional, Union

import cv2  # type: ignore
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
ANAGRAM_ALLOWLIST = string.ascii_uppercase
ARITHMETIC_ALLOWLIST = string.digits + " |/"


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


def nlongest_anagrams(
    clue: str, word_set: set[str], n_longest: int = 5, lower_bound: int = 2
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
    len_delta: int = 0,
    n_longest: int = 5,
) -> list[AnagramAnswer]:
    """
    Return all of the matching anagrams.
    """
    lower_bound = len(clue) - len_delta
    return nlongest_anagrams(clue, word_set, n_longest, lower_bound)


def conundrum(clue: str, num: int = 5, word_set: Optional[set[str]] = None) -> list[AnagramAnswer]:
    """
    Print results for a final conundrum clue.
    """
    if not word_set:
        word_set = filter_word_list(cd_legal_filter)
    return all_matching_conundrums(clue, word_set, len_delta=0, n_longest=num)


def normal(clue: str, num: int = 5, word_set: Optional[set[str]] = None) -> list[AnagramAnswer]:
    """
    Print the results of a standard anagram.
    """
    if not word_set:
        word_set = filter_word_list(cd_legal_filter)
    return nlongest_anagrams(clue, word_set, num)


def anagram_loop_mode(loops: int, debug: bool = False) -> None:
    """
    Launch into looping over runs.
    """
    word_set = filter_word_list(cd_legal_filter)
    stats: list[int] = []
    for l in range(loops):
        # Legally only up to 5 vowels are allowed:
        num_vowels = random.choice([3, 4, 5])
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
        if debug:
            print(f"Loop: {l}")
            print(f"Letters: {' '.join(letters)}")
        res = nlongest_anagrams("".join(letters), word_set, 3)
        stats.append(int(res[0][1]))
        if debug:
            pprint(res)
            print("")
    print("Loops completed!")
    print(f"Total runs: {loops}")
    stats = sorted(stats)
    print(f"Arithmetic mean score: {statistics.mean(stats)}")
    print(f"Median score: {statistics.median(stats)}")
    print(f"Multimode score: {statistics.multimode(stats)}")
    print(f"Max score: {max(stats)}")
    print(f"Min score: {min(stats)}")
    print(f"Quartiles: {statistics.quantiles(stats, n=4)}")
    print(f"Population standard deviation: {statistics.pstdev(stats)}")
    print(f"Population variance: {statistics.pvariance(stats)}")


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


def arithmetic_loop_mode(loops: int, debug: bool = False) -> None:
    """
    Launch into looping over runs.
    """
    results = []
    for l in range(loops):
        target = random.randint(101, 999)
        inputs = [
            random.choice([25, 50, 75, 100]),
            random.choice([25, 50, 75, 100]),
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
        ]
        if debug:
            print(f"Loop: {l}")
            print(f"Target: {target}")
            print(f"Inputs: {inputs}")
        res = solve_cd_arithmetic(target, inputs)
        results.append(res)
        if debug:
            print(f"Result: {res if res else 'no solution found'}\n\n")
    print("Loops completed!")
    print(f"Total runs: {loops}")
    print(f"Total solutions found: {sum(1 for _ in results if _)}")
    print(f"Total solutions not found: {sum(1 for _ in results if not _)}")


def autoclosing_pyplot_fig(image, duration_s: int = 10) -> None:
    """
    Auto-close the mpl.pyplot figure.
    """

    def _stop():
        time.sleep(duration_s)
        plot.close()

    if duration_s:
        threading.Thread(target=_stop).start()
    plot.imshow(image)
    plot.show(block=False)
    plot.pause(float(duration_s))


def show_detected_text(
    image, detected, color=RED_RGB_TUPLE, thickness: int = 3, display_length: NullableInt = None
) -> None:
    """
    Tool for showing detected text via opencv and matplotlib.
    """
    if display_length == 0:
        # Don't display if someone specified specifically 0
        return
    for d in detected:
        top_left = tuple(d[0][0])
        bottom_right = tuple(d[0][2])
        image = cv2.rectangle(image, top_left, bottom_right, color, thickness)  # pylint: disable=no-member
    if display_length:
        autoclosing_pyplot_fig(image, duration_s=display_length)
    else:
        plot.imshow(image)
        plot.show()


def preprocess_image(img_path: str, preprocess: bool, greyscale: bool = False):
    """
    Run image pre-processing on the image.
    """
    if not preprocess:
        return cv2.imread(img_path)  # pylint: disable=no-member
    if greyscale:
        image = cv2.imread(img_path, 0)  # pylint: disable=no-member
        image = cv2.equalizeHist(image)  # pylint: disable=no-member
    else:
        image = cv2.imread(img_path)  # pylint: disable=no-member
        # Split out blue only:
        _, _, image = cv2.split(image)  # pylint: disable=no-member
    image = cv2.GaussianBlur(image, (5, 5), 1)  # pylint: disable=no-member
    return image


# pylint: disable=too-many-arguments
def cd_screenshot_ocr_arithmetic(
    img_path: str,
    debug: bool,
    recog_network: str = "standard",
    detect_network: str = "craft",
    preprocess: bool = True,
    greyscale: bool = False,
    display_length: NullableInt = None,
) -> None:
    """
    Given a path to an image, perform OCR with easyocr and return
    the arithmetic solution.
    """
    reader = easyocr.Reader(
        ["en"], gpu=True, recog_network=recog_network, detect_network=detect_network
    )
    image = preprocess_image(img_path, preprocess, greyscale=greyscale)
    detected = reader.readtext(image, allowlist=ARITHMETIC_ALLOWLIST)
    if debug:
        pprint(detected)
        # Show a red line if we are pre-processing, otherwise use lime green
        color = RED_RGB_TUPLE if preprocess else GREEN_RGB_TUPLE
        show_detected_text(image, detected, color=color, display_length=display_length)
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
    res = solve_cd_arithmetic(target, inputs)
    print(f"Result: {res if res else 'no solution found'}\n")


# pylint: disable=too-many-arguments, too-many-locals
def cd_screenshot_ocr_anagram(
    img_path: str,
    debug: bool,
    recog_network: str = "standard",
    detect_network: str = "craft",
    preprocess: bool = True,
    greyscale: bool = False,
    display_length: NullableInt = None,
    text_threshold: float = 0.55,
    contrast_threshold: float = 0.25,
) -> None:
    """
    Given a path to an image, perform OCR with easyocr and return
    the anagram solution.
    """
    reader = easyocr.Reader(
        ["en"], gpu=True, recog_network=recog_network, detect_network=detect_network
    )
    image = preprocess_image(img_path, preprocess, greyscale=greyscale)
    detected = reader.readtext(
        image,
        allowlist=ANAGRAM_ALLOWLIST,
        text_threshold=text_threshold,
        contrast_ths=contrast_threshold,
    )
    if debug:
        pprint(detected)
        show_detected_text(image, detected, display_length=display_length)
    letters: list[str] = []
    for d in detected:
        letters.append(d[1].lower())
    # Missing characters are most often "i"
    raw_clue = "".join(letters)
    print(f"Detected clue: {raw_clue}")
    clue = raw_clue.ljust(9, "i")
    if raw_clue != clue:
        print(f"Using clue (with i-padding): {clue}")
    normal(clue, 5)


# pylint: disable=too-many-branches,too-many-statements
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
        help="Choose which puzzle type to solve (default: anagram)",
    )
    loop_subcommand.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Print complete debug info for each item in the loop",
    )
    video_subcommand = subparsers.add_parser(
        "video", help="Command for running OCR on a video of an episode of Countdown"
    )
    video_subcommand.add_argument("video_path", type=str, help="Path to Countdown video")
    video_subcommand.add_argument(
        "-d", "--debug", action="store_true", help="Video OCR debugging info"
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
        help="Choose which puzzle type to solve (default: anagram)",
    )
    ocr_subcommand.add_argument(
        "-r",
        "--recog-network",
        dest="recog_network",
        choices=["standard", "english_g2", "latin_g2", "latin_g1"],
        default="standard",
        required=False,
        help="Choose the recognition network for EasyOCR from the CLI",
    )
    ocr_subcommand.add_argument(
        "-e",
        "--detect-network",
        dest="detect_network",
        choices=["craft", "dbnet18"],
        default="craft",
        required=False,
        help="Choose the detect network for EasyOCR from the CLI",
    )
    ocr_subcommand.add_argument(
        "-n",
        "--no-preprocess",
        dest="preprocess",
        action="store_false",
        help="Don't preprocess the image (default: False)",
    )
    ocr_subcommand.add_argument(
        "-g", "--greyscale", action="store_true", help="Load the image greyscale"
    )
    ocr_subcommand.add_argument(
        "-d", "--debug", action="store_true", help="Show the detected text via matplotlib"
    )
    ocr_subcommand.add_argument(
        "-l",
        "--display-length",
        dest="display_length",
        type=int,
        help="How long to display the processed image (default: None)",
    )
    args = parser.parse_args()
    vars_args = vars(args)
    if vars_args.get("loops"):
        if args.type == "anagram":
            anagram_loop_mode(args.loops, args.debug)
        else:
            # args.type == "arithmetic"
            arithmetic_loop_mode(args.loops, args.debug)
    elif vars_args.get("video_path"):
        # Validate the video file:
        if not os.path.isfile(args.video_path):
            print(f"<ERROR> Path to video file {args.video_path} does not exist")
            ocr_subcommand.print_help()
            sys.exit(1)
        print(f"Proceeding with video file: {args.video_path}")
    elif vars_args.get("image_path"):
        # Validate the image file:
        if not os.path.isfile(args.image_path):
            print(f"<ERROR> Path to image {args.image_path} does not exist")
            ocr_subcommand.print_help()
            sys.exit(1)
        if args.type == "anagram":
            cd_screenshot_ocr_anagram(
                args.image_path,
                args.debug,
                recog_network=args.recog_network,
                detect_network=args.detect_network,
                preprocess=args.preprocess,
                greyscale=args.greyscale,
                display_length=vars_args.get("display_length", None),
            )
        else:
            # args.type == "arithmetic"
            cd_screenshot_ocr_arithmetic(
                args.image_path,
                args.debug,
                recog_network=args.recog_network,
                detect_network=args.detect_network,
                preprocess=args.preprocess,
                greyscale=args.greyscale,
                display_length=vars_args.get("display_length", None),
            )
    elif vars_args.get("num"):
        if args.conundrum:
            pprint(conundrum(args.clue.lower(), args.num))
        else:
            pprint(normal(args.clue.lower(), args.num))
    elif vars_args.get("target"):
        print(solve_cd_arithmetic(args.target, args.inputs))
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
