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
from collections.abc import Callable
from pprint import pprint

import cv2  # type: ignore
import easyocr  # type: ignore
import matplotlib.pyplot as plot  # type: ignore
import numpy  # type: ignore

# Types:
FilterType = Callable[[str], bool]
AnagramAnswer = list[str | int]
ArithmeticSequence = list[int] | tuple[int, ...]
RGBTuple = tuple[int, int, int]
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
# Color RGB tuples for coloring returned OCR
# images in debug mode:
RED_RGB_TUPLE: RGBTuple = (255, 0, 0)
GREEN_RGB_TUPLE: RGBTuple = (0, 255, 0)
BLUE_RGB_TUPLE: RGBTuple = (0, 0, 255)
WHITE_RGB_TUPLE: RGBTuple = (255, 255, 255)
# Allowed characters in the anagram OCR; the font used on
# the show is all upper-case.
ANAGRAM_ALLOWLIST = string.ascii_uppercase
# Allow whitespace and slash chars, after some testing this
# improves the arithmetic OCR accuracy quite a bit since we
# can split on them, rather than running together adjacent
# clues.
ARITHMETIC_ALLOWLIST = string.digits + " |/"
# Number of letters in the anagram rounds and the number
# of letters in the conundrum round:
CD_WORD_LEN = 9
# Number of distinct clues in the arithmetic rounds:
CD_ARITH_LEN = 6


class OCRDetectionError(Exception):
    """
    Raise this when OCR fails to detect the required text.
    """


def generate_word_set(wfilter: FilterType) -> set[str]:
    """
    Return a list of words matching a filter,
    from WORD_LIST

    KNOWN LIMITATIONS:
        1) This does not match the OED word list
        2) American orthography
        3) Does not account for special rules, like
           the rule allowing compounds not listed in
           the dictionary for 1 syllable words.
    """
    words = set()
    with open(WORD_LIST, encoding="utf8") as f:
        for line in f:
            w = line.strip()
            if wfilter(w):
                words.add(w.lower())
    return words


def cd_legal_filter(x: str) -> bool:
    """
    Filter to get all legal Countdown words (no
    proper nouns, up to 9 letters long).
    """
    return not x[0].isupper() and len(x) < CD_WORD_LEN + 1


def nlongest_anagrams(
    clue: str, word_set: set[str], n_longest: int = 5, lower_bound: int = 2
) -> list[AnagramAnswer]:
    """
    Find the longest anagrams matching a clue; return the n_longest,
    sorted by length (then lexicographic order within a length).
    """
    matched: set[str] = set()
    for i in range(lower_bound, len(clue) + 1):
        for perm_chars in itertools.permutations(clue, i):
            perm = "".join(perm_chars)
            if perm in word_set:
                matched.add(perm)
    longest_matches = sorted(matched, key=len, reverse=True)[:n_longest]
    return sorted([[m, len(m)] for m in longest_matches], key=lambda x: (x[1], x[0]), reverse=True)


def nlongest_conundrums(
    clue: str,
    word_set: set[str],
    len_delta: int = 0,
    n_longest: int = 5,
) -> list[AnagramAnswer]:
    """
    Return the n_longest matching anagrams; used to solve conundrums where
    we only care about perfect anagrams. This is controlled with the len_delta parameter.
    """
    lower_bound = len(clue) - len_delta
    return nlongest_anagrams(clue, word_set, n_longest, lower_bound)


def solve_cd_conundrum(
    clue: str, num: int = 5, word_set: set[str] | None = None
) -> list[AnagramAnswer]:
    """
    Helper function for calling nlongest_anagrams for a conundrum.
    """
    if not word_set:
        word_set = generate_word_set(cd_legal_filter)
    return nlongest_conundrums(clue, word_set, len_delta=0, n_longest=num)


def solve_cd_anagram(
    clue: str, num: int = 5, word_set: set[str] | None = None
) -> list[AnagramAnswer]:
    """
    Helper function for calling nlongest_anagrams for a single anagram puzzle.
    """
    if not word_set:
        word_set = generate_word_set(cd_legal_filter)
    return nlongest_anagrams(clue, word_set, num)


def generate_random_anagram_clue() -> list[str]:
    """
    Generate a random anagram puzzle. Tries to follow the rules of Countdown
    so the generated puzzle is plausible. Weights come from VOWEL_WEIGHTS
    and CONSONANT_WEIGHTS above, which are derived from this page:
        http://thecountdownpage.com/letters.htm
    """
    # Legally only up to 5 vowels are allowed:
    num_vowels = random.choice([3, 4, 5])
    num_consonants = 9 - num_vowels
    letters: list[str] = []
    letters.extend(
        random.choices(
            [_ for _ in VOWEL_WEIGHTS],
            weights=[int(x) for x in VOWEL_WEIGHTS.values()],
            k=num_vowels,
        )
    )
    letters.extend(
        random.choices(
            [_ for _ in CONSONANT_WEIGHTS],
            weights=[int(x) for x in CONSONANT_WEIGHTS.values()],
            k=num_consonants,
        )
    )
    # Randomize the list order just to make it look more
    # presentable.
    random.shuffle(letters)
    return letters


def anagram_loop_mode(loops: int, debug: bool = False) -> None:
    """
    Launch into looping over runs.
    """
    word_set = generate_word_set(cd_legal_filter)
    stats: list[int] = []
    for l in range(loops):
        letters = generate_random_anagram_clue()
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


def _add(a: int | None, b: int | None) -> int | None:
    """
    Addition function that supports null.
    """
    if a is None or b is None:
        return None
    return a + b


def _sub(a: int | None, b: int | None) -> int | None:
    """
    Subtraction function that supports null.
    """
    if a is None or b is None:
        return None
    return a - b


def _mul(a: int | None, b: int | None) -> int | None:
    """
    Multiplication function that supports null.
    """
    if a is None or b is None:
        return None
    return a * b


def _div(a: int | None, b: int | None) -> int | None:
    """
    Division function that only allows integer division without
    remainders and supports null.
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


def solve_single_arithmetic_ordering(target: int | None, inputs: ArithmeticSequence) -> list[str]:
    """
    Evaluate solutions for a single "ordering" of integer clues in
    an arithmetic problem. See solve_cd_arithmetic below for a sense of
    what this is used for.
    """
    if target is None:
        raise ValueError("Can't solve ordering with null target")
    operator_slots = len(inputs) - 1
    for op_ordering in itertools.product(ARITHMETIC_OPERATIONS, repeat=operator_slots):
        value: int | None = inputs[0]
        for i in range(operator_slots):
            value = op_ordering[i][1](value, inputs[i + 1])
        if value == target:
            return [o[0] for o in op_ordering]
    return []


def solve_cd_arithmetic(target: int | None, inputs: ArithmeticSequence) -> str:
    """
    Solve a Countdown arithmetic problem.

    KNOWN LIMITATIONS:
        1) This method ONLY works for solutions that can be evaluated
           linearly from left-to-right. Based on some testing, there
           are relatively few cases where there is no linear solution
           but there is a solution, and supporting non-linear solutions
           is order of magnitude slower, so I opted for the faster,
           dumber algorithm.
        2) The string formatting at the end takes this fact (1) into
           account, and doesn't bother adding parens which makes some
           solutions read incorrectly by PEMDAS.
    """
    if target is None:
        raise ValueError("Can't solve ordering with null target")
    for i in range(1, len(inputs) + 1):
        for perm in itertools.permutations(inputs, i):
            sol = solve_single_arithmetic_ordering(target, perm)
            if sol:
                final: list[str] = [str(perm[0])]
                for j in range(len(perm) - 1):
                    final.append(sol[j])
                    final.append(str(perm[j + 1]))
                return " ".join(final)
    return ""


def generate_random_arithmetic_clue(
    num_large: int = 2, num_small: int = 4, target_lb: int = 101, target_ub: int = 999
) -> tuple[int, list[int]]:
    """
    Generate a target and input clues for an arithmetic puzzle.

    Target is an integer between 101 and 999 inclusive, however this can be
    controlled for unusual games with the lower and upper bounds (target_lb
    and target_ub respectively).

    The number of inputs in a typical arithmetic puzzle is 6:
    Either 1 large and 5 small or 2 large and 4 small (latter being the default).

    This can be controlled using the num_large and num_small parameters.
    """
    target = random.randint(target_lb, target_ub)
    large_inputs = [random.choice([25, 50, 75, 100]) for _ in range(num_large)]
    small_inputs = [random.randint(1, 10) for _ in range(num_small)]
    inputs = large_inputs + small_inputs
    return (target, inputs)


def arithmetic_loop_mode(loops: int, debug: bool = False) -> None:
    """
    Launch into looping over arithmetic runs.
    """
    results = []
    for l in range(loops):
        target, inputs = generate_random_arithmetic_clue()
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


def autoclosing_pyplot_fig(
    image: numpy.ndarray, duration_s: int = 10, greyscale: bool = False
) -> None:
    """
    Auto-closing mpl.pyplot figure.
    """

    def _stop() -> None:
        time.sleep(duration_s)
        plot.close()

    if duration_s:
        threading.Thread(target=_stop).start()
    if greyscale:
        plot.imshow(image, cmap="gray")
    else:
        plot.imshow(image)  # Default color map
    plot.show(block=False)
    plot.pause(float(duration_s))


def show_detected_text(  # noqa: PLR0913
    image: numpy.ndarray,
    detected: list[tuple[list[list[int]], str, numpy.float64]],
    color: RGBTuple = RED_RGB_TUPLE,
    thickness: int = 3,
    display_length: int | None = None,
    greyscale: bool = False,
) -> None:
    """
    Tool for showing detected text via opencv and matplotlib.
    """
    if display_length == 0:
        # Don't display if someone specified specifically 0
        return
    for d in detected:
        top_left = tuple(int(x) for x in d[0][0])
        bottom_right = tuple(int(x) for x in d[0][2])
        image = cv2.rectangle(image, top_left, bottom_right, color, thickness)
    if display_length:
        autoclosing_pyplot_fig(image, duration_s=display_length, greyscale=greyscale)
    else:
        if greyscale:
            plot.imshow(image, cmap="gray")
        else:
            plot.imshow(image)  # Use default colormap
        plot.show()


def preprocess_image(image_path: str, preprocess: bool, greyscale: bool = False) -> numpy.ndarray:
    """
    Run image pre-processing on the image prior to OCR.
    """
    # Load image:
    if greyscale:
        # Option 1: Convert the image to greyscale
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Option 2: Return the image as-is
        image = cv2.imread(image_path)
    # Process image:
    if not preprocess:
        return image
    else:
        if greyscale:
            # Option 3: Greyscale + pre-processing
            image = cv2.equalizeHist(image)
        else:
            # Option 4: Blue-only pre-processing
            image = cv2.imread(image_path)
            # Split out blue only:
            _, _, image = cv2.split(image)
        image = cv2.GaussianBlur(image, (5, 5), 1)
    return image


def cd_screenshot_ocr_arithmetic(  # noqa: PLR0913
    image: numpy.ndarray,
    debug: bool,
    recog_network: str = "standard",
    detect_network: str = "craft",
    preprocess: bool = True,
    greyscale: bool = False,
    display_length: int | None = None,
) -> None:
    """
    Given a path to an image, perform OCR with easyocr and return
    the arithmetic solution.
    """
    reader = easyocr.Reader(
        ["en"], gpu=True, recog_network=recog_network, detect_network=detect_network
    )
    detected = reader.readtext(image, allowlist=ARITHMETIC_ALLOWLIST)
    if not detected:
        print("Nothing detected, continuing to next segment...")
        raise OCRDetectionError("Empty detected list")
    if len(detected) < CD_ARITH_LEN - 3:
        print("Noise detected, continuing to next segment...")
        raise OCRDetectionError("Truncated detected list")
    if debug:
        pprint(detected)
        # Show a red line if we are pre-processing, otherwise use lime green
        color = RED_RGB_TUPLE if preprocess else GREEN_RGB_TUPLE
        if greyscale:
            color = WHITE_RGB_TUPLE
        show_detected_text(
            image,
            detected,
            color=color,
            display_length=display_length,
            greyscale=greyscale,
        )
    target = None
    inputs: ArithmeticSequence = []
    for d in detected:
        # Target first
        if not target:
            target = int(d[1].replace("/", " ").replace("|", " ").split()[0])
        else:
            inputs.extend([int(_) for _ in d[1].replace("/", " ").replace("|", " ").split()])
    print(f"Detected target: {target}")
    print(f"Detected inputs: {inputs}")
    res = solve_cd_arithmetic(target, inputs)
    print(f"Result: {res if res else 'no solution found'}\n")


def cd_screenshot_ocr_anagram(  # noqa: PLR0913
    image: numpy.ndarray,
    debug: bool,
    recog_network: str = "standard",
    detect_network: str = "craft",
    preprocess: bool = True,
    greyscale: bool = False,
    display_length: int | None = None,
    text_threshold: float = 0.55,
    contrast_threshold: float = 0.25,
    word_set: set[str] | None = None,
) -> None:
    """
    Given a path to an image, perform OCR with easyocr and print
    the anagram solution.
    """
    reader = easyocr.Reader(
        ["en"], gpu=True, recog_network=recog_network, detect_network=detect_network
    )
    detected = reader.readtext(
        image,
        allowlist=ANAGRAM_ALLOWLIST,
        text_threshold=text_threshold,
        contrast_ths=contrast_threshold,
    )
    if not detected:
        print("Nothing detected, continuing to next segment...")
        raise OCRDetectionError("Empty detected list")
    letters: list[str] = []
    for d in detected:
        letters.append(d[1].lower())
    # Missing characters are most often "i"
    raw_clue = "".join(letters)
    if len(raw_clue) > CD_WORD_LEN + 3 or len(raw_clue) < CD_WORD_LEN - 2:
        print(f"Noise text ({raw_clue}), continuing...")
        return
    if debug:
        pprint(detected)
        # Show a red line if we are pre-processing, otherwise use lime green
        color = RED_RGB_TUPLE if preprocess else GREEN_RGB_TUPLE
        if greyscale:
            color = WHITE_RGB_TUPLE
        show_detected_text(
            image,
            detected,
            color=color,
            display_length=display_length,
            greyscale=greyscale,
        )
    print(f"Detected clue: {raw_clue}")
    clue = raw_clue.ljust(9, "i")
    if raw_clue != clue:
        print(f"Using clue (with i-padding): {clue}")
    print(f"Solutions: {solve_cd_anagram(clue, 5, word_set=word_set)}")


def cd_video_ocr(
    video_path: str,
    debug: bool,
    greyscale: bool = True,
    display_length: int | None = None,
) -> None:
    """
    Handle OCR for video files. Takes 360P video, due to pixel cropping defaults.
    Use youtube-dl format code 134.
    """
    cap = cv2.VideoCapture(video_path)
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frame count: {frame_length}")
    start = frame_length // 4  # Skip a quarter through
    print(f"Setting frame pointer to: {start}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_num = start
    word_set = generate_word_set(cd_legal_filter)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if debug:
                print(f"Video from file {video_path} is complete, exiting")
            return
        # Every 750th frame:
        if frame_num % 750 == 0:
            if greyscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Crop image:
            image = frame[220:360, 0:640]
            # Obviously, the above will not work if the video is an unexpected format
            # since we are trying to grab the bottom middle of the screen (where clues
            # are displayed for an extended period during 8OO10CDC).
            ss_ocr_completed = False
            try:
                cd_screenshot_ocr_arithmetic(
                    image,
                    debug,
                    preprocess=False,
                    greyscale=greyscale,
                    display_length=display_length,
                )
                ss_ocr_completed = True
            except (
                ZeroDivisionError,
                TypeError,
                ValueError,
                IndexError,
                OCRDetectionError,
            ):
                pass
            if not ss_ocr_completed:
                try:
                    cd_screenshot_ocr_anagram(
                        image,
                        debug,
                        preprocess=False,
                        greyscale=greyscale,
                        display_length=display_length,
                        word_set=word_set,
                    )
                except (
                    ZeroDivisionError,
                    TypeError,
                    ValueError,
                    IndexError,
                    OCRDetectionError,
                ):
                    pass
        frame_num += 1
    cap.release()
    print("Video processing complete!")


def main() -> None:  # noqa: PLR0912,PLR0915
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
    video_subcommand.add_argument(
        "-l",
        "--display-length",
        dest="display_length",
        type=int,
        help="How long to display the processed frame (default: None)",
    )
    video_subcommand.add_argument(
        "-g", "--greyscale", action="store_true", help="Load the image greyscale"
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
        "-d",
        "--debug",
        action="store_true",
        help="Show the detected text via matplotlib",
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
        cd_video_ocr(
            args.video_path,
            args.debug,
            display_length=args.display_length,
            greyscale=args.greyscale,
        )
    elif vars_args.get("image_path"):
        # Validate the image file:
        if not os.path.isfile(args.image_path):
            print(f"<ERROR> Path to image {args.image_path} does not exist")
            ocr_subcommand.print_help()
            sys.exit(1)
        if args.type == "anagram":
            image = preprocess_image(
                args.image_path, preprocess=args.preprocess, greyscale=args.greyscale
            )
            cd_screenshot_ocr_anagram(
                image,
                args.debug,
                recog_network=args.recog_network,
                detect_network=args.detect_network,
                preprocess=args.preprocess,
                greyscale=args.greyscale,
                display_length=vars_args.get("display_length", None),
            )
        else:
            # args.type == "arithmetic"
            image = preprocess_image(
                args.image_path, preprocess=args.preprocess, greyscale=args.greyscale
            )
            cd_screenshot_ocr_arithmetic(
                image,
                args.debug,
                recog_network=args.recog_network,
                detect_network=args.detect_network,
                preprocess=args.preprocess,
                greyscale=args.greyscale,
                display_length=vars_args.get("display_length", None),
            )
    elif vars_args.get("num"):
        if args.conundrum:
            pprint(solve_cd_conundrum(args.clue.lower(), args.num))
        else:
            pprint(solve_cd_anagram(args.clue.lower(), args.num))
    elif vars_args.get("target"):
        print(solve_cd_arithmetic(args.target, args.inputs))
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
