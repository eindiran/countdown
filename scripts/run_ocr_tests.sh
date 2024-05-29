#!/usr/bin/env bash
#===============================================================================
#
#          FILE: run_ocr_tests.sh
#
#         USAGE: ./scripts/run_ocr_tests.sh [-h] [-r] [-n]
#
#   DESCRIPTION: Run the OCR tests. Run this from the top dir.
#
#       OPTIONS:
#                   -h:  Print usage and exit
#                   -r:  Run arithmetic tests only
#                   -n:  Run anagram tests only
#  REQUIREMENTS: Python3, inside venv with requirements.txt installed.
#         NOTES: ---
#===============================================================================

set -Eeuo pipefail

run_arithmetic_tests=true
run_anagram_tests=true

usage() {
    # Print usage and exit with $1
    echo "run_ocr_tests.sh"
    echo "----------------"
    echo "  Script to run the OCR test panel using"
    echo "  screenshots from ocr-tests/anagrams and"
    echo "  ocr-tests/arithmetic"
    echo
    echo "Usage: ./scripts/run_ocr_tests.sh [-r] [-n] [-h]"
    echo "   No arguments runs both sets (anagrams and arithmetic)"
    echo "   -r: runs the arithmetic tests only"
    echo "   -n: runs the anagram tests only"
    echo "   -h: print help and exit"
    echo
    exit "$1"
}

while getopts "hrn" option; do
    case "${option}" in
        h)
            usage 0
            ;;
        r)
            run_anagram_tests=false
            ;;
        n)
            run_arithmetic_tests=false
            ;;
        *)
            printf "Unknown option: %s\n" "${option}"
            usage 1
            ;;
    esac
done

if [[ "${run_arithmetic_tests}" == true ]]; then
    echo "Running arithmetic OCR tests"
    set -o xtrace
    for test_image in ocr-test/arithmetic/*.png; do
        ./countdown/countdown.py ocr "${test_image}" -t arithmetic --debug --preprocess -l 1
    done
    set +x
else
    echo "Skipping arithmetic OCR tests"
fi

if [[ "${run_anagram_tests}" == true ]]; then
    echo "Running anagram OCR tests"
    set -o xtrace
    for test_image in ocr-test/anagrams/*.png; do
        ./countdown/countdown.py ocr "${test_image}" -t anagram --debug --preprocess -l 1
    done
    set +x
else
    echo "Skipping anagram OCR tests"
fi

echo "OCR tests complete!"
