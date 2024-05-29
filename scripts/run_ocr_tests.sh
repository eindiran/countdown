#!/usr/bin/env bash
#===============================================================================
#
#          FILE: run_ocr_tests.sh
#
#         USAGE: ./scripts/run_ocr_tests.sh
#
#   DESCRIPTION: Run the OCR tests. Run this from the top dir.
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#         NOTES: ---
#===============================================================================

set -Eeuo pipefail

run_arithmetic_tests=true
run_anagram_tests=true

usage() {
    # Print usage and exit with $1
    echo "usage: ./scripts/run_ocr_tests.sh [-r] [-n] [-h]"
    echo "-r: runs the arithmetic tests only"
    echo "-n: runs the anagram tests only"
    echo "no arguments runs both sets"
    echo "-h: print help and exit"
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


if test "${run_arithmetic_tests}"; then
    echo "Running arithmetic OCR tests"
    set -o xtrace
    for test_image in ocr-test/arithmetic/*.png; do
        ./countdown/countdown.py ocr "${test_image}" -t arithmetic --debug --preprocess -l 1
    done
    set +x
fi

if test "${run_anagram_tests}"; then
    echo "Running anagram OCR tests"
    set -o xtrace
    for test_image in ocr-test/anagrams/*.png; do
        ./countdown/countdown.py ocr "${test_image}" -t anagram --debug --preprocess -l 1
    done
    set +x
fi

echo "OCR tests complete!"
