#!/usr/bin/env bash
#===============================================================================
#
#          FILE: run_loop_tests.sh
#
#         USAGE: ./scripts/run_loop_tests.sh [-h] [-r] [-n]
#
#   DESCRIPTION: Run the loop tests. Run this from the top dir.
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
    echo "run_loop_tests.sh"
    echo "-----------------"
    echo "  Script to run the loop tests panel and generate"
    echo "  stats on the runs."
    echo
    echo "Usage: ./scripts/run_loop_tests.sh [-r] [-n] [-h]"
    echo "   No arguments runs both sets of loop tests (arithmetic and anagrams)"
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
    echo "Running arithmetic loop tests"
    ./countdown/countdown.py loop -t arithmetic 1000
else
    echo "Skipping arithmetic loop tests"
fi

if [[ "${run_anagram_tests}" == true ]]; then
    echo "Running anagram loop tests"
    ./countdown/countdown.py loop -t anagram 1000
else
    echo "Skipping anagram loop tests"
fi

echo "Loop tests complete!"
