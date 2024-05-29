#!/usr/bin/env bash
#===============================================================================
#
#          FILE: run_loop_tests.sh
#
#         USAGE: ./scripts/run_loop_tests.sh
#
#   DESCRIPTION: Run the loop tests. Run this from the top dir.
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
    echo "usage: ./scripts/run_loop_tests.sh [-r] [-n] [-h]"
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
    echo "Running arithmetic loop tests"
    ./countdown/countdown.py loop -t arithmetic 1000
fi

if test "${run_anagram_tests}"; then
    echo "Running anagram loop tests"
    ./countdown/countdown.py loop -t anagram 1000
fi

echo "Loop tests complete!"
