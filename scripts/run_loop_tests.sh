#!/usr/bin/env bash
#===============================================================================
#
#          FILE: run_loop_tests.sh
#
#         USAGE: ./scripts/run_loop_tests.sh [-h] [-r] [-n] [-l <LOOPS>] [-d]
#
#   DESCRIPTION: Run the loop tests. Run this from the top dir.
#
#       OPTIONS:
#                   -h:  Print usage and exit
#                   -r:  Run arithmetic tests only
#                   -n:  Run anagram tests only
#                   -d:  Enable debug mode
#                   -l:  Specify the number of loops to run (default 1000)
#  REQUIREMENTS: Python3, inside venv with requirements.txt installed.
#         NOTES: ---
#===============================================================================

set -Eeuo pipefail

run_arithmetic_tests=true
run_anagram_tests=true
debug_mode=false
loop_num=1000

usage() {
    # Print usage and exit with $1
    echo "run_loop_tests.sh"
    echo "-----------------"
    echo "  Script to run the loop tests panel and generate"
    echo "  stats on the runs."
    echo
    echo "Usage: ./scripts/run_loop_tests.sh [-r] [-n] [-h] [-l <LOOPS>] [-d]"
    echo "   No arguments runs both sets of loop tests (arithmetic and anagrams)"
    echo "   -r: runs the arithmetic tests only"
    echo "   -n: runs the anagram tests only"
    echo "   -l: specify the number of loops to run (default 1000)"
    echo "   -d: enable debug mode"
    echo "   -h: print help and exit"
    echo
    exit "$1"
}

while getopts "hrndl:" option; do
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
        d)
            debug_mode=true
            ;;
        l)
            loop_num="${OPTARG}"
            ;;
        *)
            printf "Unknown option: %s\n" "${option}"
            usage 1
            ;;
    esac
done
shift $((OPTIND-1))

if [[ "${run_arithmetic_tests}" == true ]]; then
    echo "Running arithmetic loop tests"
    if [[ "${debug_mode}" == true ]]; then
        ./countdown/countdown.py loop --debug --type arithmetic "${loop_num}"
    else
        ./countdown/countdown.py loop --type arithmetic "${loop_num}"
    fi
else
    echo "Skipping arithmetic loop tests"
fi

if [[ "${run_anagram_tests}" == true ]]; then
    echo "Running anagram loop tests"
    if [[ "${debug_mode}" == true ]]; then
        ./countdown/countdown.py loop --debug --type anagram "${loop_num}"
    else
        ./countdown/countdown.py loop --type anagram "${loop_num}"
    fi
else
    echo "Skipping anagram loop tests"
fi

echo "Loop tests complete!"
