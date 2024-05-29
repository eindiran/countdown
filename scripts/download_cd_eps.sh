#!/usr/bin/env bash
#===============================================================================
#
#          FILE: download_cd_eps.sh
#
#         USAGE: ./scripts/download_cd_eps.sh [-h] [-u <URL>]
#
#   DESCRIPTION: Download a number of episodes of 8 Out of 10 Cats Does
#                Countdown from YouTube for use with the OCR toolset.
#
#       OPTIONS: ---
#  REQUIREMENTS: youtube-dl, ffmpeg
#         NOTES: ---
#===============================================================================

set -Eeuo pipefail

usage() {
    # Print the usage and exit with $1
    echo "download_cd_eps.sh"
    echo "------------------"
    echo "  Download episodes of 8 Out of 10 Cats Does Countdown from YouTube"
    echo "  and stores them in ocr-tests/videos/"
    echo
    echo "Usage: ./scripts/download_cd_eps.sh [-h] [-u <URL>]"
    echo "   -h: print help and exit"
    echo "   -u: specify the URL used (playlist or single episode)"
    echo
    exit "$1"
}

youtube_url=""

while getopts "hu:" option; do
    case "${option}" in
        u)
            shift 1
            youtube_url="${1}"
            ;;
        h)
            usage 0
            ;;
        *)
            printf "Unknown option: %s\n" "${option}"
            usage 1
            ;;
    esac
    shift 1
done


if [ -n "${youtube_url}" ]; then
    printf "Using YouTube URL %s\n" "${youtube_url}"
else
    echo "No URL specified. Nothing to do."
    usage 0
fi
