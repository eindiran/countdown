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
#  REQUIREMENTS: youtube-dl, ffmpeg, rename(1) [perl verison]
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
            youtube_url="${OPTARG}"
            ;;
        h)
            usage 0
            ;;
        *)
            printf "Unknown option: %s\n" "${option}"
            usage 1
            ;;
    esac
done
shift $((OPTIND-1))


if [ -n "${youtube_url}" ]; then
    printf "Continuing in directory: ocr-test/videos/\n"
    cd ocr-test/videos
    printf "Using YouTube URL %s\n" "${youtube_url}"
    youtube-dl -o '%(title)s.%(ext)s' -f 134 "${youtube_url}" --verbose
    printf "Renaming downloaded video"
    rename 's/.*(S[0-9]{2}E[0-9]{2}) - (.*) (.*) (.*).mp4/$1-$2$3$4.mp4/' ./*.mp4
else
    echo "No URL specified. Nothing to do."
    usage 0
fi

echo "Downloading complete!"
