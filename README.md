# countdown
Messing around with some scripts for automatically solving puzzles from the show [8 Out of 10 Cats Does Countdown](https://en.wikipedia.org/wiki/8_Out_of_10_Cats_Does_Countdown), a comedy panel/game show with anagram and arithmetic puzzles.

The `countdown.py` script uses [EasyOCR](https://github.com/JaidedAI/EasyOCR) and [OpenCV](https://opencv.org/) to handle parsing the puzzles from screenshots or video, as well as presenting a CLI for interacting with solving puzzles as text. Additionally, the `survey` subcommand can be used for data analysis on which arithmetic puzzles are solveable and how many solutions they have.

The manual for `countdown.py` is given below.

### Manual:

```
---------
TOP-LEVEL
---------
usage: countdown.py [-h] {arithmetic,anagram,loop,video,ocr} ...

Solve Countdown anagrams and arithmetic puzzles from the CLI

positional arguments:
  {arithmetic,anagram,loop,survey,video,ocr}
                        sub-command help
    arithmetic          Command to run a single solution
    anagram             Command to run a single anagram solution
    loop                Command to loop over random inputs
    survey              Generate random arithmetic puzzles and count distinct solutions
    video               Command for running OCR on a video of an episode of Countdown
    ocr                 Command for running OCR on a screenshot of Countdown

options:
  -h, --help            show this help message and exit

----------------------
SUBCOMMAND: ARITHMETIC
----------------------
usage: countdown.py arithmetic [-h] target inputs [inputs ...]

positional arguments:
  target      Target integer
  inputs      Input integers

options:
  -h, --help  show this help message and exit

-------------------
SUBCOMMAND: ANAGRAM
-------------------
usage: countdown.py anagram [-h] [-c] [-n NUM] clue

positional arguments:
  clue               Input word / clue

options:
  -h, --help         show this help message and exit
  -c, --conundrum    Toggle on final conundrum
  -n NUM, --num NUM  Return a different number of anagrams (default: 5)

----------------
SUBCOMMAND: LOOP
----------------
usage: countdown.py loop [-h] [-t {anagram,arithmetic}] [-d] loops

positional arguments:
  loops                 Iniate looping n times over random runs

options:
  -h, --help            show this help message and exit
  -t {anagram,arithmetic}, --type {anagram,arithmetic}
                        Choose which puzzle type to solve (default: anagram)
  -d, --debug           Print complete debug info for each item in the loop

------------------
SUBCOMMAND: SURVEY
------------------
usage: countdown.py survey [-h] [-n NUM_PUZZLES] [-o OUTPUT] [-w WORKERS]

options:
  -h, --help            show this help message and exit
  -n, --num-puzzles NUM_PUZZLES
                        Number of puzzles to generate (default: 100000)
  -o, --output OUTPUT   Output CSV file path (default: survey_results.csv)
  -w, --workers WORKERS
                        Number of worker processes (default: cpu count)

-----------------
SUBCOMMAND: VIDEO
-----------------
usage: countdown.py video [-h] [-d] [-l DISPLAY_LENGTH] [-g] video_path

positional arguments:
  video_path            Path to Countdown video

options:
  -h, --help            show this help message and exit
  -d, --debug           Video OCR debugging info
  -l DISPLAY_LENGTH, --display-length DISPLAY_LENGTH
                        How long to display the processed frame (default: None)
  -g, --greyscale       Load the image greyscale

---------------
SUBCOMMAND: OCR
---------------
usage: countdown.py ocr [-h] [-t {anagram,arithmetic}]
                        [-r {standard,english_g2,latin_g2,latin_g1}] [-e {craft,dbnet18}]
                        [-n] [-g] [-d] [-l DISPLAY_LENGTH]
                        image_path

positional arguments:
  image_path            Path to Countdown screenshot

options:
  -h, --help            show this help message and exit
  -t {anagram,arithmetic}, --type {anagram,arithmetic}
                        Choose which puzzle type to solve (default: anagram)
  -r {standard,english_g2,latin_g2,latin_g1}, --recog-network {standard,english_g2,latin_g2,latin_g1}
                        Choose the recognition network for EasyOCR from the CLI
  -e {craft,dbnet18}, --detect-network {craft,dbnet18}
                        Choose the detect network for EasyOCR from the CLI
  -n, --no-preprocess   Don't preprocess the image (default: False)
  -g, --greyscale       Load the image greyscale
  -d, --debug           Show the detected text via matplotlib
  -l DISPLAY_LENGTH, --display-length DISPLAY_LENGTH
                        How long to display the processed image (default: None)
```

#### Examples:

##### Using the `Makefile` and shell scripts:
```sh
# Setup the venv
make venv
source .venv/bin/activate
# Run loop tests (randomly generated inputs according to
# standard countdown rules), 100 times each of anagram and arithmetic,
# with debug output
./scripts/run_loop_test.sh -d -l 100
# Run the OCR tests (example screenshots in ocr-tests/)
# Display the detected text in each image for 3 seconds
./scripts/run_ocr_tests.sh -d -t 3
# Run the full suite in the Makefile:
make all
```

##### Calling `countdown/countdown.py` directly

```sh
# Run an anagram clue (top 10 responses)
./countdown/countdown.py anagram ripnecyus -n 10
#    [['uprisen', 7],
#     ['unspicy', 7],
#     ['uncrisp', 7],
#     ['spicery', 7],
#     ['pyrenic', 7],
#     ['pincers', 7],
#     ['incurse', 7],
#     ['encrisp', 7],
#     ['cyprine', 7],
#     ['puisne', 6]]
./countdown/countdown.py arithmetic 256 100 10 7 6 4 2
#     100 * 10 / 4 + 6
./countdown/countdown.py ocr ocr-test/arithmetic/00025.png --type arithmetic --debug --no-preprocess --greyscale
#     Detected target: 711
#     Detected inputs: [50, 4, 2, 6, 7, 5]
#     Result: 50 * 2 * 7 + 6 + 5
```

### Survey & Analysis

The `survey` subcommand generates random arithmetic puzzles with varied large/small number mixes (0–4 large), exhaustively solves each, and writes the results to a CSV using multiprocessing:

```sh
./countdown/countdown.py survey -n 100000 -o survey_results.csv
```

Two analysis scripts in `scripts/` consume the CSV:

```sh
# Full breakdown by 17 categories (large count, parity, primality, GCD, etc.)
./scripts/analyze_survey.py survey_results.csv

# Rank all categories by predictive power (success rate spread)
./scripts/rank_analyses.py survey_results.csv
```

Both require `pandas` and `sympy`.

### Video

Note that no video examples are included but there is a script to help download them. You will want to get 360P video, which can be tested with:

```sh
ffprobe ocr-test/videos/example.mp4 2>&1 | grep "640x360" -c
```

Requires `ffmpeg` to be installed for the `ffprobe` command.

If you are downloading with `youtube-dl`, use format code `134`.

### Directory layout

```
countdown/
├── countdown/
│   └── countdown.py
├── scripts/
│   ├── analyze_survey.py
│   ├── rank_analyses.py
│   ├── download_cd_eps.sh
│   ├── run_loop_tests.sh
│   └── run_ocr_tests.sh
├── ocr-test/
│   ├── anagrams/
│   ├── arithmetic/
│   └── videos/
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```
