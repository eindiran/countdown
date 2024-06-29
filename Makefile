#==================================================================
#
#           FILE: countdown/Makefile
#
#          USAGE: make all
#
#    DESCRIPTION: Create a virtualenv and run test shell scripts.
#   REQUIREMENTS: python3, venv
#
#==================================================================

# all sets up the venv and runs the tests
.PHONY: all
all: venv test-ocr test-loop

venv: .venv/touchfile

# Build the venv and place a touchfile inside
.venv/touchfile: requirements.txt
	test -d .venv || python3 -m venv .venv
	. .venv/bin/activate; pip install -Ur requirements.txt
	touch .venv/touchfile

# Run the OCR tests inside the venv
.PHONY: test-ocr
test-ocr: venv
	. .venv/bin/activate; ./scripts/run_ocr_tests.sh

# Run the loop tests inside the venv
.PHONY: test-loop
test-loop: venv
	. .venv/bin/activate; ./scripts/run_loop_tests.sh -l 100

# Cleanup the venv
.PHONY: clean
clean:
	rm -rf .venv
	rm -rf .ruff_cache
	rm -rf .mypy_cache
