all: venv test-ocr test-loop

venv: venv/touchfile

venv/touchfile: requirements.txt
	test -d venv || python3 -m venv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/touchfile

.PHONY: test-ocr
test-ocr: venv
	. venv/bin/activate; ./scripts/run_ocr_tests.sh

.PHONY: test-loop
test-loop: venv
	. venv/bin/activate; ./scripts/run_loop_tests.sh -l 100

.PHONY: clean
clean:
	rm -rf venv
