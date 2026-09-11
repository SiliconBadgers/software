.PHONY: setup doctor test
PYTHON ?= python3
setup:
	$(PYTHON) -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ required'; print('PASS setup: standard-library starter, no packages required')"
CONTRACT ?= ../architecture/contracts/mac-v0.json
OUT ?= build/mac-vectors.txt
.PHONY: vectors
doctor: setup
test:
	$(PYTHON) -m unittest discover -s tests -v
vectors:
	$(PYTHON) generate_vectors.py --contract "$(CONTRACT)" --output "$(OUT)"
