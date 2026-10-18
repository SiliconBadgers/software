.PHONY: setup doctor test vectors
UV ?= uv
PYTHON ?= 3.12
UV_RUN = $(UV) run --locked --no-default-groups --python "$(PYTHON)"
CONTRACT ?= ../architecture/contracts/mac-v0.json
OUT ?= build/mac-vectors.txt

setup:
	$(UV) sync --locked --no-default-groups --python "$(PYTHON)"
	$(UV_RUN) python -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ required'; print('PASS setup: standard-library starter, no packages required')"
doctor: setup
test:
	$(UV_RUN) python -m unittest discover -s tests -v
vectors:
	$(UV_RUN) python generate_vectors.py --contract "$(CONTRACT)" --output "$(OUT)"
