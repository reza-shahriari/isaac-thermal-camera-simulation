# Interpreter used for every target. The project interpreter is Isaac Sim's bundled Python
# (see docs/decisions/0002-dependency-floors-and-interpreter.md), e.g.
#   make check PYTHON=/home/hunter/IsaacSim/_build/linux-x86_64/release/python.sh
# Any CPython >= 3.10 with the dev extras installed also works for the engine-free core.
PYTHON ?= python

# `make ci`: the GPU-free gate on a plain CPython venv, the same job .github/workflows/check.yml
# runs. Proves the engine-free core needs neither Isaac Sim nor CUDA.
CI_PYTHON ?= python3.10
CI_VENV ?= .venv-ci

.PHONY: install test test-slow test-all lint fmt typecheck check ci luts golden-update clean next stage \
        site site-preview site-publish

install:
	$(PYTHON) -m pip install -e ".[dev]"

# The fast tier: unit tests only. `slow` marks the validation benches (Tier 2/3/4 phenomenology),
# the end-to-end frame benches and the file-regeneration checks, plus any single test over a
# second. This is the developer loop; it is NOT the commit gate -- `make check` runs both tiers,
# so nothing escapes review by being slow (GT.1).
test:
	$(PYTHON) -m pytest tests/unit tests/golden -q -m "not slow" --durations=15

# The other half. Same suite, same machine, run by `make check` and by CI.
test-slow:
	$(PYTHON) -m pytest tests/unit tests/golden -q -m "slow" --durations=15

test-all:
	$(PYTHON) -m pytest tests -q -m ""

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

fmt:
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

typecheck:
	$(PYTHON) -m mypy src/irsim src/irsim_isaac src/irsim_eval

# What to start now, and republish the queue the roadmap shows. `make next` regenerates it;
# `make check` only verifies it, so a stale queue fails the gate instead of misleading a reader.
next:
	$(PYTHON) scripts/next_step.py
	@$(PYTHON) scripts/next_step.py --write

# Stage your own edit to a shared file (README.md, CHANGELOG.md, docs/roadmap.md) without pulling
# in another session's in-flight prose (RP.3). Snapshot BEFORE editing:
#   scripts/stage_own_hunk.sh snapshot README.md CHANGELOG.md docs/roadmap.md
# then, before committing:
#   make stage FILES="README.md CHANGELOG.md docs/roadmap.md"
stage:
	@[ -n "$(FILES)" ] || { echo "usage: make stage FILES=\"README.md CHANGELOG.md ...\""; exit 2; }
	scripts/stage_own_hunk.sh stage $(FILES)
	scripts/stage_own_hunk.sh check $(FILES)

check: lint typecheck test test-slow
	@$(PYTHON) scripts/next_step.py --check
	@echo "OK — safe to commit"

ci:
	$(CI_PYTHON) -m venv $(CI_VENV)
	$(CI_VENV)/bin/python -m pip install -q --upgrade pip
	$(CI_VENV)/bin/python -m pip install -q -e ".[dev]"
	$(MAKE) check PYTHON=$(CI_VENV)/bin/python

luts:
	$(PYTHON) scripts/generate_luts.py --configs configs/sensors --out data/lut --data data

golden-update:
	$(PYTHON) -m pytest tests/golden -q --update-golden

# The project site: every document, the module and config catalogues measured from the tree, and
# a web-sized copy of whatever renders are in outputs/ (scripts/build_site.py). `site` builds it
# into the gitignored _site/; `site-preview` serves it; `site-publish` commits it onto the
# gh-pages branch and tells you the push command.
site:
	$(PYTHON) scripts/build_site.py --out _site

site-preview:
	$(PYTHON) scripts/build_site.py --out _site --serve

site-publish: site
	scripts/publish_site.sh

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build $(CI_VENV)
	find . -name __pycache__ -type d -exec rm -rf {} +
