# Interpreter used for every target. The project interpreter is Isaac Sim's bundled Python
# (see docs/decisions/0002-dependency-floors-and-interpreter.md), e.g.
#   make check PYTHON=/home/hunter/IsaacSim/_build/linux-x86_64/release/python.sh
# Any CPython >= 3.10 with the dev extras installed also works for the engine-free core.
PYTHON ?= python

# `make ci`: the GPU-free gate on a plain CPython venv, the same job .github/workflows/check.yml
# runs. Proves the engine-free core needs neither Isaac Sim nor CUDA.
CI_PYTHON ?= python3.10
CI_VENV ?= .venv-ci

.PHONY: install test test-slow test-all test-affected test-full lint fmt typecheck check check-full ci luts \
        golden-update clean next stage \
        site site-preview site-publish viewer

install:
	$(PYTHON) -m pip install -e ".[dev]"

# Every pytest target runs on all cores (pytest-xdist, ADR 0193). `loadfile` keeps each module's
# tests in one worker, in file order, as a serial run has them: some modules share a module-level
# RNG or a module-scoped scene that each test steps forward, and splitting them changes results.
# One BLAS thread per worker, or 32 workers x 32 BLAS threads thrash the machine.
SUITE   = tests/unit tests/golden
XDIST   = -n auto --dist loadfile
TESTENV = OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTEST  = $(TESTENV) $(PYTHON) -m pytest

# The fast tier: unit tests only. `slow` marks the validation benches (Tier 2/3/4 phenomenology),
# the end-to-end frame benches and the file-regeneration checks, plus any single test over a
# second. This is the developer loop; it is NOT the commit gate (see `check` below).
test:
	$(PYTEST) $(SUITE) -q -m "not slow and not isaac and not gpu" $(XDIST) --durations=15

# The other half. Same suite, same machine.
test-slow:
	$(PYTEST) $(SUITE) -q -m "slow and not isaac and not gpu" $(XDIST) --durations=15

# Both tiers, every test, and a fresh record for `test-affected` (testmon's database and
# .testreads.json). What CI runs, and what to run before a push.
test-full:
	$(PYTEST) $(SUITE) -q -m "not isaac and not gpu" $(XDIST) --durations=15 \
	    --testmon-noselect --track-reads --track-reads-reset

# The commit gate's tests (ADR 0193): only the tests a change can affect, from two records --
#   1. pytest-testmon: the tests whose Python code (source or test) changed since they passed;
#   2. scripts/affected_tests.py: the test modules that read a data file (YAML, CSV, .npy, .md)
#      that changed, from .testreads.json (tests/read_tracker.py).
# With no record yet, or when something every test depends on changed (pyproject.toml, a file
# read at import), it runs the full suite instead. Exit 5 is "testmon selected nothing": a pass.
# `--testmon-forceselect`, not `--testmon`: plain testmon switches its selection off when -m is
# given, and pyproject addopts always gives one.
test-affected:
	@sel="$$($(PYTHON) scripts/affected_tests.py --why)" || exit $$?; \
	if [ "$$sel" = "ALL" ]; then \
	    echo "test-affected: no usable record or a global input changed -> full suite"; \
	    $(MAKE) --no-print-directory test-full; \
	else \
	    $(PYTEST) $(SUITE) -q -m "not isaac and not gpu" $(XDIST) --testmon-forceselect --track-reads; \
	    rc=$$?; [ $$rc -eq 5 ] && rc=0; \
	    if [ -n "$$sel" ]; then \
	        echo "test-affected: data changed under $$(echo "$$sel" | wc -l) module(s)"; \
	        $(PYTEST) $$sel -q -m "not isaac and not gpu" $(XDIST) --testmon-noselect --track-reads; \
	        rc2=$$?; [ $$rc2 -eq 5 ] && rc2=0; [ $$rc -ne 0 ] || rc=$$rc2; \
	    fi; \
	    exit $$rc; \
	fi

test-all:
	$(PYTHON) -m pytest tests -q -m ""

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

fmt:
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

typecheck:
	$(PYTHON) -m mypy src/irsim src/irsim_isaac src/irsim_eval src/irsim_viewer

# What to start now, and republish the queue the roadmap shows. `make next` regenerates it;
# `make check` only verifies it, so a stale queue fails the gate instead of misleading a reader.
next:
	$(PYTHON) scripts/next_step.py
	@$(PYTHON) scripts/next_step.py --write

# Stage your own edit to a shared file (TECHNICAL_REPORT.md, README.md, CHANGELOG.md, docs/roadmap.md) without pulling
# in another session's in-flight prose (RP.3). Snapshot BEFORE editing:
#   scripts/stage_own_hunk.sh snapshot README.md CHANGELOG.md docs/roadmap.md
# then, before committing:
#   make stage FILES="README.md CHANGELOG.md docs/roadmap.md"
stage:
	@[ -n "$(FILES)" ] || { echo "usage: make stage FILES=\"README.md CHANGELOG.md ...\""; exit 2; }
	scripts/stage_own_hunk.sh stage $(FILES)
	scripts/stage_own_hunk.sh check $(FILES)

# The commit gate (ADR 0193): lint, types, and every test the change can affect.
check: lint typecheck test-affected
	@$(PYTHON) scripts/next_step.py --check
	@echo "OK — safe to commit"

# The push/CI gate: the same with every test. Run before pushing, after a pull or a rebase, and
# whenever you doubt the selection.
check-full: lint typecheck test-full
	@$(PYTHON) scripts/next_step.py --check
	@echo "OK — safe to push"

ci:
	$(CI_PYTHON) -m venv $(CI_VENV)
	$(CI_VENV)/bin/python -m pip install -q --upgrade pip
	$(CI_VENV)/bin/python -m pip install -q -e ".[dev]"
	$(MAKE) check-full PYTHON=$(CI_VENV)/bin/python

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

# Click on rendered frames in a browser and read every saved plane under the pixel -- true surface
# temperature, part, range, radiance, apparent temperature, ADC code (src/irsim_viewer). RUN is a
# render's output folder or a folder of them. `scripts/viewer.sh` finds a working interpreter on
# its own, so plain `make viewer` works without PYTHON=. Guide: docs/frame-viewer.md.
RUN ?= outputs
viewer:
	@PYTHON="$(filter-out python,$(PYTHON))" scripts/viewer.sh $(RUN)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build $(CI_VENV) \
	    .testmondata .testmondata-shm .testmondata-wal .testreads.json .testreads-shards
	find . -name __pycache__ -type d -exec rm -rf {} +
