# Makefile for CompPS Simulations Project

.PHONY: help install setup clean run simulate fit analyze plot test lint format

# Default target
help:
	@echo "CompPS Simulations - Available targets:"
	@echo ""
	@echo "  Setup and Installation:"
	@echo "    make install      - Install dependencies using Poetry"
	@echo "    make setup        - Create directory structure"
	@echo ""
	@echo "  Running Simulations:"
	@echo "    make run          - Run full pipeline (simulate + fit + analyze + plot)"
	@echo "    make simulate     - Only simulate spectra"
	@echo "    make fit          - Only fit existing spectra"
	@echo "    make analyze      - Only analyze existing results"
	@echo "    make plot         - Only generate plots"
	@echo ""
	@echo "  Development:"
	@echo "    make lint         - Run linters on source code"
	@echo "    make format       - Format code"
	@echo "    make test         - Run tests (if implemented)"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean        - Remove generated data and results"
	@echo "    make clean-all    - Remove all generated files including plots"
	@echo ""
	@echo "  Variables (override with ARF=... RMF=...):"
	@echo "    ARF               - ARF filename (default: must be specified)"
	@echo "    RMF               - RMF filename (default: must be specified)"
	@echo "    EXPOSURE          - Exposure time in seconds (default: 10000)"
	@echo "    NORM              - Model normalization (default: 1.0)"
	@echo ""

# Installation and setup
install:
	@echo "Installing dependencies with Poetry..."
	poetry install

setup:
	@echo "Creating directory structure..."
	mkdir -p data/response data/simulated data/results
	@echo "Directory structure created."
	@echo ""
	@echo "IMPORTANT: Place your ARF and RMF files in data/response/"

# Pipeline execution targets
# Variables that can be overridden
ARF ?=
RMF ?=
EXPOSURE ?= 10000
NORM ?= 1.0
SCENARIOS ?= --all

# Check if ARF and RMF are set
check-response:
ifndef ARF
	$(error ARF is not set. Use: make run ARF=yourfile.arf RMF=yourfile.rmf)
endif
ifndef RMF
	$(error RMF is not set. Use: make run ARF=yourfile.arf RMF=yourfile.rmf)
endif

run: check-response
	@echo "Running full pipeline..."
	poetry run python scripts/run_simulation.py \
		--arf $(ARF) \
		--rmf $(RMF) \
		$(SCENARIOS) \
		--exposure $(EXPOSURE) \
		--norm $(NORM)

simulate: check-response
	@echo "Running simulation only..."
	poetry run python scripts/run_simulation.py \
		--arf $(ARF) \
		--rmf $(RMF) \
		$(SCENARIOS) \
		--exposure $(EXPOSURE) \
		--norm $(NORM) \
		--simulate-only

fit:
	@echo "Fitting existing spectra..."
	poetry run python scripts/run_simulation.py --fit-only

analyze:
	@echo "Analyzing existing results..."
	poetry run python scripts/run_simulation.py --analyze-only

plot:
	@echo "Generating plots from existing results..."
	poetry run python scripts/run_simulation.py --analyze-only

# Development targets
lint:
	@echo "Running linters..."
	@if command -v ruff > /dev/null; then \
		ruff check src/ config/ scripts/; \
	elif command -v flake8 > /dev/null; then \
		flake8 src/ config/ scripts/; \
	else \
		echo "No linter found. Install ruff or flake8."; \
	fi

format:
	@echo "Formatting code..."
	@if command -v black > /dev/null; then \
		black src/ config/ scripts/; \
	elif command -v autopep8 > /dev/null; then \
		autopep8 --in-place --recursive src/ config/ scripts/; \
	else \
		echo "No formatter found. Install black or autopep8."; \
	fi

test:
	@echo "Running tests..."
	@echo "No tests implemented yet."

# Cleanup targets
clean:
	@echo "Cleaning generated data..."
	rm -rf data/simulated/*.pha
	rm -rf data/simulated/*.json
	rm -rf data/results/*.json
	@echo "Cleaned simulated spectra and fit results."

clean-plots:
	@echo "Cleaning plots..."
	rm -rf data/results/*.png
	@echo "Cleaned plots."

clean-all: clean clean-plots
	@echo "All generated files removed."

# Example targets for specific use cases
example-basic: check-response
	@echo "Running basic thermal scenario..."
	poetry run python scripts/run_simulation.py \
		--arf $(ARF) \
		--rmf $(RMF) \
		--scenarios basic_thermal \
		--exposure $(EXPOSURE) \
		--norm $(NORM)

example-all: check-response
	@echo "Running all scenarios..."
	poetry run python scripts/run_simulation.py \
		--arf $(ARF) \
		--rmf $(RMF) \
		--all \
		--exposure $(EXPOSURE) \
		--norm $(NORM)

# Show project status
status:
	@echo "Project Status:"
	@echo ""
	@echo "Response files (data/response/):"
	@ls -lh data/response/ 2>/dev/null || echo "  No files found"
	@echo ""
	@echo "Simulated spectra (data/simulated/):"
	@find data/simulated/ -name "*.pha" 2>/dev/null | wc -l | xargs echo "  Number of spectra:"
	@echo ""
	@echo "Fit results (data/results/):"
	@find data/results/ -name "fit_*.json" 2>/dev/null | wc -l | xargs echo "  Number of fits:"
	@echo ""
	@echo "Plots (data/results/):"
	@find data/results/ -name "*.png" 2>/dev/null | wc -l | xargs echo "  Number of plots:"

