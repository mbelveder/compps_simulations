# Makefile for CompPS Simulations Project

.PHONY: help install setup clean run simulate fit analyze plot plot-grid test lint format tau-kTe-study rel-refl-study

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
	@echo "    make plot-grid    - Generate XCM spectra grid plot"
	@echo "    make ktbb-study   - Run kTbb variation study"
	@echo "    make tau-kTe-study - Run tau_y vs kTe parameter grid study"
	@echo "    make rel-refl-study - Run rel_refl parameter grid study"
	@echo "    make doc-params   - Generate parameter documentation (Markdown)"
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
	@echo "    ENERGY_MIN        - Minimum fit energy in keV (default: 0.2)"
	@echo "    ENERGY_MAX        - Maximum fit energy in keV (default: 10.0)"
	@echo "    STAT_METHOD       - Fit statistic: chi/cstat/pgstat (default: cstat)"
	@echo "    KTBB_VALUES       - kTbb values for study (default: 0.1,0.3,0.5,0.8)"
	@echo "                        Example: make ktbb-study KTBB_VALUES=0.01,0.05,0.1"
	@echo "    TAU_VALUES        - tau_y values for tau-kTe/rel-refl studies (default: -0.1,-0.2,-0.5,-1.0,-1.5,-2)"
	@echo "    KTE_VALUES        - kTe values for tau-kTe/rel-refl studies (default: 50,100,150)"
	@echo "    SCENARIO          - Base scenario for tau-kTe/rel-refl studies (default: typical_agn_slab)"
    @echo "    REL_REFL_VALUES   - rel_refl values for rel-refl study (default: 0,0.1,0.3,0.5,1,2,5,10,50,100,-1)"
    @echo "    VERBOSE           - Enable verbose logging for studies (default: 0, set to 1 to enable)"
    @echo "    PLOT_MODELS       - Generate CompPS model plots for tau-kTe study (default: 0, set to 1 to enable)"
    @echo "    SHOW_ERROR_BARS   - Show error bars on data points in plots (default: 0, set to 1 to enable)"
	@echo ""
	@echo "  Notes:"
	@echo "    - Spectra are automatically grouped with ftgrouppha (groupscale=3)"
	@echo "    - Bad channels are automatically ignored during fitting"
	@echo "    - C-statistic (cstat) is used by default for low-count data"
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
# Helper for comma substitution in arguments
comma := ,

ARF ?= src_5359_020_ARF_00001.fits.gz
RMF ?= src_5359_020_RMF_00001.fits.gz
EXPOSURE ?= 1000000
NORM ?= 1.0
SCENARIOS ?= --all
ENERGY_MIN ?= 0.3
ENERGY_MAX ?= 10.0
STAT_METHOD ?= cstat
KTBB_VALUES ?= 0.1,0.3,0.5,0.8

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
		--norm $(NORM) \
		--energy-min $(ENERGY_MIN) \
		--energy-max $(ENERGY_MAX) \
		--stat-method $(STAT_METHOD)

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
	poetry run python scripts/run_simulation.py \
		--fit-only \
		--energy-min $(ENERGY_MIN) \
		--energy-max $(ENERGY_MAX) \
		--stat-method $(STAT_METHOD)

analyze:
	@echo "Analyzing existing results..."
	poetry run python scripts/run_simulation.py --analyze-only

plot:
	@echo "Generating plots from existing results..."
	poetry run python scripts/run_simulation.py --analyze-only

plot-grid:
	@echo "Generating XCM spectra grid plot..."
	poetry run python scripts/plot_xcm_grid.py

ktbb-study: check-response
	@echo "Running kTbb variation study..."
	poetry run python scripts/run_ktbb_study.py \
		--arf $(ARF) \
		--rmf $(RMF) \
		--exposure $(EXPOSURE) \
		--norm $(NORM) \
		--energy-min $(ENERGY_MIN) \
		--energy-max $(ENERGY_MAX) \
		--stat-method $(STAT_METHOD) \
		--ktbb-values $(KTBB_VALUES)

# Variables for tau-kTe study
# TAU_VALUES ?= 0.2,0.5,1.0,1.5,2,2.5,3,3.5
TAU_VALUES ?= -0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1,-1.1,-1.2,-1.3,-1.5
KTE_VALUES ?= 50,100,150
SCENARIO ?= typical_agn_slab
VERBOSE ?= 0
PLOT_MODELS ?= 0
SHOW_ERROR_BARS ?= 0

tau-kTe-study: check-response
	@echo "Running tau_y vs kTe parameter grid study..."
	poetry run python scripts/run_tau_kTe_study.py \
		--scenario $(SCENARIO) \
		--arf $(ARF) \
		--rmf $(RMF) \
		--tau-values $(subst $(comma), ,$(TAU_VALUES)) \
		--kTe-values $(subst $(comma), ,$(KTE_VALUES)) \
		--exposure $(EXPOSURE) \
		--normalization $(NORM) \
		--energy-range $(ENERGY_MIN) $(ENERGY_MAX) \
		$(if $(filter 1,$(VERBOSE)),-v,) \
		$(if $(filter 1,$(PLOT_MODELS)),--plot-models,) \
		$(if $(filter 1,$(SHOW_ERROR_BARS)),--show-error-bars,)

# Variables for rel-refl study
REL_REFL_VALUES ?= 0,0.1,0.3,0.5,1,2,5,10,50,100,-1

rel-refl-study: check-response
	@echo "Running rel_refl parameter grid study..."
	poetry run python scripts/run_rel_refl_study.py \
		--scenario $(SCENARIO) \
		--arf $(ARF) \
		--rmf $(RMF) \
		--rel-refl-values $(subst $(comma), ,$(REL_REFL_VALUES)) \
		--tau-values $(subst $(comma), ,$(TAU_VALUES)) \
		--kTe-values $(subst $(comma), ,$(KTE_VALUES)) \
		--exposure $(EXPOSURE) \
		--normalization $(NORM) \
		--energy-range $(ENERGY_MIN) $(ENERGY_MAX) \
		$(if $(filter 1,$(VERBOSE)),-v,) \
		$(if $(filter 1,$(SHOW_ERROR_BARS)),--show-error-bars,)

doc-params:
	@echo "Generating parameter documentation..."
	poetry run python scripts/generate_parameter_doc.py

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
	rm -rf data/results/*.xcm
	@echo "Cleaned simulated spectra and fit results."

clean-plots:
	@echo "Cleaning plots..."
	rm -rf data/results/*.png
	@echo "Cleaned plots."

clean-all: clean clean-plots
	@echo "All generated files removed."

clean-run: clean-all run
	@echo "All generated files removed and the whole pipline runned."

# Example targets for specific use cases
example-basic: check-response
	@echo "Running basic thermal scenario..."
	poetry run python scripts/run_simulation.py \
		--arf $(ARF) \
		--rmf $(RMF) \
		--scenarios basic_thermal \
		--exposure $(EXPOSURE) \
		--norm $(NORM) \
		--energy-min $(ENERGY_MIN) \
		--energy-max $(ENERGY_MAX) \
		--stat-method $(STAT_METHOD)

example-all: check-response
	@echo "Running all scenarios..."
	poetry run python scripts/run_simulation.py \
		--arf $(ARF) \
		--rmf $(RMF) \
		--all \
		--exposure $(EXPOSURE) \
		--norm $(NORM) \
		--energy-min $(ENERGY_MIN) \
		--energy-max $(ENERGY_MAX) \
		--stat-method $(STAT_METHOD)

# Show project status
status:
	@echo "Project Status:"
	@echo ""
	@echo "Response files (data/response/):"
	@ls -lh data/response/ 2>/dev/null || echo "  No files found"
	@echo ""
	@echo "Simulated spectra (data/simulated/):"
	@find data/simulated/ -name "*.pha" 2>/dev/null | wc -l | xargs echo "  Total spectra:"
	@find data/simulated/ -name "*_grp.pha" 2>/dev/null | wc -l | xargs echo "  Grouped spectra:"
	@echo ""
	@echo "Fit results (data/results/):"
	@find data/results/ -name "fit_*.json" 2>/dev/null | wc -l | xargs echo "  Number of fits:"
	@find data/results/ -name "fit_*.xcm" 2>/dev/null | wc -l | xargs echo "  XSPEC sessions (.xcm):"
	@echo ""
	@echo "Plots (data/results/):"
	@find data/results/ -name "*.png" 2>/dev/null | wc -l | xargs echo "  Number of plots:"

