# CompPS Simulations

A Python package to simulate X-ray spectra using the CompPS (Comptonization, Poutanen & Svensson) model from XSPEC, and fit them with an absorbed powerlaw model to analyze the relationship between Comptonization parameters and fitted spectral indices.

## Overview

This project provides a complete pipeline for:

1. **Simulating** X-ray spectra using the CompPS model with various parameter combinations
2. **Fitting** the simulated spectra using an absorbed powerlaw model (`tbabs*powerlaw`)
3. **Analyzing** the relationship between input CompPS parameters and fitted powerlaw parameters
4. **Visualizing** results through comprehensive plots and statistical summaries

The CompPS model describes Comptonization in different geometries (slab, sphere, cylinder, hemisphere) with various electron distributions and seed photon sources. This project helps investigate whether different Comptonization scenarios can produce spectra that are well-fit by simple powerlaw models, and what photon indices result.

## Installation

### Prerequisites

- Python 3.11 or higher
- [Poetry](https://python-poetry.org/) for dependency management
- [HEASOFT](https://heasarc.gsfc.nasa.gov/lheasoft/) with XSPEC installed
- PyXspec (Python bindings for XSPEC)

### Setup

1. Clone or navigate to this repository:

```bash
cd compps_simulations
```

2. Install dependencies using Poetry:

```bash
make install
# or
poetry install
```

3. Create the directory structure:

```bash
make setup
```

4. Place your ARF and RMF response files in `data/response/`:

```bash
cp /path/to/your/instrument.arf data/response/
cp /path/to/your/instrument.rmf data/response/
```

## Project Structure

```
compps_simulations/
├── config/
│   ├── __init__.py
│   └── parameters.py          # Define CompPS parameter sets here
├── src/
│   ├── __init__.py
│   ├── xspec_interface.py     # PyXspec wrapper functions
│   ├── simulator.py           # Spectrum generation
│   ├── fitter.py              # Absorbed powerlaw fitting
│   ├── analysis.py            # Result analysis
│   └── plotting.py            # Visualization
├── data/
│   ├── response/              # Place ARF and RMF files here
│   ├── simulated/             # Generated spectra (auto-created)
│   └── results/               # Fit results and plots (auto-created)
├── scripts/
│   └── run_simulation.py      # Main execution script
├── Makefile                   # Convenience commands
├── pyproject.toml             # Poetry dependencies
└── README.md                  # This file
```

## Usage

### Quick Start

Run the complete pipeline with all predefined scenarios:

```bash
make run ARF=instrument.arf RMF=instrument.rmf
```

This will:
1. Simulate spectra for all scenarios in `config/parameters.py`
2. Group spectra using `ftgrouppha` (min grouping with groupscale=3)
3. Fit each grouped spectrum with `tbabs*powerlaw`
4. Analyze results and generate statistics
5. Create summary plots

**Note:** The pipeline automatically groups spectra after generation using `ftgrouppha` from HEASOFT. 
If `ftgrouppha` is not available, ungrouped spectra will be used for fitting.

### Defining Parameter Sets

Edit `config/parameters.py` to define your own CompPS parameter combinations. Example:

```python
COMPPS_PARAMS = {
    'my_scenario': {
        'kTe': 50.0,         # Electron temperature (keV) [20, 100000]
        'EleIndex': 2.0,     # Power-law index [0, 5]
        'Gmin': -1.0,        # Min Lorentz factor [-1, 10] (-1 for Maxwellian)
        'Gmax': 10.0,        # Max Lorentz factor [10, 10000]
        'kTbb': 0.5,         # Seed photon temperature (keV) [0.001, 10]
        'tau_y': 1.0,        # Optical depth [0.05, 3]
        'geom': 1.0,         # Geometry [−5, 4] (1=slab, 4=sphere)
        'HovR_cyl': 1.0,     # Height/radius (for cylinder) [0.5, 2]
        'cosIncl': 0.5,      # Cosine of inclination [0.05, 0.95]
        'cov_frac': 1.0,     # Covering factor [0, 1]
        'rel_refl': 0.0,     # Reflection amount [0, 10000]
        'Fe_ab_re': 1.0,     # Iron abundance [0.1, 10]
        'Me_ab': 1.0,        # Metal abundance [0.1, 10]
        'xi': 0.0,           # Ionization parameter [0, 100000]
        'Tdisk': 30000.0,    # Disk temperature (K) [10000, 1e6]
        'Betor10': -10.0,    # Emissivity law [-10, 10]
        'Rin': 6.01,         # Inner disk radius (Rg) [6.001, 10000]
        'Rout': 100.0,       # Outer disk radius (Rg) [0, 1e6]
        'Redshift': 0.0,     # Redshift [-0.999, 10]
    },
}
```

### Using the Makefile

The Makefile provides convenient shortcuts:

```bash
# Full pipeline
make run ARF=nustar.arf RMF=nustar.rmf

# Run specific scenarios
make run ARF=nustar.arf RMF=nustar.rmf SCENARIOS="--scenarios basic_thermal hot_with_reflection"

# Custom exposure time and normalization
make run ARF=nustar.arf RMF=nustar.rmf EXPOSURE=20000 NORM=0.5

# Run only specific steps
make simulate ARF=nustar.arf RMF=nustar.rmf  # Only simulate
make fit                                      # Only fit existing spectra
make analyze                                  # Only analyze existing results

# Cleanup
make clean        # Remove simulated spectra and fit results
make clean-all    # Remove everything including plots

# Project status
make status       # Show number of files in each directory
```

### Using Python Directly

For more control, use the Python script directly:

```bash
# Full pipeline
poetry run python scripts/run_simulation.py \
    --arf nustar.arf \
    --rmf nustar.rmf \
    --all \
    --exposure 10000 \
    --norm 1.0

# Specific scenarios
poetry run python scripts/run_simulation.py \
    --arf nustar.arf \
    --rmf nustar.rmf \
    --scenarios basic_thermal hot_with_reflection

# Only simulate
poetry run python scripts/run_simulation.py \
    --arf nustar.arf \
    --rmf nustar.rmf \
    --all \
    --simulate-only

# Only fit existing spectra
poetry run python scripts/run_simulation.py --fit-only

# Only analyze (skip plotting)
poetry run python scripts/run_simulation.py --analyze-only --no-plots

# Custom energy range for fitting
poetry run python scripts/run_simulation.py \
    --arf nustar.arf \
    --rmf nustar.rmf \
    --all \
    --energy-min 3.0 \
    --energy-max 79.0
```

### Output

Results are saved in `data/results/`:

- **Individual fit results**: `fit_sim_<scenario>_<timestamp>.json`
- **Fit log**: `fit_log.json` (all fits)
- **Combined results**: `combined_results.csv`
- **Parameter comparison**: `parameter_comparison.csv`
- **Analysis report**: `analysis_report.txt`
- **Plots**:
  - `summary_photon_index_dist.png` - Distribution of fitted photon indices
  - `summary_Te_vs_photon_index.png` - Temperature vs photon index
  - `summary_tau_vs_photon_index.png` - Optical depth vs photon index
  - `summary_chi_squared.png` - Distribution of fit quality

## CompPS Model Parameters

The CompPS model has 19 parameters (see [XSPEC documentation](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node156.html)):

| Parameter | Description |
|-----------|-------------|
| `Te` | Electron temperature (keV) |
| `p` | Electron power-law index |
| `gmin` | Minimum Lorentz factor |
| `gmax` | Maximum Lorentz factor |
| `Tbb` | Seed photon temperature (keV, negative for disk) |
| `tau` | Optical depth (or y-parameter if negative) |
| `geom` | Geometry (1=slab, 2=cylinder, 3=hemisphere, 4=sphere) |
| `H_R` | Height-to-radius ratio (cylinder only) |
| `cosIncl` | Cosine of inclination angle |
| `cov_fac` | Covering factor |
| `R` | Reflection amount (Ω/2π) |
| `FeAb` | Iron abundance (solar units) |
| `MeAb` | Heavy element abundance (solar units) |
| `xi` | Disk ionization parameter |
| `temp` | Disk temperature (K) |
| `beta` | Reflection emissivity law |
| `Rin` | Inner disk radius (Rg) |
| `Rout` | Outer disk radius (Rg) |
| `redshift` | Redshift |

## Example Workflows

### Investigate Temperature Dependence

```bash
# Edit config/parameters.py to create scenarios with varying Te
# Then run:
make run ARF=instrument.arf RMF=instrument.rmf

# Check the temperature vs photon index plot:
open data/results/summary_Te_vs_photon_index.png
```

### Study Different Geometries

Create scenarios in `config/parameters.py` with different `geom` values:
- `geom=1.0`: slab
- `geom=4.0`: sphere

```bash
make run ARF=instrument.arf RMF=instrument.rmf
cat data/results/analysis_report.txt
```

### Parameter Sweep

Create a grid of parameters programmatically:

```python
# In config/parameters.py
import numpy as np

COMPPS_PARAMS = {}
for te in [30, 50, 75, 100, 150]:
    for tau in [0.3, 0.5, 1.0, 1.5]:
        scenario_name = f"Te{te}_tau{tau}"
        COMPPS_PARAMS[scenario_name] = {
            'kTe': float(te),
            'tau_y': float(tau),
            'kTbb': 0.5,
            # ... other parameters ...
        }
```

## Development

### Code Quality

```bash
# Format code
make format

# Run linters
make lint
```

### Adding New Features

The modular structure makes it easy to extend:

- **New models**: Add to `src/xspec_interface.py`
- **New analysis**: Add functions to `src/analysis.py`
- **New plots**: Add to `src/plotting.py`

## Troubleshooting

### PyXspec Import Error

If you get `ImportError: No module named xspec`:

1. Ensure HEASOFT is installed and initialized:
   ```bash
   source /path/to/heasoft/headas-init.sh
   ```

2. PyXspec should be available after HEASOFT initialization

### Response Files Not Found

Place ARF and RMF files in `data/response/` and use only the filename (not full path) in commands:

```bash
# Correct
make run ARF=nustar.arf RMF=nustar.rmf

# Incorrect
make run ARF=/full/path/nustar.arf RMF=/full/path/nustar.rmf
```

### Fit Failures

If fits fail, check:
- Energy range is appropriate for your detector
- Initial parameter values in `src/fitter.py`
- Spectrum has sufficient counts

## References

- [CompPS Model Documentation](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node156.html)
- Poutanen & Svensson (1996, ApJ 470, 249) - Original CompPS paper
- [XSPEC Manual](https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/manual.html)

## License

This project is intended for scientific research purposes.

## Author

Michael Belvedersky (mike.belveder@gmail.com)

## Citation

If you use this code in your research, please cite the original CompPS paper:
Poutanen, J., & Svensson, R. 1996, ApJ, 470, 249

