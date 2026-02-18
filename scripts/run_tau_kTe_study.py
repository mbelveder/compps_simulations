#!/usr/bin/env python3
"""
Parameter grid study: tau_y vs kTe effect on photon index.

This script:
1. Takes a base scenario from parameters.py
2. Generates simulations with varying tau_y and kTe
3. Fits each simulation with absorbed powerlaw
4. Plots photon index vs tau_y for different kTe values

Each run creates a dedicated timestamped directory to avoid mixing
results from different runs.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parameters import COMPPS_PARAMS
from src.grid_study_common import (
    setup_logging,
    run_simulations,
    run_fits,
    extract_results_for_plotting,
    plot_results,
    save_results_csv,
    print_results_table,
    format_tau_for_filename,
    validate_tau_values
)
from src.plotting import plot_compps_models_by_kte


def create_run_directory(base_output_dir: str, scenario_name: str,
                         timestamp: str, logger) -> Path:
    """
    Create a dedicated directory for this run.

    Parameters
    ----------
    base_output_dir : str
        Base output directory (e.g., 'data/results')
    scenario_name : str
        Base scenario name
    timestamp : str
        Timestamp string for this run
    logger : logging.Logger
        Logger instance

    Returns
    -------
    Path
        Path to the run directory
    """
    study_name = f"tau_kTe_study_{scenario_name}_{timestamp}"
    run_dir = Path(base_output_dir) / study_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (run_dir / "spectra").mkdir(exist_ok=True)
    (run_dir / "fits").mkdir(exist_ok=True)

    logger.info(f"Created run directory: {run_dir}")
    logger.info(f"  Spectra will be saved to: {run_dir / 'spectra'}")
    logger.info(f"  Fit results will be saved to: {run_dir / 'fits'}")

    return run_dir


def save_run_metadata(run_dir: Path, args, base_params: dict,
                      grid_params: dict, timestamp: str, logger):
    """
    Save run metadata to JSON file.

    Parameters
    ----------
    run_dir : Path
        Run directory
    args : argparse.Namespace
        Command line arguments
    base_params : dict
        Base scenario parameters
    grid_params : dict
        Grid parameters
    timestamp : str
        Run timestamp
    logger : logging.Logger
        Logger instance
    """
    metadata = {
        'run_timestamp': timestamp,
        'base_scenario': args.scenario,
        'tau_values': args.tau_values,
        'kTe_values': args.kTe_values,
        'exposure': args.exposure,
        'normalization': args.normalization,
        'energy_range': args.energy_range,
        'arf_file': args.arf,
        'rmf_file': args.rmf,
        'base_parameters': base_params,
        'grid_scenarios': list(grid_params.keys()),
        'total_combinations': len(grid_params)
    }

    metadata_file = run_dir / "run_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  Run metadata saved to: {metadata_file}")


def parse_tau_from_filename(tau_str: str) -> float:
    """
    Parse tau value from filename format back to float.

    Handles 'm' prefix for negative values.
    E.g., "m0.50" → -0.50, "0.50" → 0.50

    Parameters
    ----------
    tau_str : str
        Tau string from filename (e.g., "m0.50" or "0.50")

    Returns
    -------
    float
        Parsed tau value
    """
    if tau_str.startswith('m'):
        return -float(tau_str[1:])
    else:
        return float(tau_str)


def create_parameter_grid(base_params, tau_values, kTe_values, logger):
    """
    Create parameter grid by varying tau_y and kTe.

    Parameters
    ----------
    base_params : dict
        Base parameter set
    tau_values : list
        List of tau_y values to test
    kTe_values : list
        List of kTe values to test
    logger : logging.Logger
        Logger instance

    Returns
    -------
    dict
        Dictionary of {scenario_name: params}
    """
    logger.info("Creating parameter grid...")
    logger.info("  Base parameters will be modified for tau_y and kTe")
    kTbb = base_params['kTbb']
    geom = base_params['geom']
    cosIncl = base_params['cosIncl']
    logger.debug(f"  Base params: kTbb={kTbb}, geom={geom}, cosIncl={cosIncl}")

    grid_params = {}

    for kTe in kTe_values:
        for tau in tau_values:
            # Handle negative tau values in scenario name
            tau_str = format_tau_for_filename(tau)
            scenario_name = f"grid_kTe{kTe:.0f}_tau{tau_str}"
            params = base_params.copy()
            params['kTe'] = kTe
            params['tau_y'] = tau
            grid_params[scenario_name] = params
            logger.debug(f"  Created scenario: {scenario_name}")

    logger.info(f"  Total grid points: {len(grid_params)}")
    grid_structure = f"{len(kTe_values)} kTe × {len(tau_values)} tau_y"
    logger.info(f"  Grid structure: {grid_structure}")

    return grid_params


def main():
    parser = argparse.ArgumentParser(
        description='Parameter grid study: tau_y vs kTe effect on photon index'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        required=True,
        choices=list(COMPPS_PARAMS.keys()),
        help='Base scenario from parameters.py'
    )
    parser.add_argument(
        '--arf',
        type=str,
        default='src_5359_020_ARF_00001.fits.gz',
        help='ARF filename in data/response/'
    )
    parser.add_argument(
        '--rmf',
        type=str,
        default='src_5359_020_RMF_00001.fits.gz',
        help='RMF filename in data/response/'
    )
    parser.add_argument(
        '--tau-values',
        type=float,
        nargs='+',
        default=[0.2, 0.5, 1.0, 1.5],
        help='List of tau_y values to test (default: 0.2 0.5 1.0 1.5)'
    )
    parser.add_argument(
        '--kTe-values',
        type=float,
        nargs='+',
        default=[50.0, 100.0, 150.0],
        help='List of kTe values to test (default: 50 100 150)'
    )
    parser.add_argument(
        '--exposure',
        type=float,
        default=100000.0,
        help='Exposure time in seconds (default: 100000)'
    )
    parser.add_argument(
        '--normalization',
        type=float,
        default=1.0,
        help='Model normalization (default: 1.0)'
    )
    parser.add_argument(
        '--energy-range',
        type=float,
        nargs=2,
        default=[0.3, 10.0],
        help='Energy range for fitting in keV (default: 0.3 10.0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/results',
        help='Base output directory (default: data/results)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )
    parser.add_argument(
        '--plot-models',
        action='store_true',
        help='Generate model plots from .xcm files (grouped by kTe)'
    )
    parser.add_argument(
        '--show-error-bars',
        action='store_true',
        help='Show error bars on data points in the plot (default: False)'
    )

    args = parser.parse_args()

    # Generate run timestamp (used for directory and all outputs)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create run directory first (needed for log file)
    study_name = f"tau_kTe_study_{args.scenario}_{run_timestamp}"
    run_dir = Path(args.output_dir) / study_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging with file output
    log_file = run_dir / "run.log"
    logger = setup_logging(args.verbose, log_file)

    logger.info(f"Log file: {log_file}")

    # Get base parameters
    base_params = COMPPS_PARAMS[args.scenario]

    # Print study configuration
    logger.info("")
    logger.info("=" * 70)
    logger.info("PARAMETER GRID STUDY: tau_y vs kTe")
    logger.info("=" * 70)
    logger.info("")
    logger.info("CONFIGURATION:")
    logger.info(f"  Run timestamp: {run_timestamp}")
    logger.info(f"  Base scenario: {args.scenario}")
    logger.info(f"  tau_y values: {args.tau_values}")
    logger.info(f"  kTe values: {args.kTe_values}")
    total_combos = len(args.tau_values) * len(args.kTe_values)
    logger.info(f"  Total combinations: {total_combos}")
    logger.info(f"  Exposure: {args.exposure:.0f} s")
    logger.info(f"  Normalization: {args.normalization}")
    logger.info(f"  Energy range: {args.energy_range[0]:.1f} - "
                f"{args.energy_range[1]:.1f} keV")
    logger.info("")
    logger.info("BASE SCENARIO PARAMETERS:")
    for key, value in base_params.items():
        if key in ['kTe', 'tau_y', 'kTbb', 'geom', 'cosIncl', 'rel_refl']:
            logger.info(f"  {key}: {value}")

    # Validate tau values early - must be all positive or all negative
    tau_mode = validate_tau_values(args.tau_values, logger)

    # Create subdirectories (run_dir already created for log file)
    spectra_dir = run_dir / "spectra"
    fits_dir = run_dir / "fits"
    spectra_dir.mkdir(exist_ok=True)
    fits_dir.mkdir(exist_ok=True)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"  Spectra will be saved to: {spectra_dir}")
    logger.info(f"  Fit results will be saved to: {fits_dir}")

    # Create parameter grid
    grid_params = create_parameter_grid(
        base_params,
        args.tau_values,
        args.kTe_values,
        logger
    )

    # Save run metadata
    save_run_metadata(run_dir, args, base_params, grid_params,
                      run_timestamp, logger)

    # Run simulations (output to run-specific spectra directory)
    spectrum_files = run_simulations(
        grid_params,
        args.arf,
        args.rmf,
        spectra_dir,
        args.exposure,
        args.normalization,
        logger
    )

    if not spectrum_files:
        logger.error("No successful simulations. Exiting.")
        return 1

    # Generate model plots from .xcm files (if requested)
    if args.plot_models:
        plot_compps_models_by_kte(
            spectra_dir=str(spectra_dir),
            output_dir=str(run_dir),
            energy_range=args.energy_range,
            xspec_plt_en_range="0.001 1e4 1000 log",
            logger=logger
        )

    # Run fits (output to run-specific fits directory)
    fit_results = run_fits(
        spectrum_files,
        fits_dir,
        tuple(args.energy_range),
        logger
    )

    if not fit_results:
        logger.error("No successful fits. Exiting.")
        return 1

    # Extract results for plotting
    results_by_kTe = extract_results_for_plotting(
        fit_results,
        args.kTe_values,
        args.tau_values,
        logger
    )

    # Print results table
    print_results_table(results_by_kTe, logger)

    # Save results to CSV (in run directory)
    csv_file = run_dir / "results.csv"
    save_results_csv(results_by_kTe, csv_file, logger)

    # Create plot (in run directory)
    plot_file = run_dir / "photon_index_vs_tau.png"
    plot_results(
        results_by_kTe, plot_file, args.scenario, logger,
        fit_energy_range=list(args.energy_range), tau_mode=tau_mode,
        show_error_bars=args.show_error_bars
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info("STUDY COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Run directory: {run_dir}")
    logger.info(f"  Total simulations: {len(grid_params)}")
    logger.info(f"  Successful fits: {len(fit_results)}")
    logger.info("")
    logger.info("OUTPUT FILES:")
    logger.info(f"  Log file: {log_file}")
    logger.info(f"  Metadata: {run_dir / 'run_metadata.json'}")
    logger.info(f"  Results CSV: {csv_file}")
    logger.info(f"  Plot: {plot_file}")
    if args.plot_models:
        logger.info(f"  Model plots: {run_dir / 'plots'}/")
    logger.info(f"  Spectra: {spectra_dir}/")
    logger.info(f"  Fit results: {fits_dir}/")
    logger.info("")

    return 0


if __name__ == '__main__':
    sys.exit(main())
