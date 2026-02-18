#!/usr/bin/env python3
"""
Parameter grid study: rel_refl effect on photon index vs tau_y/y-parameter.

This script:
1. Takes a base scenario from parameters.py
2. For each rel_refl value, generates simulations with varying tau_y and kTe
3. Fits each simulation with absorbed powerlaw
4. Plots photon index vs tau_y/y-parameter for different kTe values (one plot per rel_refl)

Each run creates a dedicated timestamped directory to avoid mixing
results from different runs.
"""

import argparse
import json
import shutil
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


def create_parameter_grid(base_params, tau_values, kTe_values, rel_refl, logger):
    """
    Create parameter grid by varying tau_y and kTe with fixed rel_refl.

    Parameters
    ----------
    base_params : dict
        Base parameter set
    tau_values : list
        List of tau_y values to test
    kTe_values : list
        List of kTe values to test
    rel_refl : float
        Fixed rel_refl value for this grid
    logger : logging.Logger
        Logger instance

    Returns
    -------
    dict
        Dictionary of {scenario_name: params}
    """
    logger.info("Creating parameter grid...")
    logger.info(f"  Fixed rel_refl = {rel_refl}")
    logger.info("  Base parameters will be modified for tau_y and kTe")
    logger.debug(f"  Base params: kTbb={base_params['kTbb']}, "
                 f"geom={base_params['geom']}, cosIncl={base_params['cosIncl']}")

    grid_params = {}

    for kTe in kTe_values:
        for tau in tau_values:
            # Handle negative tau values in scenario name
            tau_str = format_tau_for_filename(tau)
            scenario_name = f"grid_kTe{kTe:.0f}_tau{tau_str}"
            params = base_params.copy()
            params['kTe'] = kTe
            params['tau_y'] = tau
            params['rel_refl'] = rel_refl
            grid_params[scenario_name] = params
            logger.debug(f"  Created scenario: {scenario_name}")

    logger.info(f"  Total grid points: {len(grid_params)}")
    logger.info(f"  Grid structure: {len(kTe_values)} kTe × {len(tau_values)} tau_y")

    return grid_params


def main():
    parser = argparse.ArgumentParser(
        description='Parameter grid study: rel_refl effect on photon index vs tau_y/y-parameter'
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
        '--rel-refl-values',
        type=float,
        nargs='+',
        default=[0, 0.1, 0.3, 0.5, 1, 2, 5, 10],
        help='List of rel_refl values to test (default: 0,0.1,0.3,0.5,1,2,5,10,50,100,-1)'
    )
    parser.add_argument(
        '--tau-values',
        type=float,
        nargs='+',
        default=[-0.1, -0.2, -0.5, -1.0, -1.5, -2.0],
        help='List of tau_y values to test (default: -0.1 -0.2 -0.5 -1.0 -1.5 -2.0)'
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
        '--show-error-bars',
        action='store_true',
        help='Show error bars on data points in the plot (default: False)'
    )

    args = parser.parse_args()

    # Generate run timestamp (used for directory and all outputs)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get base parameters
    base_params = COMPPS_PARAMS[args.scenario]
    geom = base_params.get('geom', 'UNKNOWN')

    # Create main run directory
    run_dir = (
        Path(args.output_dir) /
        f"rel_refl_study_{args.scenario}_geom_{geom}_{run_timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging with file output
    log_file = run_dir / "run.log"
    logger = setup_logging(args.verbose, log_file)

    logger.info(f"Log file: {log_file}")

    # Print study configuration
    logger.info("")
    logger.info("=" * 70)
    logger.info("PARAMETER GRID STUDY: rel_refl effect on photon index")
    logger.info("=" * 70)
    logger.info("")
    logger.info("CONFIGURATION:")
    logger.info(f"  Run timestamp: {run_timestamp}")
    logger.info(f"  Base scenario: {args.scenario}")
    logger.info(f"  rel_refl values: {args.rel_refl_values}")
    logger.info(f"  tau_y values: {args.tau_values}")
    logger.info(f"  kTe values: {args.kTe_values}")
    logger.info(f"  Total rel_refl studies: {len(args.rel_refl_values)}")
    logger.info(f"  Combinations per rel_refl: {len(args.tau_values) * len(args.kTe_values)}")
    logger.info(f"  Total simulations: {len(args.rel_refl_values) * len(args.tau_values) * len(args.kTe_values)}")
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

    logger.info(f"Run directory: {run_dir}")

    # Store metadata for all rel_refl studies
    all_studies_metadata = {
        'run_timestamp': run_timestamp,
        'base_scenario': args.scenario,
        'rel_refl_values': args.rel_refl_values,
        'tau_values': args.tau_values,
        'kTe_values': args.kTe_values,
        'exposure': args.exposure,
        'normalization': args.normalization,
        'energy_range': args.energy_range,
        'arf_file': args.arf,
        'rmf_file': args.rmf,
        'base_parameters': base_params,
        'total_rel_refl_studies': len(args.rel_refl_values),
        'total_combinations_per_rel_refl': len(args.tau_values) * len(args.kTe_values),
        'rel_refl_studies': {}
    }

    # Loop over each rel_refl value
    for rel_refl_idx, rel_refl in enumerate(args.rel_refl_values, 1):
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"STUDY {rel_refl_idx}/{len(args.rel_refl_values)}: rel_refl = {rel_refl}")
        logger.info("=" * 70)

        # Create subdirectory for this rel_refl value
        rel_refl_dir = run_dir / f"rel_refl_{rel_refl}"
        spectra_dir = rel_refl_dir / "spectra"
        fits_dir = rel_refl_dir / "fits"
        spectra_dir.mkdir(parents=True, exist_ok=True)
        fits_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"  Output directory: {rel_refl_dir}")
        logger.info(f"  Spectra will be saved to: {spectra_dir}")
        logger.info(f"  Fit results will be saved to: {fits_dir}")

        # Create parameter grid for this rel_refl
        grid_params = create_parameter_grid(
            base_params,
            args.tau_values,
            args.kTe_values,
            rel_refl,
            logger
        )

        # Store grid info in metadata
        study_metadata = {
            'rel_refl': rel_refl,
            'grid_scenarios': list(grid_params.keys()),
            'total_combinations': len(grid_params),
            'results_dir': f"rel_refl_{rel_refl}"
        }

        # Run simulations (discard amplification factors — not used in this study)
        spectrum_files, _ = run_simulations(
            grid_params,
            args.arf,
            args.rmf,
            spectra_dir,
            args.exposure,
            args.normalization,
            logger
        )

        if not spectrum_files:
            logger.error(f"  No successful simulations for rel_refl={rel_refl}. Skipping.")
            study_metadata['status'] = 'failed_simulations'
            all_studies_metadata['rel_refl_studies'][str(rel_refl)] = study_metadata
            continue

        # Run fits
        fit_results = run_fits(
            spectrum_files,
            fits_dir,
            tuple(args.energy_range),
            logger
        )

        if not fit_results:
            logger.error(f"  No successful fits for rel_refl={rel_refl}. Skipping.")
            study_metadata['status'] = 'failed_fits'
            all_studies_metadata['rel_refl_studies'][str(rel_refl)] = study_metadata
            continue

        # Extract results for plotting
        results_by_kTe = extract_results_for_plotting(
            fit_results,
            args.kTe_values,
            args.tau_values,
            logger
        )

        # Print results table
        print_results_table(results_by_kTe, logger)

        # Save results to CSV
        csv_file = rel_refl_dir / "results.csv"
        save_results_csv(results_by_kTe, csv_file, logger)

        # Create plot
        plot_file = rel_refl_dir / "photon_index_vs_tau.png"
        plot_results(
            results_by_kTe, plot_file, args.scenario, logger,
            fit_energy_range=args.energy_range,
            tau_mode=tau_mode, rel_refl=rel_refl,
            show_error_bars=args.show_error_bars
            )

        # Update metadata
        study_metadata['status'] = 'success'
        study_metadata['successful_simulations'] = len(spectrum_files)
        study_metadata['successful_fits'] = len(fit_results)
        all_studies_metadata['rel_refl_studies'][str(rel_refl)] = study_metadata

        logger.info("")
        logger.info(f"  Study complete for rel_refl = {rel_refl}")
        logger.info(f"    Successful simulations: {len(spectrum_files)}")
        logger.info(f"    Successful fits: {len(fit_results)}")
        logger.info(f"    Plot: {plot_file}")
        logger.info(f"    CSV: {csv_file}")

    # Save combined metadata
    metadata_file = run_dir / "run_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(all_studies_metadata, f, indent=2)

    # Copy all plots to a separate plots directory
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    copied_plots = []

    logger.info("")
    logger.info("=" * 70)
    logger.info("COPYING PLOTS TO CENTRAL DIRECTORY")
    logger.info("=" * 70)

    for rel_refl in args.rel_refl_values:
        plot_file = run_dir / f"rel_refl_{rel_refl}" / "photon_index_vs_tau.png"
        if plot_file.exists():
            # Create a descriptive filename
            plot_name = f"photon_index_vs_tau_rel_refl_{rel_refl}.png"
            dest_file = plots_dir / plot_name
            shutil.copy2(plot_file, dest_file)
            copied_plots.append(plot_name)
            logger.info(f"  Copied: {plot_name}")

    logger.info(f"  Total plots copied: {len(copied_plots)}")
    logger.info(f"  Plots directory: {plots_dir}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("ALL STUDIES COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Run directory: {run_dir}")
    logger.info(f"  Total rel_refl studies: {len(args.rel_refl_values)}")
    logger.info("")
    logger.info("OUTPUT FILES:")
    logger.info(f"  Log file: {log_file}")
    logger.info(f"  Metadata: {metadata_file}")
    logger.info(f"  Plots directory: {plots_dir}/ ({len(copied_plots)} plots)")
    logger.info("")
    logger.info("Per-rel_refl directories:")
    for rel_refl in args.rel_refl_values:
        rel_refl_dir = run_dir / f"rel_refl_{rel_refl}"
        if rel_refl_dir.exists():
            logger.info(f"  rel_refl_{rel_refl}/")
            logger.info(f"    - spectra/")
            logger.info(f"    - fits/")
            logger.info(f"    - results.csv")
            logger.info(f"    - photon_index_vs_tau.png")
    logger.info("")

    return 0


if __name__ == '__main__':
    sys.exit(main())

