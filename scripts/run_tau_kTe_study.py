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
import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parameters import COMPPS_PARAMS
from src.simulator import SpectrumSimulator
from src.fitter import SpectrumFitter


def validate_tau_values(tau_values: list, logger) -> str:
    """
    Validate tau_y values - must be all positive or all negative.

    Parameters
    ----------
    tau_values : list
        List of tau_y values to validate
    logger : logging.Logger
        Logger instance

    Returns
    -------
    str
        'positive' if all values > 0, 'negative' if all values < 0

    Raises
    ------
    ValueError
        If values are mixed positive and negative, or if list is empty
    """
    if not tau_values:
        raise ValueError("tau_values list cannot be empty")

    positive_count = sum(1 for v in tau_values if v > 0)
    negative_count = sum(1 for v in tau_values if v < 0)
    zero_count = sum(1 for v in tau_values if v == 0)

    if zero_count > 0:
        raise ValueError(
            "tau_y values cannot be zero (must be strictly positive or negative)"
        )

    if positive_count > 0 and negative_count > 0:
        raise ValueError(
            "tau_y values must be all positive or all negative, not mixed. "
            f"Found {positive_count} positive and {negative_count} negative values."
        )

    if positive_count > 0:
        logger.info(f"  tau_y mode: positive (optical depth)")
        return 'positive'
    else:
        logger.info(f"  tau_y mode: negative (Compton y-parameter)")
        return 'negative'


def setup_logging(verbose: bool = False, log_file: Path = None):
    """
    Set up logging configuration.

    Parameters
    ----------
    verbose : bool
        If True, set DEBUG level; otherwise INFO level
    log_file : Path, optional
        If provided, also log to this file

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)  # Always DEBUG for file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


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
    run_dir = Path(base_output_dir) / f"tau_kTe_study_{scenario_name}_{timestamp}"
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


def format_tau_for_filename(tau: float) -> str:
    """
    Format tau value for use in filenames.
    
    Negative values use 'm' prefix instead of minus sign to create valid filenames.
    E.g., -0.50 → "m0.50", 0.50 → "0.50"
    
    Parameters
    ----------
    tau : float
        Tau value (can be positive or negative)
        
    Returns
    -------
    str
        Formatted tau string suitable for filenames
    """
    if tau < 0:
        return f"m{abs(tau):.2f}"
    else:
        return f"{tau:.2f}"


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
            grid_params[scenario_name] = params
            logger.debug(f"  Created scenario: {scenario_name}")

    logger.info(f"  Total grid points: {len(grid_params)}")
    logger.info(f"  Grid structure: {len(kTe_values)} kTe × {len(tau_values)} tau_y")

    return grid_params


def run_simulations(grid_params, arf_file, rmf_file, spectra_dir,
                    exposure, normalization, logger):
    """
    Run simulations for all parameter combinations.

    Parameters
    ----------
    grid_params : dict
        Parameter grid
    arf_file : str
        ARF filename
    rmf_file : str
        RMF filename
    spectra_dir : Path
        Directory for simulated spectra (run-specific)
    exposure : float
        Exposure time (seconds)
    normalization : float
        Model normalization
    logger : logging.Logger
        Logger instance

    Returns
    -------
    dict
        Dictionary of {scenario_name: spectrum_file}
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 1: SPECTRUM SIMULATION")
    logger.info("=" * 70)
    logger.info(f"ARF file: {arf_file}")
    logger.info(f"RMF file: {rmf_file}")
    logger.info(f"Output directory: {spectra_dir}")
    logger.info(f"Exposure time: {exposure:.0f} seconds")
    logger.info(f"Normalization: {normalization}")
    logger.info("")

    simulator = SpectrumSimulator(
        arf_file=arf_file,
        rmf_file=rmf_file,
        output_dir=str(spectra_dir)
    )
    spectrum_files = {}
    failed_simulations = []

    total = len(grid_params)
    for i, (scenario_name, params) in enumerate(grid_params.items(), 1):
        logger.info(f"[{i}/{total}] Simulating: {scenario_name}")
        logger.info(f"         kTe = {params['kTe']:.1f} keV")
        logger.info(f"         tau_y = {params['tau_y']:.2f}")

        try:
            grouped_spectrum = simulator.simulate_spectrum(
                scenario_name=scenario_name,
                compps_params=params,
                exposure=exposure,
                norm=normalization
            )

            spectrum_files[scenario_name] = grouped_spectrum
            logger.info(f"         ✓ Output: {Path(grouped_spectrum).name}")

        except Exception as e:
            logger.error(f"         ✗ FAILED: {e}")
            failed_simulations.append(scenario_name)
            continue

    logger.info("")
    logger.info("-" * 70)
    logger.info("SIMULATION SUMMARY")
    logger.info("-" * 70)
    logger.info(f"  Successful: {len(spectrum_files)}/{total}")
    logger.info(f"  Failed: {len(failed_simulations)}/{total}")
    if failed_simulations:
        logger.warning(f"  Failed scenarios: {', '.join(failed_simulations)}")

    return spectrum_files


def run_fits(spectrum_files, fits_dir, energy_range, logger):
    """
    Fit all simulated spectra.

    Parameters
    ----------
    spectrum_files : dict
        Dictionary of {scenario_name: spectrum_file}
    fits_dir : Path
        Directory for fit results (run-specific)
    energy_range : tuple
        Energy range for fitting (min, max) in keV
    logger : logging.Logger
        Logger instance

    Returns
    -------
    dict
        Dictionary of {scenario_name: fit_result}
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 2: SPECTRAL FITTING")
    logger.info("=" * 70)
    logger.info("Model: tbabs * powerlaw")
    logger.info(f"Energy range: {energy_range[0]:.1f} - {energy_range[1]:.1f} keV")
    logger.info(f"Results directory: {fits_dir}")
    logger.info("")

    fitter = SpectrumFitter(
        output_dir=str(fits_dir),
        energy_range=energy_range
    )

    fit_results = {}
    failed_fits = []

    total = len(spectrum_files)
    for i, (scenario_name, spectrum_file) in enumerate(spectrum_files.items(), 1):
        logger.info(f"[{i}/{total}] Fitting: {scenario_name}")
        logger.debug(f"         Spectrum: {spectrum_file}")

        try:
            result = fitter.fit_spectrum(
                spectrum_file=spectrum_file,
                nh_init=0.01,
                photon_index_init=2.0,
                norm_init=1e-3
            )

            if result['fit_status'] == 'success':
                fit_results[scenario_name] = result
                gamma = result['parameters']['powerlaw.PhoIndex']
                chi2 = result.get('reduced_chi_squared', 0)

                # Extract errors if available
                errors = result.get('errors', {})
                phoindex_error = errors.get('powerlaw.PhoIndex')
                if (isinstance(phoindex_error, (list, tuple)) and
                        len(phoindex_error) == 2):
                    err_neg = gamma - phoindex_error[0]
                    err_pos = phoindex_error[1] - gamma
                    logger.info(f"         ✓ Γ = {gamma:.3f} "
                                f"(-{err_neg:.3f}/+{err_pos:.3f})")
                else:
                    logger.info(f"         ✓ Γ = {gamma:.3f}")
                logger.info(f"           χ²_red = {chi2:.3f}")
            else:
                logger.warning(f"         ✗ Fit failed "
                               f"(status: {result['fit_status']})")
                failed_fits.append(scenario_name)

        except Exception as e:
            logger.error(f"         ✗ ERROR: {e}")
            failed_fits.append(scenario_name)
            continue

    logger.info("")
    logger.info("-" * 70)
    logger.info("FITTING SUMMARY")
    logger.info("-" * 70)
    logger.info(f"  Successful: {len(fit_results)}/{total}")
    logger.info(f"  Failed: {len(failed_fits)}/{total}")
    if failed_fits:
        logger.warning(f"  Failed scenarios: {', '.join(failed_fits)}")

    return fit_results


def extract_results_for_plotting(fit_results, kTe_values, tau_values, logger):
    """
    Extract photon indices and errors organized by kTe.

    Parameters
    ----------
    fit_results : dict
        Fit results dictionary
    kTe_values : list
        List of kTe values
    tau_values : list
        List of tau_y values
    logger : logging.Logger
        Logger instance

    Returns
    -------
    dict
        Dictionary with structure:
        {kTe: {'tau': [...], 'gamma': [...],
               'gamma_err_neg': [...], 'gamma_err_pos': [...]}}
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 3: EXTRACTING RESULTS FOR PLOTTING")
    logger.info("=" * 70)

    results_by_kTe = {}

    for kTe in kTe_values:
        results_by_kTe[kTe] = {
            'tau': [],
            'gamma': [],
            'gamma_err_neg': [],
            'gamma_err_pos': []
        }

        logger.info(f"  kTe = {kTe:.0f} keV:")

        for tau in tau_values:
            # Handle negative tau values in scenario name lookup
            tau_str = format_tau_for_filename(tau)
            scenario_name = f"grid_kTe{kTe:.0f}_tau{tau_str}"

            if scenario_name in fit_results:
                result = fit_results[scenario_name]
                gamma = result['parameters'].get('powerlaw.PhoIndex')
                errors = result.get('errors', {})
                phoindex_error = errors.get('powerlaw.PhoIndex')

                if gamma is not None:
                    results_by_kTe[kTe]['tau'].append(tau)
                    results_by_kTe[kTe]['gamma'].append(gamma)

                    if (isinstance(phoindex_error, (list, tuple)) and
                            len(phoindex_error) == 2):
                        err_neg = gamma - phoindex_error[0]
                        err_pos = phoindex_error[1] - gamma
                        results_by_kTe[kTe]['gamma_err_neg'].append(err_neg)
                        results_by_kTe[kTe]['gamma_err_pos'].append(err_pos)
                        logger.info(f"    tau={tau:.2f}: Γ={gamma:.3f} "
                                    f"(-{err_neg:.3f}/+{err_pos:.3f})")
                    else:
                        results_by_kTe[kTe]['gamma_err_neg'].append(0)
                        results_by_kTe[kTe]['gamma_err_pos'].append(0)
                        logger.info(f"    tau={tau:.2f}: Γ={gamma:.3f} "
                                    "(no errors)")
            else:
                logger.warning(f"    tau={tau:.2f}: MISSING")

    return results_by_kTe


def plot_results(results_by_kTe, output_file, base_scenario_name, logger,
                 tau_mode: str = 'positive'):
    """
    Create plot of photon index vs tau_y for different kTe values.

    Parameters
    ----------
    results_by_kTe : dict
        Results organized by kTe
    output_file : Path
        Output plot filename
    base_scenario_name : str
        Name of base scenario
    logger : logging.Logger
        Logger instance
    tau_mode : str
        'positive' for optical depth mode, 'negative' for Compton y-parameter mode
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 4: GENERATING PLOT")
    logger.info("=" * 70)

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 7)
    plt.rcParams['font.size'] = 10

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results_by_kTe)))

    for (kTe, data), color in zip(sorted(results_by_kTe.items()), colors):
        if len(data['tau']) > 0:
            gamma = np.array(data['gamma'])
            err_neg = np.array(data['gamma_err_neg'])
            err_pos = np.array(data['gamma_err_pos'])

            # Inverse y-parameter values when plotting
            if tau_mode == 'negative':
                tau = -np.array(data['tau'])
            else:
                tau = np.array(data['tau'])

            logger.info(f"  Plotting kTe = {kTe:.0f} keV: {len(tau)} data points")

            ax.errorbar(
                tau, gamma,
                yerr=[err_neg, err_pos],
                marker='o',
                markersize=3,
                linestyle='-',
                linewidth=2,
                capsize=5,
                capthick=2,
                label=f'kTe = {kTe:.0f} keV',
                color=color,
                ecolor=color,
                alpha=0.8
            )
        else:
            logger.warning(f"  kTe = {kTe:.0f} keV: No data points to plot")

    # Update x-axis label based on tau_mode
    if tau_mode == 'negative':
        ax.set_xlabel(r'Compton y-parameter (4 × Θ × τ)', fontsize=14)
        title_x_label = 'Compton y-parameter'
        ax.axvline(1, color='k', alpha=.3)
    else:
        ax.set_xlabel(r'Optical Depth ($\tau_y$)', fontsize=14)
        title_x_label = 'Optical Depth'
    
    ax.set_ylabel(r'Photon Index ($\Gamma$, tbabs*po)', fontsize=14)
    base_params = COMPPS_PARAMS['typical_agn_slab']
    kTbb = base_params.get('kTbb', 'UNKNOWN')
    ax.set_title(
        (f'Photon Index vs {title_x_label}\nBase Scenario: {base_scenario_name}, '
        f'kTbb: {kTbb} keV'),
        fontsize=16,
        pad=20
    )
    ax.legend(fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(1.3, alpha=0.6, linestyle='--', color='k')

    ax.set_xlim(0, 2.1)
    ax.set_ylim(0.5, 4)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"  Plot saved: {output_file}")
    plt.close()


def print_results_table(results_by_kTe, logger):
    """
    Print a formatted table of results.

    Parameters
    ----------
    results_by_kTe : dict
        Results organized by kTe
    logger : logging.Logger
        Logger instance
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS TABLE")
    logger.info("=" * 70)

    # Header
    header = "| {:>8} | {:>8} | {:>12} | {:>10} | {:>10} |".format(
        "kTe", "tau_y", "Gamma", "err_neg", "err_pos"
    )
    separator = ("|" + "-" * 10 + "|" + "-" * 10 + "|" +
                 "-" * 14 + "|" + "-" * 12 + "|" + "-" * 12 + "|")

    logger.info(separator)
    logger.info(header)
    logger.info(separator)

    for kTe in sorted(results_by_kTe.keys()):
        data = results_by_kTe[kTe]
        for i in range(len(data['tau'])):
            row = "| {:>8.0f} | {:>8.2f} | {:>12.4f} | {:>10.4f} | {:>10.4f} |".format(
                kTe,
                data['tau'][i],
                data['gamma'][i],
                data['gamma_err_neg'][i],
                data['gamma_err_pos'][i]
            )
            logger.info(row)
        logger.info(separator)


def save_results_csv(results_by_kTe, output_file, logger):
    """
    Save results to CSV file.

    Parameters
    ----------
    results_by_kTe : dict
        Results organized by kTe
    output_file : Path
        Output CSV filename
    logger : logging.Logger
        Logger instance
    """
    with open(output_file, 'w') as f:
        f.write("kTe,tau_y,gamma,gamma_err_neg,gamma_err_pos\n")
        for kTe in sorted(results_by_kTe.keys()):
            data = results_by_kTe[kTe]
            for i in range(len(data['tau'])):
                f.write(f"{kTe},{data['tau'][i]},{data['gamma'][i]},"
                        f"{data['gamma_err_neg'][i]},{data['gamma_err_pos'][i]}\n")

    logger.info(f"  Results CSV saved: {output_file}")


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

    args = parser.parse_args()

    # Generate run timestamp (used for directory and all outputs)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create run directory first (needed for log file)
    run_dir = Path(args.output_dir) / f"tau_kTe_study_{args.scenario}_{run_timestamp}"
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
    logger.info(f"  Total combinations: {len(args.tau_values) * len(args.kTe_values)}")
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
    plot_results(results_by_kTe, plot_file, args.scenario, logger, tau_mode)

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
    logger.info(f"  Spectra: {spectra_dir}/")
    logger.info(f"  Fit results: {fits_dir}/")
    logger.info("")

    return 0


if __name__ == '__main__':
    sys.exit(main())
