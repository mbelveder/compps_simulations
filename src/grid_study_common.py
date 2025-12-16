"""
Common functions for parameter grid studies.

This module contains shared functionality used by multiple grid study scripts
(e.g., tau_kTe_study, rel_refl_study) to avoid code duplication.
"""

from ast import List
import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
        msg = ("tau_y values cannot be zero "
               "(must be strictly positive or negative)")
        raise ValueError(msg)

    if positive_count > 0 and negative_count > 0:
        msg = ("tau_y values must be all positive or all negative, "
               "not mixed. "
               f"Found {positive_count} positive and "
               f"{negative_count} negative values.")
        raise ValueError(msg)

    if positive_count > 0:
        logger.info("  tau_y mode: positive (optical depth)")
        return 'positive'
    else:
        logger.info("  tau_y mode: negative (Compton y-parameter)")
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


def format_tau_for_filename(tau: float) -> str:
    """
    Format tau value for use in filenames.

    Negative values use 'm' prefix instead of minus sign to create valid
    filenames. E.g., -0.50 → "m0.50", 0.50 → "0.50"

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
                 fit_energy_range: List, tau_mode: str = 'positive',
                 rel_refl: float = None, show_error_bars: bool = False):
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
    rel_refl : float, optional
        If provided, include rel_refl value in title (for rel_refl studies)
    show_error_bars : bool, optional
        If True, plot error bars on data points (default: False)
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 4: GENERATING PLOT")
    logger.info("=" * 70)

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 7)
    plt.rcParams['font.size'] = 14

    _, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results_by_kTe)))

    for (kTe, data), color in zip(sorted(results_by_kTe.items()), colors):
        if len(data['tau']) > 0:
            gamma = np.array(data['gamma'])

            # Inverse y-parameter values when plotting
            if tau_mode == 'negative':
                tau = -np.array(data['tau'])
            else:
                tau = np.array(data['tau'])

            logger.info(f"  Plotting kTe = {kTe:.0f} keV: {len(tau)} data points")

            # Create smooth curve using polynomial fitting
            degree = 3  # cubic polynomial
            coeffs = np.polyfit(tau, gamma, degree)
            poly = np.poly1d(coeffs)

            tau_smooth = np.linspace(tau.min(), tau.max(), 300)
            gamma_smooth = poly(tau_smooth)

            ax.plot(
                tau_smooth, gamma_smooth,
                marker=None,
                lw=3,
                color=color,
                alpha=0.8,
                label=f'kTe = {kTe:.0f} keV'
            )

            if show_error_bars:
                err_neg = np.array(data['gamma_err_neg'])
                err_pos = np.array(data['gamma_err_pos'])

                ax.errorbar(
                    tau, gamma,
                    yerr=[err_neg, err_pos],
                    marker='o',
                    markersize=3,
                    linestyle='',
                    linewidth=2,
                    capsize=5,
                    capthick=2,
                    color=color,
                    ecolor=color,
                    alpha=0.8
                )
        else:
            logger.warning(f"  kTe = {kTe:.0f} keV: No data points to plot")

    # Update x-axis label based on tau_mode
    if tau_mode == 'negative':
        ax.set_xlabel(
            r'Compton y-parameter' + r' ($4\tau\frac{k_B T_e}{m_e c^2}$)',
            fontsize=16
        )
        title_x_label = 'Compton\ y-parameter'
        # ax.axvline(1, color='k', alpha=.3)
    else:
        ax.set_xlabel(r'Optical Depth ($\tau_y$)', fontsize=14)
        title_x_label = 'Optical Depth'

    ax.set_ylabel(r'Photon Index ($\Gamma$, tbabs*po)', fontsize=16)
    base_params = COMPPS_PARAMS[base_scenario_name]
    # kTbb = base_params.get('kTbb', 'UNKNOWN')
    base_rel_refl = base_params.get('rel_refl', 'UNKNOWN')
    # geom = base_params.get('geom', 'UNKNOWN')
    # cosIncl = base_params.get('cosIncl', 'UNKNOWN')

    # Build title with optional rel_refl override
    if rel_refl is not None:
        title_rel_refl = rel_refl
    else:
        title_rel_refl = base_rel_refl

    title = (
        '\n\n\n'
        # fr'$\bf{{Photon\ Index\ vs\ {title_x_label}}}$'
        # f'\ngeom: {geom}, rel_refl: {title_rel_refl}, '
        # f'kTbb: {kTbb} keV, cosIncl: {cosIncl}\n'
        # r'$\mathtt{tbabs*po}$' + ' fitting energy range: '
        # f'{fit_energy_range[0]:.1f}–{fit_energy_range[1]:.1f} keV'
    )
    ax.set_title(title, fontsize=16, pad=20)
    ax.legend(fontsize=12, frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(1.3, alpha=0.6, linestyle='--', color='k')

    ax.set_xlim(0.1, 1)
    ax.set_ylim(0, 4.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # transparent=True
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
            row_fmt = ("| {:>8.0f} | {:>8.2f} | {:>12.4f} | "
                       "{:>10.4f} | {:>10.4f} |")
            row = row_fmt.format(
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
                    f"{data['gamma_err_neg'][i]},"
                    f"{data['gamma_err_pos'][i]}\n")

    logger.info(f"  Results CSV saved: {output_file}")

