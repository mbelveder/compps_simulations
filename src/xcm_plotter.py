"""
XCM Grid Plotter Module

This module creates a grid of spectrum plots from XSPEC .xcm files,
showing the fitted spectra and residuals for multiple scenarios.
"""

import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

try:
    from xspec import AllData, AllModels, Xset, Plot
    XSPEC_AVAILABLE = True
except ImportError:
    XSPEC_AVAILABLE = False
    print("Warning: PyXspec not available. XCM plotting will not work.")

from src.utils import ChangeDir
from src import plot_settings


def get_scenario_number_map() -> Dict[str, int]:
    """
    Get consistent scenario numbering based on COMPPS_PARAMS order.

    Returns
    -------
    dict
        Mapping of scenario names to numbers (1-indexed)
    """
    from config.parameters import COMPPS_PARAMS
    return {name: idx + 1 for idx, name in enumerate(COMPPS_PARAMS.keys())}


def find_xcm_files(results_dir: str = 'data/results') -> List[Tuple[str, str]]:
    """
    Find all .xcm files and extract scenario names.

    Parameters
    ----------
    results_dir : str
        Directory containing .xcm files

    Returns
    -------
    list of tuple
        List of (xcm_file_path, scenario_name) tuples, sorted by scenario number
    """
    results_path = Path(results_dir)
    xcm_pattern = re.compile(r'fit_sim_(.+?)_\d{8}_\d{6}_grp\.xcm')

    xcm_files = []
    for xcm_file in results_path.glob('fit_sim_*_grp.xcm'):
        match = xcm_pattern.match(xcm_file.name)
        if match:
            scenario_name = match.group(1)
            xcm_files.append((str(xcm_file), scenario_name))

    # Sort by scenario number for consistent ordering
    scenario_map = get_scenario_number_map()
    xcm_files.sort(key=lambda x: scenario_map.get(x[1], 999))

    return xcm_files


def plot_single_spectrum(ax1, ax2, xcm_file: str, scenario_name: str,
                         scenario_num: int, rebin: int = 10,
                         group_num: int = 1, color: str = 'C0'):
    """
    Plot a single spectrum with model and residuals.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        Axis for spectrum plot
    ax2 : matplotlib.axes.Axes
        Axis for residuals plot
    xcm_file : str
        Path to .xcm file
    scenario_name : str
        Name of the scenario
    scenario_num : int
        Scenario number for title
    rebin : int
        Rebinning factor for plotting
    group_num : int
        Group number for XSPEC plotting
    color : str
        Color for data points
    """
    if not XSPEC_AVAILABLE:
        ax1.text(0.5, 0.5, 'PyXspec not available',
                ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, 'PyXspec not available',
                ha='center', va='center', transform=ax2.transAxes)
        return

    try:
        # Clear and restore XSPEC session
        AllData.clear()
        AllModels.clear()
        Plot.device = '/null'
        Xset.restore(xcm_file)
        Plot.xAxis = 'keV'

        # Set rebinning and plot
        Plot.setRebin(rebin, rebin)
        Plot("eeufs ra")

        # Extract plot data
        energies = Plot.x(group_num, 1)
        edeltas = Plot.xErr(group_num, 1)
        rates = Plot.y(group_num, 1)
        errors = Plot.yErr(group_num, 1)
        foldedmodel = Plot.model(group_num)
        dataLabels = Plot.labels(1)

        # Prepare step energies for model plot
        nE = len(energies)
        stepenergies = []
        for i in range(nE):
            stepenergies.append(energies[i] - edeltas[i])
        stepenergies.append(energies[-1] + edeltas[-1])
        foldedmodel.append(foldedmodel[-1])

        # Get residuals
        resid = Plot.y(group_num, 2)
        residerr = Plot.yErr(group_num, 2)
        # residLabels = Plot.labels(2)

        # Plot spectrum
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.errorbar(
            energies, rates, xerr=edeltas, yerr=errors,
            fmt='.', alpha=0.8, color=color, zorder=10
        )
        ax1.scatter(
            energies, rates, color=color, s=1, zorder=10
        )
        ax1.plot(
            stepenergies, foldedmodel, color='lime', lw=1,
            alpha=0.8
        )

        ax1.set_xlabel(None)
        ax1.set_xlim(0.3, 11)
        ax1.set_ylim(8e-7, 1e-3)
        # ax1.set_ylabel(dataLabels[1], fontsize=6)
        ax1.set_ylabel(None)
        ax1.set_title(f'{scenario_num}. {scenario_name}', fontsize=10, weight='bold')
        ax1.tick_params(labelsize=5)

        # Plot residuals
        ax2.set_xscale('log')
        ax2.errorbar(
            energies, resid, xerr=edeltas, yerr=residerr, fmt='.',
            color=color, alpha=0.8
        )
        ax2.axhline(1, ls='-', color='lime', alpha=0.4)
        ax2.set_ylim(-1, 3)
        ax2.set_xlim(0.3, 10)
        # ax2.set_xlabel(residLabels[0], fontsize=6)
        ax2.set_xlabel(None)
        # ax2.set_ylabel(residLabels[1], fontsize=6)
        ax2.set_ylabel(None)
        ax2.tick_params(labelsize=5)

    except Exception as e:
        # Handle errors gracefully
        ax1.text(0.5, 0.5, f'Error loading\n{scenario_name}',
                ha='center', va='center', transform=ax1.transAxes,
                fontsize=6)
        ax2.text(0.5, 0.5, f'Error: {str(e)[:20]}',
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=5)
        print(f"Warning: Failed to plot {scenario_name}: {e}")


def plot_xcm_grid(results_dir: str = 'data/results',
                  output_file: str = 'spectra_grid.png',
                  figsize: Tuple[float, float] = (30, 20),
                  dpi: int = 300) -> str:
    """
    Create a grid plot of all .xcm spectra with models and residuals.

    Parameters
    ----------
    results_dir : str
        Directory containing .xcm files
    output_file : str
        Output filename (will be saved in results_dir)
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Resolution for saved figure

    Returns
    -------
    str
        Path to saved figure
    """
    # Set up plotting style
    plot_settings.set_mpl()

    # Find all .xcm files
    xcm_files = find_xcm_files(results_dir)

    if not xcm_files:
        print(f"No .xcm files found in {results_dir}")
        return None

    print(f"Found {len(xcm_files)} .xcm files")

    # Get scenario numbering
    scenario_map = get_scenario_number_map()

    # Determine grid size (4x4 for 16 scenarios)
    n_scenarios = len(xcm_files)
    n_cols = 4
    n_rows = int(np.ceil(n_scenarios / n_cols))

    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)

    # Use ChangeDir to ensure correct working directory for XSPEC
    with ChangeDir(Path(__file__).parent.parent):
        for idx, (xcm_file, scenario_name) in enumerate(xcm_files):
            scenario_num = scenario_map.get(scenario_name, idx + 1)

            # Create subplot pair for this scenario
            # Each scenario gets 2 rows (spectrum + residuals) + spacer
            scenario_block = idx // n_cols
            row_offset = scenario_block * 3  # 3 = 2 rows + 1 spacer
            row = row_offset
            col = idx % n_cols

            # Create height_ratios with spacers between scenario rows
            if idx == 0:  # Only create GridSpec once
                height_ratios = []
                for i in range(n_rows):
                    height_ratios.extend([3, 1])  # spectrum + residuals
                    if i < n_rows - 1:  # Don't add spacer after last row
                        height_ratios.append(0.3)  # spacer between scenarios

                total_rows = n_rows * 2 + (n_rows - 1)
                gs = GridSpec(total_rows, n_cols, figure=fig,
                             height_ratios=height_ratios,
                             hspace=0.05, wspace=0.3)
            ax1 = fig.add_subplot(gs[row, col])
            ax2 = fig.add_subplot(gs[row + 1, col], sharex=ax1)

            # Hide x-axis labels on spectrum plot
            plt.setp(ax1.get_xticklabels(), visible=False)
            # plt.setp(ax2.get_xticklabels(), visible=False)

            print(f"Plotting {scenario_num}. {scenario_name}")

            # Plot the spectrum
            plot_single_spectrum(ax1, ax2, xcm_file, scenario_name,
                               scenario_num)

    # Save figure
    output_path = Path(results_dir) / output_file
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"\nGrid plot saved to: {output_path}")
    return str(output_path)
