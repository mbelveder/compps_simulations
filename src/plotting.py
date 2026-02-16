"""
Plotting module for visualization of spectra and analysis results.

This module provides functions to plot simulated spectra, fitted models,
and analysis results.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from astropy.io import fits

from .analysis import ResultsAnalyzer
from .utils import ChangeDir
from config.parameters import get_scenario_number_map

try:
    from xspec import AllData, AllModels, Xset, Plot, Model
    XSPEC_AVAILABLE = True
except ImportError:
    XSPEC_AVAILABLE = False


class SpectrumPlotter:
    """Create plots for spectra and analysis results."""

    def __init__(self,
                 output_dir: str = "data/results",
                 figsize: Tuple[int, int] = (10, 8),
                 dpi: int = 300):
        """
        Initialize plotter.

        Parameters
        ----------
        output_dir : str
            Directory for saving plots
        figsize : tuple
            Default figure size (width, height)
        dpi : int
            Resolution for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi

        # Set plotting style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 10

    @staticmethod
    def _get_varying_parameters(scenarios: List[str]) -> List[str]:
        """
        Identify CompPS parameters that vary across scenarios.

        Parameters
        ----------
        scenarios : list of str
            List of scenario names

        Returns
        -------
        list of str
            Parameter names that have different values across scenarios
        """
        from config.parameters import COMPPS_PARAMS

        if len(scenarios) <= 1:
            return []

        # Get all parameter names from first scenario
        first_scenario = scenarios[0]
        all_params = list(COMPPS_PARAMS[first_scenario].keys())

        varying_params = []
        for param in all_params:
            # Get all values for this parameter
            values = [COMPPS_PARAMS[s][param] for s in scenarios]
            # Check if any value differs
            if len(set(values)) > 1:
                varying_params.append(param)

        return varying_params

    def plot_spectrum(self,
                     spectrum_file: str,
                     title: Optional[str] = None,
                     save_as: Optional[str] = None,
                     show: bool = True) -> plt.Figure:
        """
        Plot a single spectrum.

        Parameters
        ----------
        spectrum_file : str
            Path to spectrum file
        title : str, optional
            Plot title
        save_as : str, optional
            Filename to save plot
        show : bool
            Whether to display plot

        Returns
        -------
        plt.Figure
            Figure object
        """
        # Read spectrum data
        with fits.open(spectrum_file) as hdul:
            data = hdul['SPECTRUM'].data
            channel = data['CHANNEL']
            counts = data['COUNTS']

            # Try to get energy information
            if 'ENERGY' in data.columns.names:
                energy = data['ENERGY']
            else:
                # Use channel as proxy
                energy = channel

        fig, (ax1, ax) = plt.subplots(2, 1, figsize=self.figsize,
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Plot counts spectrum
        ax1.step(energy, counts, where='mid', linewidth=1.5)
        ax1.set_ylabel('Counts', fontsize=12)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        if title:
            ax1.set_title(title, fontsize=14)
        else:
            ax1.set_title(f'Spectrum: {Path(spectrum_file).name}', fontsize=14)

        # Plot counts per energy bin (rate-like)
        ax.step(energy, counts, where='mid', linewidth=1.5, color='C1')
        ax.set_xlabel('Energy (keV)', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_as:
            save_path = self.output_dir / save_as
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_fit_comparison(self,
                           spectrum_file: str,
                           fit_result: Dict[str, Any],
                           title: Optional[str] = None,
                           save_as: Optional[str] = None,
                           show: bool = True) -> plt.Figure:
        """
        Plot spectrum with fitted model and residuals.

        Parameters
        ----------
        spectrum_file : str
            Path to spectrum file
        fit_result : dict
            Fit result dictionary
        title : str, optional
            Plot title
        save_as : str, optional
            Filename to save plot
        show : bool
            Whether to display plot

        Returns
        -------
        plt.Figure
            Figure object
        """
        # Read spectrum
        with fits.open(spectrum_file) as hdul:
            data = hdul['SPECTRUM'].data
            channel = data['CHANNEL']
            counts = data['COUNTS']

        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)

        ax_spec = fig.add_subplot(gs[0])
        ax_resid = fig.add_subplot(gs[1], sharex=ax_spec)
        ax_chi = fig.add_subplot(gs[2], sharex=ax_spec)

        # Plot spectrum
        ax_spec.step(channel, counts, where='mid', label='Data', linewidth=1.5)

        # Add fit parameters to title
        if title is None:
            params = fit_result.get('parameters', {})
            photon_index = params.get('powerlaw.PhoIndex', 0)
            nh = params.get('tbabs.nH', 0)
            chi2 = fit_result.get('reduced_chi_squared', 0)
            title = (f"Fit: Γ={photon_index:.2f}, "
                    f"nH={nh:.3f}×10²² cm⁻², χ²ᵣ={chi2:.2f}")

        ax_spec.set_title(title, fontsize=12)
        ax_spec.set_ylabel('Counts', fontsize=12)
        ax_spec.set_yscale('log')
        ax_spec.legend(loc='best')
        ax_spec.grid(True, alpha=0.3)
        plt.setp(ax_spec.get_xticklabels(), visible=False)

        # Plot residuals (placeholder - would need model values)
        ax_resid.axhline(0, color='k', linestyle='--', linewidth=1)
        ax_resid.set_ylabel('Residuals', fontsize=10)
        ax_resid.grid(True, alpha=0.3)
        plt.setp(ax_resid.get_xticklabels(), visible=False)

        # Plot chi contributions
        ax_chi.set_xlabel('Channel', fontsize=12)
        ax_chi.set_ylabel('χ', fontsize=10)
        ax_chi.grid(True, alpha=0.3)

        if save_as:
            save_path = self.output_dir / save_as
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_parameter_comparison(self,
                                  df: pd.DataFrame,
                                  x_param: str,
                                  y_param: str,
                                  title: Optional[str] = None,
                                  save_as: Optional[str] = None,
                                  show: bool = True) -> plt.Figure:
        """
        Plot comparison between two parameters.

        Parameters
        ----------
        df : pd.DataFrame
            Results DataFrame
        x_param : str
            X-axis parameter name
        y_param : str
            Y-axis parameter name
        title : str, optional
            Plot title
        save_as : str, optional
            Filename to save plot
        show : bool
            Whether to display plot

        Returns
        -------
        plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot data points
        scenarios = df['scenario_name'].unique()
        for scenario in scenarios:
            scenario_df = df[df['scenario_name'] == scenario]
            ax.scatter(scenario_df[x_param], scenario_df[y_param],
                      label=scenario, alpha=0.7, s=80)

        # Add trend line if enough points
        if len(df) > 2:
            x_data = df[x_param].dropna()
            y_data = df[y_param].dropna()
            mask = x_data.index.isin(y_data.index)
            x_data = x_data[mask]
            y_data = y_data[mask]

            if len(x_data) > 1:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_data.min(), x_data.max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.5,
                       label=f'Linear fit: y={z[0]:.3f}x+{z[1]:.3f}')

        ax.set_xlabel(x_param, fontsize=12)
        ax.set_ylabel(y_param, fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'{y_param} vs {x_param}', fontsize=14)

        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_as:
            save_path = self.output_dir / save_as
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_photon_index_distribution(self,
                                       df: pd.DataFrame,
                                       save_as: Optional[str] = None,
                                       show: bool = True) -> plt.Figure:
        """
        Plot distribution of fitted photon indices.

        Parameters
        ----------
        df : pd.DataFrame
            Results DataFrame
        save_as : str, optional
            Filename to save plot
        show : bool
            Whether to display plot

        Returns
        -------
        plt.Figure
            Figure object
        """
        from config.parameters import COMPPS_PARAMS

        # Create figure with GridSpec for plot + table layout
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 2], wspace=0.1)
        ax_plot = fig.add_subplot(gs[0])
        ax_table = fig.add_subplot(gs[1])

        # Error bar plot by scenario with asymmetric errors
        # Get scenario numbering for consistent labels and sorting
        scenario_map = get_scenario_number_map()

        # Sort scenarios by their number to ensure consistent order
        available_scenarios = df['scenario_name'].unique()
        scenarios = [
            s for s in scenario_map.keys() if s in available_scenarios
        ]

        # Check if asymmetric error columns exist (new deviation format)
        has_errors = (
            'fit_powerlaw.PhoIndex_error_neg' in df.columns and
            'fit_powerlaw.PhoIndex_error_pos' in df.columns
        )

        y_positions = []
        gamma_values = []
        gamma_errors_neg = []
        gamma_errors_pos = []
        scenario_labels = []

        for i, scenario in enumerate(scenarios):
            scenario_df = df[df['scenario_name'] == scenario]
            pho_indices = scenario_df['fit_powerlaw.PhoIndex'].dropna()

            if len(pho_indices) == 0:
                continue

            n_spectra = len(scenario_df)
            # For multiple spectra per scenario, add small offsets
            # For single spectrum (N=1), no offset needed
            offset_increment = 0.08 if n_spectra > 1 else 0

            # For each spectrum in the scenario, plot with error bars
            for j, (_, row) in enumerate(scenario_df.iterrows()):
                if pd.isna(row['fit_powerlaw.PhoIndex']):
                    continue

                # Center multiple points around integer position
                if n_spectra > 1:
                    offset = (j - (n_spectra - 1) / 2) * offset_increment
                else:
                    offset = 0

                y_positions.append(i + 1 + offset)
                gamma_values.append(row['fit_powerlaw.PhoIndex'])

                # Get asymmetric error deviations if available
                if has_errors:
                    nerr = row.get('fit_powerlaw.PhoIndex_error_neg', 0)
                    perr = row.get('fit_powerlaw.PhoIndex_error_pos', 0)
                    gamma_errors_neg.append(nerr if not pd.isna(nerr) else 0)
                    gamma_errors_pos.append(perr if not pd.isna(perr) else 0)
                else:
                    gamma_errors_neg.append(0)
                    gamma_errors_pos.append(0)

            scenario_labels.append(
                f"{scenario_map.get(scenario, '?')}. {scenario}"
            )

        # Plot with asymmetric error bars
        if has_errors:
            ax_plot.errorbar(
                gamma_values, y_positions,
                xerr=[gamma_errors_neg, gamma_errors_pos],
                fmt='o', markersize=3, capsize=4, capthick=2,
                label='90% CI'
            )
            title_suffix = ' (with 90% CI)'
        else:
            ax_plot.scatter(gamma_values, y_positions, s=50)
            title_suffix = ''

        ax_plot.set_xlabel('Photon Index (Γ)', fontsize=12)
        ax_plot.set_ylabel('Scenario', fontsize=12)
        ax_plot.set_title(
            f'Photon Index by Scenario{title_suffix}', fontsize=14
        )
        ax_plot.set_yticks(range(1, len(scenarios) + 1))
        ax_plot.set_yticklabels(scenario_labels)
        ax_plot.invert_yaxis()  # Scenario 1 at top
        ax_plot.grid(True, alpha=0.5, axis='y')
        ax_plot.axvline(1.3, ls='--', color='k', alpha=.4)
        ax_plot.set_xlim(0.4, 4.1)
        # if has_errors:
        #     ax_plot.legend()

        # Create parameter table
        varying_params = self._get_varying_parameters(scenarios)

        # Format parameter values for table
        table_data = []
        for scenario in scenarios:
            row = [f"{scenario_map[scenario]}"]
            for param in varying_params:
                value = COMPPS_PARAMS[scenario][param]
                # Format value based on type and magnitude
                if param == 'kTe':
                    row.append(f"{value:.0f}")
                elif param in ['tau_y', 'cosIncl', 'cov_frac']:
                    row.append(f"{value:.2f}")
                elif param in ['kTbb', 'rel_refl']:
                    row.append(f"{value:.1f}")
                elif param == 'geom':
                    row.append(f"{value:.0f}")
                elif param == 'xi':
                    row.append(f"{value:.0f}")
                elif param == 'Tdisk':
                    row.append(f"{value/1000:.0f}k")
                else:
                    row.append(f"{value:.2g}")
            # Add everything without scenario number
            table_data.append(row[1:])

        # Create table header with abbreviated parameter names
        param_abbrev = {
            'kTe': 'kTe\n(keV)',
            'tau_y': 'τ',
            'kTbb': 'kTbb\n(keV)',
            'geom': 'geom',
            'cosIncl': 'cos i',
            'rel_refl': 'R',
            'xi': 'ξ',
            'Tdisk': 'Tdisk',
            'HovR_cyl': 'H/R',
            'Fe_ab_re': 'Fe',
            'Me_ab': 'Me',
            'Betor10': 'β',
            'Rin': 'Rin',
            'Rout': 'Rout',
            'cov_frac': 'C',
            'EleIndex': 'p',
        }
        # col_labels = ['#'] + [param_abbrev.get(p, p) for p in varying_params]
        col_labels = [param_abbrev.get(p, p) for p in varying_params]

        # Turn off axis for table
        ax_table.axis('off')

        # Create table using matplotlib
        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            bbox=[0, 0.02, 1, 1]
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        # Style header row
        for i in range(len(col_labels)):
            cell = table[(0, i)]
            cell.set_height(0.025)
            cell.set_facecolor('#E0E0E0')
            cell.set_text_props(weight='bold', fontsize=8)

        # Alternate row colors for readability
        for i in range(len(table_data)):
            for j in range(len(col_labels)):
                cell = table[(i + 1, j)]
                if i % 2 != 0:
                    cell.set_facecolor('#dddddd')
                # if i in [4, 6]:
                    # cell.set_facecolor('#ff9fa4b0')

        ax_table.set_title('CompPS Parameters', fontsize=12, pad=20)

        plt.tight_layout()

        if save_as:
            save_path = self.output_dir / save_as
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_photon_index_distribution_by_ktbb(self,
                                               df: pd.DataFrame,
                                               ktbb_values: Optional[List[float]] = None,
                                               output_prefix: str = 'photon_index_dist',
                                               show: bool = False) -> List[str]:
        """
        Plot photon index distribution for each kTbb value separately.

        Creates separate plots for each kTbb value, showing all base scenarios
        at that kTbb value.

        Parameters
        ----------
        df : pd.DataFrame
            Results DataFrame
        ktbb_values : list, optional
            List of kTbb values to plot (default: [0.1, 0.3, 0.5, 0.8])
        output_prefix : str
            Prefix for output filenames (default: 'photon_index_dist')
        show : bool
            Whether to display plots

        Returns
        -------
        list of str
            List of saved plot filenames
        """
        from config.parameters import (
            get_ktbb_scenario_number_map, COMPPS_PARAMS, DEFAULT_KTBB_VALUES
        )

        if ktbb_values is None:
            ktbb_values = DEFAULT_KTBB_VALUES

        saved_plots = []

        for ktbb_val in ktbb_values:
            # Filter dataframe for this kTbb value
            ktbb_str = f"{ktbb_val:.2f}".rstrip('0').rstrip('.')
            pattern = f'_ktbb{ktbb_str}$'
            ktbb_df = df[df['scenario_name'].str.contains(pattern, regex=True, na=False)].copy()

            if ktbb_df.empty:
                print(f"No data found for kTbb = {ktbb_val:.2f}")
                continue

            print(f"Creating photon index distribution plot for kTbb = {ktbb_val:.2f}")

            # Create figure with GridSpec for plot + table layout
            fig = plt.figure(figsize=(14, 10))
            fig.suptitle(f'kTbb = {ktbb_val:.2f} keV', fontsize=16, weight='bold', y=0.995)
            gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 2], wspace=0.1)
            ax_plot = fig.add_subplot(gs[0])
            ax_table = fig.add_subplot(gs[1])

            # Get scenario numbering
            scenario_map = get_ktbb_scenario_number_map(ktbb_values=ktbb_values)

            # Extract base scenario names (remove kTbb suffix)
            ktbb_df['base_scenario'] = ktbb_df['scenario_name'].str.replace(
                pattern, '', regex=True
            )

            # Get unique base scenarios and sort by their base number
            base_scenarios = ktbb_df['base_scenario'].unique()
            base_map = get_scenario_number_map()
            base_scenarios = sorted(
                base_scenarios,
                key=lambda x: base_map.get(x, 999)
            )

            # Check if asymmetric error columns exist
            has_errors = (
                'fit_powerlaw.PhoIndex_error_neg' in ktbb_df.columns and
                'fit_powerlaw.PhoIndex_error_pos' in ktbb_df.columns
            )

            y_positions = []
            gamma_values = []
            gamma_errors_neg = []
            gamma_errors_pos = []
            scenario_labels = []

            for i, base_scenario in enumerate(base_scenarios):
                scenario_df = ktbb_df[ktbb_df['base_scenario'] == base_scenario]
                pho_indices = scenario_df['fit_powerlaw.PhoIndex'].dropna()

                if len(pho_indices) == 0:
                    continue

                n_spectra = len(scenario_df)
                offset_increment = 0.08 if n_spectra > 1 else 0

                for j, (_, row) in enumerate(scenario_df.iterrows()):
                    if pd.isna(row['fit_powerlaw.PhoIndex']):
                        continue

                    if n_spectra > 1:
                        offset = (j - (n_spectra - 1) / 2) * offset_increment
                    else:
                        offset = 0

                    y_positions.append(i + 1 + offset)
                    gamma_values.append(row['fit_powerlaw.PhoIndex'])

                    if has_errors:
                        nerr = row.get('fit_powerlaw.PhoIndex_error_neg', 0)
                        perr = row.get('fit_powerlaw.PhoIndex_error_pos', 0)
                        gamma_errors_neg.append(nerr if not pd.isna(nerr) else 0)
                        gamma_errors_pos.append(perr if not pd.isna(perr) else 0)
                    else:
                        gamma_errors_neg.append(0)
                        gamma_errors_pos.append(0)

                scenario_labels.append(
                    f"{base_map.get(base_scenario, '?')}. {base_scenario}"
                )

            # Plot with asymmetric error bars
            if has_errors:
                ax_plot.errorbar(
                    gamma_values, y_positions,
                    xerr=[gamma_errors_neg, gamma_errors_pos],
                    fmt='o', markersize=3, capsize=4, capthick=2,
                    label='90% CI'
                )
                title_suffix = ' (with 90% CI)'
            else:
                ax_plot.scatter(gamma_values, y_positions, s=50)
                title_suffix = ''

            ax_plot.set_xlabel('Photon Index (Γ)', fontsize=12)
            ax_plot.set_ylabel('Scenario', fontsize=12)
            ax_plot.set_title(
                f'Photon Index by Scenario{title_suffix}', fontsize=14
            )
            ax_plot.set_yticks(range(1, len(base_scenarios) + 1))
            ax_plot.set_yticklabels(scenario_labels)
            ax_plot.invert_yaxis()
            ax_plot.grid(True, alpha=0.5, axis='y')
            ax_plot.axvline(1.3, ls='--', color='k', alpha=.4)
            ax_plot.set_xlim(0.4, 4.1)

            # Create parameter table for base scenarios
            varying_params = self._get_varying_parameters(base_scenarios)

            table_data = []
            for base_scenario in base_scenarios:
                row = []
                for param in varying_params:
                    value = COMPPS_PARAMS[base_scenario][param]
                    if param == 'kTe':
                        row.append(f"{value:.0f}")
                    elif param in ['tau_y', 'cosIncl', 'cov_frac']:
                        row.append(f"{value:.2f}")
                    elif param in ['kTbb', 'rel_refl']:
                        row.append(f"{value:.1f}")
                    elif param == 'geom':
                        row.append(f"{value:.0f}")
                    elif param == 'xi':
                        row.append(f"{value:.0f}")
                    elif param == 'Tdisk':
                        row.append(f"{value/1000:.0f}k")
                    else:
                        row.append(f"{value:.2g}")
                table_data.append(row)

            param_abbrev = {
                'kTe': 'kTe\n(keV)',
                'tau_y': 'τ',
                'kTbb': 'kTbb\n(keV)',
                'geom': 'geom',
                'cosIncl': 'cos i',
                'rel_refl': 'R',
                'xi': 'ξ',
                'Tdisk': 'Tdisk',
                'HovR_cyl': 'H/R',
                'Fe_ab_re': 'Fe',
                'Me_ab': 'Me',
                'Betor10': 'β',
                'Rin': 'Rin',
                'Rout': 'Rout',
                'cov_frac': 'C',
                'EleIndex': 'p',
            }
            col_labels = [param_abbrev.get(p, p) for p in varying_params]

            ax_table.axis('off')
            table = ax_table.table(
                cellText=table_data,
                colLabels=col_labels,
                cellLoc='center',
                loc='center',
                bbox=[0, 0.02, 1, 1]
            )

            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)

            for i in range(len(col_labels)):
                cell = table[(0, i)]
                cell.set_height(0.025)
                cell.set_facecolor('#E0E0E0')
                cell.set_text_props(weight='bold', fontsize=8)

            for i in range(len(table_data)):
                for j in range(len(col_labels)):
                    cell = table[(i + 1, j)]
                    if i % 2 != 0:
                        cell.set_facecolor('#dddddd')
                    if i in [4, 6]:
                        cell.set_facecolor('#ff9fa4b0')

            ax_table.set_title('CompPS Parameters', fontsize=12, pad=20)

            plt.tight_layout()

            # Save figure
            output_file = f"{output_prefix}_ktbb{ktbb_str}.png"
            save_path = self.output_dir / output_file
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
            saved_plots.append(output_file)

            if show:
                plt.show()
            else:
                plt.close()

        return saved_plots

    def plot_ktbb_vs_photon_index(self,
                                  df: pd.DataFrame,
                                  save_as: Optional[str] = None,
                                  show: bool = True) -> plt.Figure:
        """
        Plot kTbb vs fitted photon index relationship.

        Parameters
        ----------
        df : pd.DataFrame
            Results DataFrame
        save_as : str, optional
            Filename to save plot
        show : bool
            Whether to display plot

        Returns
        -------
        plt.Figure
            Figure object
        """

        # Extract kTbb values from scenario names
        df_ktbb = df.copy()
        df_ktbb['kTbb'] = df_ktbb['scenario_name'].str.extract(
            r'_ktbb([\d.]+)$', expand=False
        ).astype(float)

        # Extract base scenario names
        df_ktbb['base_scenario'] = df_ktbb['scenario_name'].str.replace(
            r'_ktbb[\d.]+$', '', regex=True
        )

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot each base scenario as a separate line
        base_scenarios = df_ktbb['base_scenario'].unique()
        for base_scenario in base_scenarios:
            scenario_df = df_ktbb[df_ktbb['base_scenario'] == base_scenario]
            scenario_df = scenario_df.sort_values('kTbb')

            ax.plot(scenario_df['kTbb'], scenario_df['fit_powerlaw.PhoIndex'],
                   marker='o', label=base_scenario, alpha=0.7)

        ax.set_xlabel('kTbb (keV)', fontsize=12)
        ax.set_ylabel('Photon Index (Γ)', fontsize=12)
        ax.set_title('kTbb vs Fitted Photon Index', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        plt.tight_layout()

        if save_as:
            save_path = self.output_dir / save_as
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_temperature_vs_photon_index(self,
                                        df: pd.DataFrame,
                                        save_as: Optional[str] = None,
                                        show: bool = True) -> plt.Figure:
        """
        Plot electron temperature vs fitted photon index.

        Parameters
        ----------
        df : pd.DataFrame
            Results DataFrame
        save_as : str, optional
            Filename to save plot
        show : bool
            Whether to display plot

        Returns
        -------
        plt.Figure
            Figure object
        """
        return self.plot_parameter_comparison(
            df=df,
            x_param='input_Te',
            y_param='fit_powerlaw.PhoIndex',
            title='Electron Temperature vs Fitted Photon Index',
            save_as=save_as,
            show=show
        )

    def create_summary_plots(self,
                           analyzer: ResultsAnalyzer,
                           prefix: str = "summary") -> List[str]:
        """
        Create a complete set of summary plots.

        Parameters
        ----------
        analyzer : ResultsAnalyzer
            Results analyzer instance
        prefix : str
            Prefix for saved plot filenames

        Returns
        -------
        list
            List of saved plot filenames
        """
        print("Creating summary plots...")

        df = analyzer.load_all_results()

        if df.empty:
            print("No results to plot.")
            return []

        saved_plots = []

        # Photon index distribution
        if 'fit_powerlaw.PhoIndex' in df.columns:
            self.plot_photon_index_distribution(
                df,
                save_as=f"{prefix}_photon_index_dist.png",
                show=False
            )
            saved_plots.append(f"{prefix}_photon_index_dist.png")

        # Temperature vs photon index
        if 'input_Te' in df.columns and 'fit_powerlaw.PhoIndex' in df.columns:
            self.plot_temperature_vs_photon_index(
                df,
                save_as=f"{prefix}_Te_vs_photon_index.png",
                show=False
            )
            saved_plots.append(f"{prefix}_Te_vs_photon_index.png")

        # Optical depth vs photon index
        if 'input_tau' in df.columns and 'fit_powerlaw.PhoIndex' in df.columns:
            self.plot_parameter_comparison(
                df,
                x_param='input_tau',
                y_param='fit_powerlaw.PhoIndex',
                title='Optical Depth vs Fitted Photon Index',
                save_as=f"{prefix}_tau_vs_photon_index.png",
                show=False
            )
            saved_plots.append(f"{prefix}_tau_vs_photon_index.png")

        # Chi-squared distribution
        if 'reduced_chi_squared' in df.columns:
            _, ax = plt.subplots(figsize=self.figsize)
            chi2 = df['reduced_chi_squared'].dropna()
            ax.hist(chi2, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(1.0, color='r', linestyle='--', linewidth=2,
                      label='χ²ᵣ = 1 (ideal)')
            ax.axvline(chi2.mean(), color='g', linestyle='--', linewidth=2,
                      label=f'Mean: {chi2.mean():.2f}')
            ax.set_xlabel('Reduced χ²', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of Fit Quality', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_path = self.output_dir / f"{prefix}_chi_squared.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            saved_plots.append(f"{prefix}_chi_squared.png")

        print(f"Created {len(saved_plots)} summary plots")
        return saved_plots


def plot_compps_models_by_kte(spectra_dir: str,
                              output_dir: str,
                              energy_range: List[float],
                              xspec_plt_en_range: str = "0.01 50 1000 log",
                              logger=None) -> List[str]:
    """
    Plot CompPS model spectra from .xcm files, grouped by kTe value.

    For each kTe value, creates a single plot showing all tau values
    with visually distinguishable colors/linestyles.

    Parameters
    ----------
    spectra_dir : str
        Directory containing .xcm files (pattern: sim_grid_kTe{kTe}_tau{tau}_{timestamp}.xcm)
    output_dir : str
        Directory to save plots (will create plots/ subdirectory)
    xspec_plt_en_range : str
        Energy range string for XSPEC plotting (default: "0.1 50 1000 log")
    logger : logging.Logger, optional
        Logger instance for progress messages

    Returns
    -------
    list of str
        List of paths to saved plot files
    """
    if not XSPEC_AVAILABLE:
        if logger:
            logger.error("PyXspec not available. Cannot generate model plots.")
        return []

    def parse_tau_from_filename(tau_str: str) -> float:
        """
        Parse tau value from filename format back to float.

        Handles 'm' prefix for negative values.
        E.g., "m0.50" → -0.50, "0.50" → 0.50
        """
        if tau_str.startswith('m'):
            return -float(tau_str[1:])
        else:
            return float(tau_str)

    spectra_path = Path(spectra_dir)
    output_path = Path(output_dir)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info("")
        logger.info("=" * 70)
        logger.info("GENERATING COMPPS MODEL PLOTS")
        logger.info("=" * 70)
        logger.info(f"Spectra directory: {spectra_dir}")
        logger.info(f"Output directory: {plots_dir}")
        logger.info(f"Energy range: {xspec_plt_en_range}")

    # Find all .xcm files matching the pattern
    # Pattern: sim_grid_kTe{kTe:.0f}_tau{tau_str}_{timestamp}.xcm
    xcm_pattern = re.compile(r'sim_grid_kTe(\d+)_tau([\dm\d.]+)_\d{8}_\d{6}\.xcm')
    xcm_files = []

    for xcm_file in spectra_path.glob('sim_grid_kTe*.xcm'):
        match = xcm_pattern.match(xcm_file.name)
        if match:
            kTe_str = match.group(1)
            tau_str = match.group(2)
            try:
                kTe = float(kTe_str)
                tau = parse_tau_from_filename(tau_str)
                xcm_files.append((str(xcm_file), kTe, tau))
            except (ValueError, AttributeError) as e:
                if logger:
                    logger.warning(f"  Could not parse filename: {xcm_file.name} ({e})")
                continue

    if not xcm_files:
        if logger:
            logger.warning("  No matching .xcm files found")
        return []

    # Group files by kTe value
    files_by_kte: Dict[float, List[Tuple[str, float]]] = {}
    for xcm_file, kTe, tau in xcm_files:
        if kTe not in files_by_kte:
            files_by_kte[kTe] = []
        files_by_kte[kTe].append((xcm_file, tau))

    # Sort tau values for each kTe group
    for kTe in files_by_kte:
        files_by_kte[kTe].sort(key=lambda x: x[1])

    if logger:
        logger.info(f"  Found {len(xcm_files)} .xcm files")
        logger.info(f"  Grouped into {len(files_by_kte)} kTe values")

    saved_plots = []

    # Use ChangeDir to ensure correct working directory for XSPEC
    with ChangeDir(Path(__file__).parent.parent):
        for kTe in sorted(files_by_kte.keys()):
            tau_files = files_by_kte[kTe]

            if logger:
                logger.info(f"  Plotting kTe = {kTe:.0f} keV ({len(tau_files)} tau values)")

            # Create figure
            _, ax = plt.subplots(figsize=(8, 5))

            # Use colormap for different tau values
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(tau_files)))

            for (xcm_file, tau), color in zip(tau_files, colors):
                try:
                    # Clear and restore XSPEC session
                    AllData.clear()
                    AllModels.clear()
                    Xset.restore(xcm_file)

                    # Set energy range and resolution for plotting
                    AllModels.setEnergies(xspec_plt_en_range)

                    # Generate the plot data with model components
                    Plot("eemo")  # E: Energy, E: Energy, M: Model, O: (No data)

                    # Extract the energy grid and total model flux
                    en = list(map(float, Plot.x()))
                    total = list(map(float, Plot.model()))  # Total model

                    # Format tau label
                    tau_label = f"y = {-tau:.2f}"

                    # Plot model
                    ax.plot(
                        en, total, alpha=0.8, color=color, lw=2, ls='-' #, label=tau_label
                        )

                except Exception as e:
                    if logger:
                        logger.warning(f"    Failed to plot {Path(xcm_file).name}: {e}")
                    continue

            # Extract model parameters from the first file (they should be the same for all)
            if tau_files:
                try:
                    if logger:
                        logger.info("Extracting model parameters to place them on the plot...")

                    AllData.clear()
                    AllModels.clear()
                    Xset.restore(tau_files[0][0])  # Load first file to get parameters

                    model = AllModels(1)
                    compPS = model.compPS

                    # Extract key parameters
                    params = {
                        # 'kTe': compPS.kTe.values[0],
                        'kTbb': compPS.kTbb.values[0],
                        'geom': compPS.geom.values[0],
                        'cosIncl': compPS.cosIncl.values[0],
                        'rel_refl': compPS.rel_refl.values[0]  # Add this line
                    }

                    # Create parameter text
                    param_text = (
                        # f"kTe = {params['kTe']:.1f} keV\n"
                        f"kTbb = {params['kTbb']:.3f} keV\n"
                        f"geom = {params['geom']:.0f}\n"
                        f"cosIncl = {params['cosIncl']:.2f}\n"
                    )

                    # Add text box to plot
                    ax.text(0.02, 0.02, param_text,
                            transform=ax.transAxes,
                            fontsize=9,
                            verticalalignment='bottom',
                            horizontalalignment='left',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                            family='monospace')
                except Exception as e:
                    if logger:
                        logger.warning(f"    Could not extract model parameters: {e}")

            # Create a separate Black Body model
            AllModels.clear()
            Model('bbody')

            # Set the energy range for plotting
            AllModels.setEnergies('0.01 50 1000 log')

            # Get the black body component
            bb = AllModels(1).bbody

            # Set black body parameters
            # kT is in keV, norm is the normalization
            bb.kT = 1e-2  # Temperature in keV
            bb.norm = 0.12e-4  # Normalization

            # Generate the plot data
            Plot("eemo")

            # Extract energy and flux
            en_bb = list(map(float, Plot.x()))
            flux_bb = list(map(float, Plot.model()))
            ax.plot(en_bb, flux_bb, label='Black Body', alpha=.8, color='red', lw=2, ls='-')

            # Format plot
            ax.set_xlabel('Energy (keV)', fontsize=14)
            ax.set_ylabel('Model Flux', fontsize=14)
            rel_refl = params['rel_refl']
            ax.set_title(
                (
                    f'CompPS Models: kTe = {kTe:.0f} keV, '
                    f'rel_refl: {rel_refl}'
                    '\ny-parameter from 0.1 to 1.5'
                ),
                fontsize=16, pad=20
            )
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='upper left', frameon=True)
            # ax.axvline(2, color='k', ls='--')
            # ax.axvline(7, color='k', ls='--')

            ax.set_xlim(1e-2, 50)
            ax.set_ylim(1e-5, 1e-3)

            # Save plot
            output_file = plots_dir / f"model_plot_kTe{kTe:.0f}.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            saved_plots.append(str(output_file))
            if logger:
                logger.info(f"    Saved: {output_file.name}")

    if logger:
        logger.info("")
        logger.info("-" * 70)
        logger.info(f"MODEL PLOTTING SUMMARY")
        logger.info("-" * 70)
        logger.info(f"  Generated {len(saved_plots)} plots")
        logger.info(f"  Output directory: {plots_dir}")

    return saved_plots
