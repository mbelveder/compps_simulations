"""
Plotting module for visualization of spectra and analysis results.

This module provides functions to plot simulated spectra, fitted models,
and analysis results.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from astropy.io import fits

from .analysis import ResultsAnalyzer


class SpectrumPlotter:
    """Create plots for spectra and analysis results."""

    def __init__(self,
                 output_dir: str = "data/results",
                 figsize: Tuple[int, int] = (10, 6),
                 dpi: int = 100):
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

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize,
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
        ax2.step(energy, counts, where='mid', linewidth=1.5, color='C1')
        ax2.set_xlabel('Energy (keV)', fontsize=12)
        ax2.set_ylabel('Counts', fontsize=12)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        photon_indices = df['fit_powerlaw.PhoIndex'].dropna()

        # Histogram
        ax1.hist(photon_indices, bins=20, alpha=0.7, edgecolor='black')
        ax1.axvline(photon_indices.mean(), color='r', linestyle='--',
                   linewidth=2, label=f'Mean: {photon_indices.mean():.3f}')
        ax1.axvline(photon_indices.median(), color='g', linestyle='--',
                   linewidth=2, label=f'Median: {photon_indices.median():.3f}')
        ax1.set_xlabel('Photon Index (Γ)', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Distribution of Fitted Photon Indices', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot by scenario
        scenarios = df['scenario_name'].unique()
        scenario_data = [df[df['scenario_name'] == s]['fit_powerlaw.PhoIndex'].dropna()
                        for s in scenarios]

        ax2.boxplot(scenario_data, labels=scenarios)
        ax2.set_xlabel('Scenario', fontsize=12)
        ax2.set_ylabel('Photon Index (Γ)', fontsize=12)
        ax2.set_title('Photon Index by Scenario', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

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
            fig, ax = plt.subplots(figsize=self.figsize)
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

