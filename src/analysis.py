"""
Analysis module for comparing simulation inputs and fit results.

This module provides tools to analyze fit results, compare them with
input CompPS parameters, and generate statistical summaries.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from .simulator import load_simulation_metadata
from .fitter import load_fit_result, find_fit_results


class ResultsAnalyzer:
    """Analyze and compare simulation and fitting results."""

    def __init__(self,
                 simulated_dir: str = "data/simulated",
                 results_dir: str = "data/results"):
        """
        Initialize results analyzer.

        Parameters
        ----------
        simulated_dir : str
            Directory containing simulated spectra
        results_dir : str
            Directory containing fit results
        """
        self.simulated_dir = Path(simulated_dir)
        self.results_dir = Path(results_dir)

    def load_all_results(self) -> pd.DataFrame:
        """
        Load all simulation and fit results into a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Combined results with both input parameters and fit results
        """
        result_files = find_fit_results(str(self.results_dir))

        if not result_files:
            print("No fit results found.")
            return pd.DataFrame()

        combined_data = []

        for result_file in result_files:
            try:
                # Load fit result
                fit_result = load_fit_result(result_file)

                if fit_result.get('fit_status') != 'success':
                    continue

                # Get corresponding spectrum file
                spectrum_file = fit_result['spectrum_file']

                # Load simulation metadata
                try:
                    sim_metadata = load_simulation_metadata(spectrum_file)
                except FileNotFoundError:
                    print(f"Warning: Metadata not found for {spectrum_file}")
                    continue

                # Combine into single record
                record = {
                    'scenario_name': sim_metadata['scenario_name'],
                    'spectrum_file': spectrum_file,
                    'timestamp': fit_result['timestamp'],
                }

                # Add input CompPS parameters
                for param, value in sim_metadata['compps_parameters'].items():
                    record[f'input_{param}'] = value

                # Add simulation settings
                record['input_exposure'] = sim_metadata['exposure']
                record['input_normalization'] = sim_metadata['normalization']

                # Add fit parameters
                for param, value in fit_result.get('parameters', {}).items():
                    record[f'fit_{param}'] = value

                # Add fit statistics
                record['chi_squared'] = fit_result.get('chi_squared')
                record['dof'] = fit_result.get('dof')
                record['reduced_chi_squared'] = fit_result.get('reduced_chi_squared')
                record['flux'] = fit_result.get('flux')
                record['flux_error'] = fit_result.get('flux_error')

                combined_data.append(record)

            except Exception as e:
                print(f"Error processing {result_file}: {e}")
                continue

        df = pd.DataFrame(combined_data)
        return df

    def compare_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare input CompPS parameters with fitted powerlaw parameters.

        Parameters
        ----------
        df : pd.DataFrame
            Combined results DataFrame

        Returns
        -------
        pd.DataFrame
            Comparison statistics
        """
        if df.empty:
            return pd.DataFrame()

        comparison = pd.DataFrame()

        # Group by scenario
        for scenario in df['scenario_name'].unique():
            scenario_df = df[df['scenario_name'] == scenario]

            # Get input CompPS parameters (average if multiple)
            scenario_stats = {
                'scenario': scenario,
                'n_spectra': len(scenario_df),
            }

            # Key CompPS parameters
            compps_params = ['kTe', 'tau_y', 'kTbb', 'geom', 'cosIncl']
            for param in compps_params:
                col = f'input_{param}'
                if col in scenario_df.columns:
                    scenario_stats[f'compps_{param}'] = scenario_df[col].iloc[0]

            # Fitted powerlaw parameters
            if 'fit_powerlaw.PhoIndex' in scenario_df.columns:
                photon_indices = scenario_df['fit_powerlaw.PhoIndex'].dropna()
                scenario_stats['photon_index_mean'] = photon_indices.mean()
                scenario_stats['photon_index_std'] = photon_indices.std()
                scenario_stats['photon_index_min'] = photon_indices.min()
                scenario_stats['photon_index_max'] = photon_indices.max()

            if 'fit_tbabs.nH' in scenario_df.columns:
                nh_values = scenario_df['fit_tbabs.nH'].dropna()
                scenario_stats['nH_mean'] = nh_values.mean()
                scenario_stats['nH_std'] = nh_values.std()

            if 'reduced_chi_squared' in scenario_df.columns:
                chi2 = scenario_df['reduced_chi_squared'].dropna()
                scenario_stats['chi2_mean'] = chi2.mean()
                scenario_stats['chi2_std'] = chi2.std()

            comparison = pd.concat([comparison, pd.DataFrame([scenario_stats])],
                                  ignore_index=True)

        return comparison

    def calculate_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlations between input parameters and fit results.

        Parameters
        ----------
        df : pd.DataFrame
            Combined results DataFrame

        Returns
        -------
        pd.DataFrame
            Correlation matrix
        """
        if df.empty:
            return pd.DataFrame()

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Focus on input CompPS parameters and fit results
        input_cols = [col for col in numeric_cols if col.startswith('input_')]
        fit_cols = [col for col in numeric_cols if col.startswith('fit_') or
                   col in ['chi_squared', 'reduced_chi_squared', 'flux']]

        relevant_cols = input_cols + fit_cols

        if not relevant_cols:
            return pd.DataFrame()

        correlation_matrix = df[relevant_cols].corr()
        return correlation_matrix

    def analyze_photon_index_vs_temperature(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze relationship between electron temperature and fitted photon index.

        Parameters
        ----------
        df : pd.DataFrame
            Combined results DataFrame

        Returns
        -------
        dict
            Analysis results
        """
        if df.empty or 'input_Te' not in df.columns or 'fit_powerlaw.PhoIndex' not in df.columns:
            return {}

        te_values = df['input_Te'].values
        photon_indices = df['fit_powerlaw.PhoIndex'].values

        # Remove NaN values
        mask = ~(np.isnan(te_values) | np.isnan(photon_indices))
        te_values = te_values[mask]
        photon_indices = photon_indices[mask]

        if len(te_values) < 2:
            return {}

        # Calculate correlation
        correlation = np.corrcoef(te_values, photon_indices)[0, 1]

        # Fit linear relationship
        coeffs = np.polyfit(te_values, photon_indices, 1)
        slope, intercept = coeffs

        analysis = {
            'correlation': correlation,
            'linear_fit': {
                'slope': slope,
                'intercept': intercept,
                'equation': f"Γ = {slope:.4f} * Te + {intercept:.4f}"
            },
            'n_points': len(te_values),
            'Te_range': (te_values.min(), te_values.max()),
            'photon_index_range': (photon_indices.min(), photon_indices.max())
        }

        return analysis

    def generate_summary_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive summary report.

        Parameters
        ----------
        output_file : str, optional
            If provided, save report to this file

        Returns
        -------
        str
            Report text
        """
        df = self.load_all_results()

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("CompPS SIMULATION AND FITTING ANALYSIS REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")

        if df.empty:
            report_lines.append("No results found.")
            report = "\n".join(report_lines)
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(report)
            return report

        report_lines.append(f"Total number of successful fits: {len(df)}")
        report_lines.append(f"Number of scenarios: {df['scenario_name'].nunique()}")
        report_lines.append("")

        # Summary by scenario
        report_lines.append("SUMMARY BY SCENARIO")
        report_lines.append("-" * 70)

        comparison = self.compare_parameters(df)
        for _, row in comparison.iterrows():
            report_lines.append(f"\nScenario: {row['scenario']}")
            report_lines.append(f"  Number of spectra: {row['n_spectra']:.0f}")

            if 'compps_Te' in row:
                report_lines.append(f"  Input Te: {row['compps_Te']:.1f} keV")
            if 'compps_tau' in row:
                report_lines.append(f"  Input tau: {row['compps_tau']:.2f}")

            if 'photon_index_mean' in row and not np.isnan(row['photon_index_mean']):
                report_lines.append(
                    f"  Fitted Photon Index: {row['photon_index_mean']:.3f} "
                    f"± {row['photon_index_std']:.3f}"
                )

            if 'nH_mean' in row and not np.isnan(row['nH_mean']):
                report_lines.append(
                    f"  Fitted nH: {row['nH_mean']:.4f} ± {row['nH_std']:.4f} "
                    f"x 10^22 cm^-2"
                )

            if 'chi2_mean' in row and not np.isnan(row['chi2_mean']):
                report_lines.append(
                    f"  Reduced chi-squared: {row['chi2_mean']:.3f} "
                    f"± {row['chi2_std']:.3f}"
                )

        # Temperature vs photon index analysis
        report_lines.append("\n" + "-" * 70)
        report_lines.append("TEMPERATURE VS PHOTON INDEX ANALYSIS")
        report_lines.append("-" * 70)

        te_analysis = self.analyze_photon_index_vs_temperature(df)
        if te_analysis:
            report_lines.append(f"Correlation coefficient: {te_analysis['correlation']:.3f}")
            report_lines.append(f"Linear fit: {te_analysis['linear_fit']['equation']}")
            report_lines.append(
                f"Temperature range: {te_analysis['Te_range'][0]:.1f} - "
                f"{te_analysis['Te_range'][1]:.1f} keV"
            )
            report_lines.append(
                f"Photon index range: {te_analysis['photon_index_range'][0]:.3f} - "
                f"{te_analysis['photon_index_range'][1]:.3f}"
            )
        else:
            report_lines.append("Insufficient data for analysis")

        report_lines.append("\n" + "=" * 70)

        report = "\n".join(report_lines)

        if output_file:
            output_path = self.results_dir / output_file
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")

        return report

    def export_to_csv(self, filename: str = "combined_results.csv"):
        """
        Export combined results to CSV file.

        Parameters
        ----------
        filename : str
            Output CSV filename
        """
        df = self.load_all_results()

        if df.empty:
            print("No results to export.")
            return

        output_file = self.results_dir / filename
        df.to_csv(output_file, index=False)
        print(f"Results exported to: {output_file}")

    def export_comparison_to_csv(self, filename: str = "parameter_comparison.csv"):
        """
        Export parameter comparison to CSV file.

        Parameters
        ----------
        filename : str
            Output CSV filename
        """
        df = self.load_all_results()

        if df.empty:
            print("No results to export.")
            return

        comparison = self.compare_parameters(df)
        output_file = self.results_dir / filename
        comparison.to_csv(output_file, index=False)
        print(f"Comparison exported to: {output_file}")

