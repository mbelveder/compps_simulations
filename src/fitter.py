"""
Spectrum fitting module.

This module handles fitting of simulated spectra using an
absorbed powerlaw model (tbabs*powerlaw).
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

from .xspec_interface import XspecSession


class SpectrumFitter:
    """Fit X-ray spectra with absorbed powerlaw model."""

    def __init__(self,
                 output_dir: str = "data/results",
                 energy_range: Tuple[float, float] = (0.2, 10.0),
                 stat_method: str = "cstat"):
        """
        Initialize spectrum fitter.

        Parameters
        ----------
        output_dir : str
            Directory for fit results
        energy_range : tuple
            Energy range for fitting (min, max) in keV
        stat_method : str
            Fit statistic method (default: 'cstat' for C-statistic)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.energy_min, self.energy_max = energy_range
        self.stat_method = stat_method
        self.xspec = XspecSession()
        self.fit_log = []

    def fit_spectrum(self,
                    spectrum_file: str,
                    nh_init: float = 0.01,
                    photon_index_init: float = 2.0,
                    norm_init: float = 1e-3,
                    max_iterations: int = 1000,
                    energy_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Fit a single spectrum with absorbed powerlaw.

        Parameters
        ----------
        spectrum_file : str
            Path to spectrum file (preferably grouped)
        nh_init : float
            Initial nH value (10^22 cm^-2)
        photon_index_init : float
            Initial photon index
        norm_init : float
            Initial normalization
        max_iterations : int
            Maximum fit iterations (default: 1000)
        energy_range : tuple, optional
            Override default energy range for fitting

        Returns
        -------
        dict
            Fit results
        """
        spectrum_path = Path(spectrum_file)
        if not spectrum_path.exists():
            raise FileNotFoundError(f"Spectrum file not found: {spectrum_file}")

        print(f"Fitting spectrum: {spectrum_path.name}")

        # Set energy range for fitting
        if energy_range is None:
            energy_range = (self.energy_min, self.energy_max)

        try:
            import xspec
            
            # Clear previous data and models
            xspec.AllModels.clear()
            xspec.AllData.clear()

            # Load grouped spectrum (has all necessary file links in header)
            spectrum = xspec.Spectrum(spectrum_file)
            print(f"  Loaded spectrum: {spectrum_path.name}")

            # Set up absorbed powerlaw model (tbabs*powerlaw)
            model = self.xspec.setup_absorbed_powerlaw(
                nh=nh_init,
                photon_index=photon_index_init,
                norm=norm_init
            )
            print("  Model defined: tbabs*powerlaw")

            # Ignore ranges outside specified range and bad channels
            spectrum.ignore(f"**-{energy_range[0]} {energy_range[1]}-**")
            spectrum.ignore("bad")
            emin, emax = energy_range
            print(f"  Ignoring outside {emin}-{emax} keV and bad channels")

            # Set fit parameters
            xspec.Fit.nIterations = max_iterations
            xspec.Fit.statMethod = self.stat_method
            xspec.Fit.query = "yes"  # Non-interactive mode
            print(f"  Settings: iterations={max_iterations}, "
                  f"statistic={self.stat_method}")

            # Perform fit
            print("  Performing fit...")
            xspec.Fit.perform()
            print("  Fit complete")

            # Save XSPEC session to .xcm file
            xcm_file = self._save_xspec_session(spectrum_file)
            print(f"  Saved XSPEC session: {Path(xcm_file).name}")

            # Extract fit results
            results = self._extract_fit_results(model, spectrum_file, energy_range)
            results['xcm_file'] = xcm_file

            # Print summary
            stat_name = ("C-statistic" if self.stat_method == "cstat"
                        else "Chi-squared")
            print(f"  {stat_name}: {results['statistic']:.2f}")
            if results.get('reduced_chi_squared') is not None:
                red_stat = results['reduced_chi_squared']
                print(f"  Reduced statistic: {red_stat:.3f}")
            if 'TBabs.nH' in results['parameters']:
                nh = results['parameters']['TBabs.nH']
                print(f"  nH: {nh:.4f} x 10^22 cm^-2")
            if 'powerlaw.PhoIndex' in results['parameters']:
                gamma = results['parameters']['powerlaw.PhoIndex']
                print(f"  Photon Index: {gamma:.3f}")
            if 'powerlaw.norm' in results['parameters']:
                norm = results['parameters']['powerlaw.norm']
                print(f"  Normalization: {norm:.3e}")

        except Exception as e:
            print(f"  ERROR: Fit failed: {e}")
            results = {
                'spectrum_file': str(spectrum_file),
                'fit_status': 'failed',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }

        # Add to fit log
        self.fit_log.append(results)

        return results

    def _extract_fit_results(self, model, spectrum_file: str,
                            energy_range: Tuple[float, float]) -> Dict[str, Any]:
        """
        Extract fit results from XSPEC model.

        Parameters
        ----------
        model : xspec.Model
            Fitted model
        spectrum_file : str
            Path to spectrum file
        energy_range : tuple
            Energy range used for fitting

        Returns
        -------
        dict
            Fit results dictionary
        """
        import xspec

        results = {
            'statistic': xspec.Fit.statistic,
            'dof': xspec.Fit.dof,
            'stat_method': self.stat_method,
            'parameters': {},
            'errors': {}
        }

        # Calculate reduced statistic
        if xspec.Fit.dof > 0:
            results['reduced_chi_squared'] = xspec.Fit.statistic / xspec.Fit.dof
        else:
            results['reduced_chi_squared'] = None

        # Extract parameter values and errors
        for comp in model.componentNames:
            component = getattr(model, comp)
            for param_name in component.parameterNames:
                param = getattr(component, param_name)
                full_name = f"{comp}.{param_name}"
                results['parameters'][full_name] = param.values[0]

                # Error is stored as (lower, upper) bounds
                if hasattr(param, 'error'):
                    results['errors'][full_name] = param.error

        # Calculate flux
        try:
            flux, flux_err = self.xspec.get_model_flux(
                energy_min=energy_range[0],
                energy_max=energy_range[1]
            )
            results['flux'] = flux
            results['flux_error'] = flux_err
        except Exception as e:
            print(f"  Warning: Flux calculation failed: {e}")
            results['flux'] = None
            results['flux_error'] = None

        # Add metadata
        results['spectrum_file'] = str(spectrum_file)
        results['energy_range'] = energy_range
        results['fit_status'] = 'success'
        results['timestamp'] = datetime.now().isoformat()

        return results

    def _save_xspec_session(self, spectrum_file: str) -> str:
        """
        Save XSPEC session to .xcm file with additional commands.

        Parameters
        ----------
        spectrum_file : str
            Path to spectrum file (used to generate xcm filename)

        Returns
        -------
        str
            Path to saved .xcm file
        """
        import xspec

        spectrum_path = Path(spectrum_file)
        xcm_filename = f"fit_{spectrum_path.stem}.xcm"
        xcm_file = self.output_dir / xcm_filename

        # Save the current XSPEC session
        xspec.Xset.save(str(xcm_file), info='a')

        # Append additional commands to make the session interactive-ready
        with open(xcm_file, 'a') as f:
            f.write('\n')
            f.write('# Additional commands for interactive use\n')
            f.write('ign bad\n')
            f.write('ign **-0.3 9.0-**\n')
            f.write('setpl en\n')
            f.write('fit\n')
            f.write('cpd /xw\n')
            f.write('setplot r 10 10\n')
            f.write('pl eeufs\n')

        return str(xcm_file)

    def fit_multiple(self,
                    spectrum_files: List[str],
                    nh_init: float = 0.01,
                    photon_index_init: float = 2.0,
                    norm_init: float = 1e-3) -> List[Dict[str, Any]]:
        """
        Fit multiple spectra.

        Parameters
        ----------
        spectrum_files : list
            List of spectrum file paths
        nh_init : float
            Initial nH value
        photon_index_init : float
            Initial photon index
        norm_init : float
            Initial normalization

        Returns
        -------
        list
            List of fit results
        """
        results_list = []

        print(f"\nFitting {len(spectrum_files)} spectra...")
        print("=" * 60)

        for i, spectrum_file in enumerate(spectrum_files, 1):
            print(f"\n[{i}/{len(spectrum_files)}]")
            try:
                results = self.fit_spectrum(
                    spectrum_file=spectrum_file,
                    nh_init=nh_init,
                    photon_index_init=photon_index_init,
                    norm_init=norm_init
                )
                results_list.append(results)

                # Save individual fit result
                self.save_fit_result(results, spectrum_file)

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        print("\n" + "=" * 60)
        successful = sum(1 for r in results_list if r.get('fit_status') == 'success')
        print(f"Fitting complete: {successful}/{len(spectrum_files)} successful")

        return results_list

    def save_fit_result(self, result: Dict[str, Any], spectrum_file: str):
        """
        Save fit result to JSON file.

        Parameters
        ----------
        result : dict
            Fit result dictionary
        spectrum_file : str
            Original spectrum file path
        """
        spectrum_path = Path(spectrum_file)
        result_file = self.output_dir / f"fit_{spectrum_path.stem}.json"

        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

    def save_fit_log(self, filename: str = "fit_log.json"):
        """
        Save all fit results to log file.

        Parameters
        ----------
        filename : str
            Log filename
        """
        log_file = self.output_dir / filename
        with open(log_file, 'w') as f:
            json.dump(self.fit_log, f, indent=2)
        print(f"\nFit log saved to: {log_file}")

    def get_fit_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of fits.

        Returns
        -------
        dict
            Summary statistics
        """
        if not self.fit_log:
            return {'num_fits': 0}

        successful_fits = [f for f in self.fit_log if f.get('fit_status') == 'success']

        if not successful_fits:
            return {
                'num_fits': len(self.fit_log),
                'num_successful': 0,
                'success_rate': 0.0
            }

        # Extract parameter values (handle both tbabs and TBabs)
        photon_indices = [
            f['parameters'].get('powerlaw.PhoIndex')
            for f in successful_fits
            if 'parameters' in f and 'powerlaw.PhoIndex' in f['parameters']
        ]

        nh_values = [
            f['parameters'].get('TBabs.nH', f['parameters'].get('tbabs.nH'))
            for f in successful_fits
            if 'parameters' in f and (
                'TBabs.nH' in f['parameters'] or
                'tbabs.nH' in f['parameters']
            )
        ]

        chi_squared = [
            f['reduced_chi_squared']
            for f in successful_fits
            if f.get('reduced_chi_squared') is not None
        ]

        summary = {
            'num_fits': len(self.fit_log),
            'num_successful': len(successful_fits),
            'success_rate': len(successful_fits) / len(self.fit_log),
            'photon_index': {
                'mean': np.mean(photon_indices) if photon_indices else None,
                'std': np.std(photon_indices) if photon_indices else None,
                'min': np.min(photon_indices) if photon_indices else None,
                'max': np.max(photon_indices) if photon_indices else None,
            },
            'nH': {
                'mean': np.mean(nh_values) if nh_values else None,
                'std': np.std(nh_values) if nh_values else None,
                'min': np.min(nh_values) if nh_values else None,
                'max': np.max(nh_values) if nh_values else None,
            },
            'reduced_chi_squared': {
                'mean': np.mean(chi_squared) if chi_squared else None,
                'std': np.std(chi_squared) if chi_squared else None,
                'min': np.min(chi_squared) if chi_squared else None,
                'max': np.max(chi_squared) if chi_squared else None,
            }
        }

        return summary


def load_fit_result(result_file: str) -> Dict[str, Any]:
    """
    Load fit result from JSON file.

    Parameters
    ----------
    result_file : str
        Path to fit result file

    Returns
    -------
    dict
        Fit result
    """
    with open(result_file, 'r') as f:
        result = json.load(f)
    return result


def find_fit_results(results_dir: str = "data/results",
                    pattern: Optional[str] = None) -> List[str]:
    """
    Find fit result files.

    Parameters
    ----------
    results_dir : str
        Directory containing fit results
    pattern : str, optional
        Pattern to match (e.g., "fit_basic_*")

    Returns
    -------
    list
        List of result file paths
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        return []

    if pattern:
        search_pattern = f"{pattern}.json"
    else:
        search_pattern = "fit_*.json"

    result_files = sorted(results_path.glob(search_pattern))
    return [str(f) for f in result_files if f.stem != "fit_log"]

