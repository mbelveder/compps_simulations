"""
Spectrum simulator module.

This module handles the generation of fake X-ray spectra using
the CompPS model and XSPEC's fakeit functionality.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

from .xspec_interface import XspecSession, validate_response_files


class SpectrumSimulator:
    """Simulate X-ray spectra using CompPS model."""

    def __init__(self,
                 arf_file: str,
                 rmf_file: str,
                 output_dir: str = "data/simulated",
                 response_dir: str = "data/response"):
        """
        Initialize spectrum simulator.

        Parameters
        ----------
        arf_file : str
            Name of ARF file (should be in response_dir)
        rmf_file : str
            Name of RMF file (should be in response_dir)
        output_dir : str
            Directory for simulated spectra
        response_dir : str
            Directory containing response files
        """
        self.response_dir = Path(response_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Full paths to response files
        self.arf_file = str(self.response_dir / arf_file)
        self.rmf_file = str(self.response_dir / rmf_file)

        # Validate response files exist
        if not validate_response_files(self.arf_file, self.rmf_file):
            raise FileNotFoundError(
                f"Response files not found in {self.response_dir}"
            )

        self.xspec = XspecSession()
        self.simulation_log = []

    def group_spectrum(self,
                      spectrum_file: str,
                      background_file: Optional[str] = None,
                      grouptype: str = "min",
                      groupscale: int = 3) -> str:
        """
        Group spectrum using ftgrouppha.

        Parameters
        ----------
        spectrum_file : str
            Path to spectrum file to group
        background_file : str, optional
            Background file to associate
        grouptype : str
            Grouping type (default: 'min')
        groupscale : int
            Grouping scale (default: 3)

        Returns
        -------
        str
            Path to grouped spectrum file
        """
        spec_path = Path(spectrum_file)
        grouped_file = spec_path.parent / f"{spec_path.stem}_grp{spec_path.suffix}"

        # Build ftgrouppha command
        cmd = [
            "ftgrouppha",
            f"infile={spectrum_file}",
            f"outfile={grouped_file}",
            f"grouptype={grouptype}",
            f"groupscale={groupscale}",
            "clobber=yes"
        ]

        # Add background file if specified
        if background_file:
            cmd.append(f"backfile={background_file}")

        print(f"  Grouping spectrum with ftgrouppha...")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"  Grouped spectrum saved: {grouped_file}")
            return str(grouped_file)
        except subprocess.CalledProcessError as e:
            print(f"  Warning: ftgrouppha failed: {e.stderr}")
            print(f"  Continuing with ungrouped spectrum: {spectrum_file}")
            return spectrum_file
        except FileNotFoundError:
            print(f"  Warning: ftgrouppha not found in PATH")
            print(f"  Continuing with ungrouped spectrum: {spectrum_file}")
            return spectrum_file

    def simulate_spectrum(self,
                         scenario_name: str,
                         compps_params: Dict[str, float],
                         exposure: float = 10000.0,
                         norm: float = 1.0,
                         background: Optional[str] = None
                         ) -> Tuple[str, Optional[float]]:
        """
        Simulate a single spectrum with CompPS model.

        Parameters
        ----------
        scenario_name : str
            Name for this scenario (used in filename)
        compps_params : dict
            CompPS model parameters
        exposure : float
            Exposure time in seconds
        norm : float
            Model normalization
        background : str, optional
            Background file path

        Returns
        -------
        str
            Path to generated spectrum file
        """
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"sim_{scenario_name}_{timestamp}.pha"

        print(f"Simulating spectrum: {scenario_name}")
        print(f"  Output: {output_file}")
        print(f"  Exposure: {exposure:.1f} s")
        print(f"  Normalization: {norm:.2e}")

        # Clear previous session
        self.xspec.clear_session()

        # Set up CompPS model
        model = self.xspec.setup_compps_model(compps_params, norm=norm)

        # Save .xcm file for this model
        xcm_file = output_file.with_suffix('.xcm')
        self.xspec.save_xcm_file(str(xcm_file))
        print(f"  Model saved to: {xcm_file}")

        # Generate fake spectrum
        self.xspec.generate_fake_spectrum(
            arf_file=self.arf_file,
            rmf_file=self.rmf_file,
            output_file=str(output_file),
            model=model,
            exposure=exposure,
            background=background
        )

        # Compute amplification factor while CompPS model is still active
        amplification_factor = self.xspec.compute_amplification_factor(model)
        if amplification_factor is not None:
            print(f"  Amplification factor: A = {amplification_factor:.4f}")
        else:
            print("  Amplification factor: computation failed")

        # Group the spectrum for fitting
        grouped_file = self.group_spectrum(
            spectrum_file=str(output_file),
            background_file=background,
            grouptype="min",
            groupscale=3
        )

        # Save simulation metadata
        metadata = {
            'scenario_name': scenario_name,
            'timestamp': timestamp,
            'spectrum_file': str(output_file),
            'grouped_spectrum_file': grouped_file,
            'arf_file': self.arf_file,
            'rmf_file': self.rmf_file,
            'exposure': exposure,
            'normalization': norm,
            'compps_parameters': compps_params,
            'background': background,
            'amplification_factor': amplification_factor,
        }

        # Save metadata to JSON
        metadata_file = output_file.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Add to simulation log
        self.simulation_log.append(metadata)

        print(f"  Simulation complete: {grouped_file}")
        return grouped_file, amplification_factor

    def simulate_multiple(self,
                         scenarios: Dict[str, Dict[str, float]],
                         exposure: float = 10000.0,
                         norm: float = 1.0) -> List[str]:
        """
        Simulate multiple spectra from different scenarios.

        Parameters
        ----------
        scenarios : dict
            Dictionary of {scenario_name: compps_params}
        exposure : float
            Exposure time in seconds
        norm : float
            Model normalization

        Returns
        -------
        list
            List of paths to generated spectrum files
        """
        spectrum_files = []

        print(f"\nSimulating {len(scenarios)} scenarios...")
        print("=" * 60)

        for i, (scenario_name, params) in enumerate(scenarios.items(), 1):
            print(f"\n[{i}/{len(scenarios)}] Scenario: {scenario_name}")
            try:
                spectrum_file, _ = self.simulate_spectrum(
                    scenario_name=scenario_name,
                    compps_params=params,
                    exposure=exposure,
                    norm=norm
                )
                spectrum_files.append(spectrum_file)
            except Exception as e:
                print(f"  ERROR: Failed to simulate {scenario_name}: {e}")
                continue

        print("\n" + "=" * 60)
        print(f"Simulation complete: {len(spectrum_files)}/{len(scenarios)} successful")

        return spectrum_files

    def save_simulation_log(self, filename: str = "simulation_log.json"):
        """
        Save simulation log to file.

        Parameters
        ----------
        filename : str
            Log filename
        """
        log_file = self.output_dir / filename
        with open(log_file, 'w') as f:
            json.dump(self.simulation_log, f, indent=2)
        print(f"Simulation log saved to: {log_file}")

    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Get summary of simulations performed.

        Returns
        -------
        dict
            Summary statistics
        """
        if not self.simulation_log:
            return {'num_simulations': 0}

        exposures = [s['exposure'] for s in self.simulation_log]
        norms = [s['normalization'] for s in self.simulation_log]

        summary = {
            'num_simulations': len(self.simulation_log),
            'scenarios': [s['scenario_name'] for s in self.simulation_log],
            'arf_file': self.arf_file,
            'rmf_file': self.rmf_file,
            'exposure_range': (min(exposures), max(exposures)),
            'norm_range': (min(norms), max(norms)),
            'output_directory': str(self.output_dir)
        }

        return summary


def load_simulation_metadata(spectrum_file: str) -> Dict[str, Any]:
    """
    Load simulation metadata from JSON file.

    Parameters
    ----------
    spectrum_file : str
        Path to spectrum file (can be grouped or ungrouped)

    Returns
    -------
    dict
        Simulation metadata
    """
    spec_path = Path(spectrum_file)
    
    # If this is a grouped spectrum, look for the ungrouped metadata
    if '_grp' in spec_path.stem:
        # Remove _grp suffix to get original filename
        original_stem = spec_path.stem.replace('_grp', '')
        metadata_file = spec_path.parent / f"{original_stem}.json"
    else:
        metadata_file = spec_path.with_suffix('.json')

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return metadata


def find_simulated_spectra(simulated_dir: str = "data/simulated",
                           scenario_pattern: Optional[str] = None,
                           prefer_grouped: bool = True) -> List[str]:
    """
    Find simulated spectrum files.

    Parameters
    ----------
    simulated_dir : str
        Directory containing simulated spectra
    scenario_pattern : str, optional
        Pattern to match scenario names (e.g., "basic_*")
    prefer_grouped : bool
        If True, prefer grouped spectra over ungrouped (default: True)

    Returns
    -------
    list
        List of spectrum file paths
    """
    simulated_path = Path(simulated_dir)

    if not simulated_path.exists():
        return []

    if scenario_pattern:
        pattern = f"sim_{scenario_pattern}*.pha"
    else:
        pattern = "sim_*.pha"

    spectrum_files = sorted(simulated_path.glob(pattern))
    
    if prefer_grouped:
        # Filter to prefer grouped files (_grp.pha)
        # For each base spectrum, use grouped version if it exists
        grouped_files = [f for f in spectrum_files if '_grp' in f.name]
        if grouped_files:
            return [str(f) for f in grouped_files]
    
    return [str(f) for f in spectrum_files]

