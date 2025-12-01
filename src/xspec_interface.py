"""
XSPEC interface module using PyXspec.

This module provides wrapper functions for interacting with XSPEC
through the PyXspec Python bindings.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    import xspec
except ImportError:
    raise ImportError(
        "PyXspec is not installed. Please install XSPEC with Python bindings.\n"
        "Note: PyXspec typically needs to be installed separately from HEASOFT."
    )


# CompPS parameter hard limits (from XSPEC)
COMPPS_LIMITS = {
    'kTe': (20.0, 100000.0),
    'EleIndex': (0.0, 5.0),
    'Gmin': (-1.0, 10.0),
    'Gmax': (10.0, 10000.0),
    'kTbb': (0.001, 10.0),
    'tau_y': (-4.0, 3.0),  # Extended range for y-parameter support (negative = Compton y-parameter)
    'geom': (-5.0, 4.0),
    'HovR_cyl': (0.5, 2.0),
    'cosIncl': (0.05, 0.95),
    'cov_frac': (0.0, 1.0),
    'rel_refl': (0.0, 10000.0),
    'Fe_ab_re': (0.1, 10.0),
    'Me_ab': (0.1, 10.0),
    'xi': (0.0, 100000.0),
    'Tdisk': (10000.0, 1e6),
    'Betor10': (-10.0, 10.0),
    'Rin': (6.001, 10000.0),
    'Rout': (0.0, 1e6),
    'Redshift': (-0.999, 10.0),
}


def validate_compps_params(params: Dict[str, float]) -> Tuple[bool, list]:
    """
    Validate CompPS parameters against hard limits.

    Parameters
    ----------
    params : dict
        Dictionary of CompPS parameters

    Returns
    -------
    tuple
        (is_valid, list_of_errors)
    """
    errors = []
    for param_name, value in params.items():
        if param_name in COMPPS_LIMITS:
            min_val, max_val = COMPPS_LIMITS[param_name]
            if value < min_val or value > max_val:
                errors.append(
                    f"{param_name}={value} is outside valid range [{min_val}, {max_val}]"
                )
    return len(errors) == 0, errors


class XspecSession:
    """Manage an XSPEC session for CompPS simulations."""

    def __init__(self, chatter: int = 0, abund: str = "wilm", xsect: str = "vern"):
        """
        Initialize XSPEC session with default settings.

        Parameters
        ----------
        chatter : int
            XSPEC chatter level (0=quiet, 10=verbose)
        abund : str
            Abundance table (default: 'wilm')
        xsect : str
            Cross-section table (default: 'vern')
        """
        xspec.Xset.chatter = chatter
        xspec.Xset.abund = abund
        xspec.Xset.xsect = xsect
        xspec.AllData.clear()
        xspec.AllModels.clear()

    def setup_compps_model(self, params: Dict[str, float], norm: float = 1.0) -> xspec.Model:
        """
        Set up a CompPS model with specified parameters.

        Parameters
        ----------
        params : dict
            Dictionary containing CompPS parameters using PyXspec names:
            kTe, EleIndex, Gmin, Gmax, kTbb, tau_y, geom, HovR_cyl,
            cosIncl, cov_frac, rel_refl, Fe_ab_re, Me_ab, xi, Tdisk,
            Betor10, Rin, Rout, Redshift
        norm : float
            Model normalization

        Returns
        -------
        xspec.Model
            Configured CompPS model

        Raises
        ------
        ValueError
            If any parameter is outside its valid range
        """
        # Validate parameters first
        is_valid, errors = validate_compps_params(params)
        if not is_valid:
            error_msg = "Invalid CompPS parameters:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

        # Create CompPS model
        model = xspec.Model("compps")

        # Set parameters using actual PyXspec parameter names
        for param_name, param_value in params.items():
            if hasattr(model.compPS, param_name):
                if param_name == 'tau_y':
                    # Set extended limits for tau_y to allow negative values (y-parameter mode)
                    # Format: [value, delta, min, bot, top, max]
                    getattr(model.compPS, param_name).values = [
                        param_value, 0.01, -4.0, -3.0, 3.0, 4.0
                    ]
                else:
                    getattr(model.compPS, param_name).values = param_value

        # Set normalization
        model.compPS.norm.values = norm

        return model

    def setup_absorbed_powerlaw(self,
                                nh: float = 0.01,
                                photon_index: float = 2.0,
                                norm: float = 1e-3) -> xspec.Model:
        """
        Set up an absorbed powerlaw model for fitting.

        Parameters
        ----------
        nh : float
            Hydrogen column density (10^22 cm^-2)
        photon_index : float
            Powerlaw photon index
        norm : float
            Normalization

        Returns
        -------
        xspec.Model
            Configured tbabs*powerlaw model
        """
        model = xspec.Model("tbabs*powerlaw")

        # Set initial parameter values
        model.TBabs.nH.values = nh
        model.powerlaw.PhoIndex.values = photon_index
        model.powerlaw.norm.values = norm

        # Allow parameters to vary in fitting
        model.TBabs.nH.frozen = False
        model.powerlaw.PhoIndex.frozen = False
        model.powerlaw.norm.frozen = False

        return model

    def generate_fake_spectrum(self,
                              arf_file: str,
                              rmf_file: str,
                              output_file: str,
                              model: xspec.Model,
                              exposure: float = 10000.0,
                              background: Optional[str] = None) -> None:
        """
        Generate a fake spectrum using fakeit.

        Parameters
        ----------
        arf_file : str
            Path to ARF file
        rmf_file : str
            Path to RMF file
        output_file : str
            Output spectrum filename
        model : xspec.Model
            Source model for simulation
        exposure : float
            Exposure time in seconds
        background : str, optional
            Background file (if None, no background)
        """
        # Clear any existing data
        xspec.AllData.clear()

        # Set up fakeit parameters
        fakeit_kwargs = {
            'response': rmf_file,
            'arf': arf_file,
            'exposure': exposure,
            'fileName': output_file,
        }
        
        # Only add background if specified
        if background is not None:
            fakeit_kwargs['background'] = background
        
        fakeit_settings = xspec.FakeitSettings(**fakeit_kwargs)

        # Generate fake spectrum
        xspec.AllData.fakeit(
            nSpectra=1,
            settings=fakeit_settings,
            applyStats=True
        )

    def fit_spectrum(self,
                    spectrum_file: str,
                    model: xspec.Model,
                    max_iterations: int = 100) -> Dict[str, Any]:
        """
        Fit a spectrum with specified model.

        Parameters
        ----------
        spectrum_file : str
            Path to spectrum file
        model : xspec.Model
            Model to fit
        max_iterations : int
            Maximum number of fit iterations

        Returns
        -------
        dict
            Fit results including parameters, errors, and statistics
        """
        # Load spectrum
        xspec.AllData.clear()
        spectrum = xspec.Spectrum(spectrum_file)

        # Perform fit
        xspec.Fit.nIterations = max_iterations
        xspec.Fit.perform()

        # Calculate errors for PhoIndex using 90% confidence level
        try:
            xspec.Fit.error("1-10")
            xspec.Fit.error("2.706 2")  # 2.706 = 90% confidence, 2 = PhoIndex parameter
        except Exception as e:
            print(f"Warning: PhoIndex error calculation failed: {e}")

        # Extract fit results
        results = {
            'statistic': xspec.Fit.statistic,
            'dof': xspec.Fit.dof,
            'chi_squared': xspec.Fit.statistic,
            'reduced_chi_squared': xspec.Fit.statistic / xspec.Fit.dof if xspec.Fit.dof > 0 else None,
            'parameters': {},
            'errors': {}
        }

        # Extract parameter values and errors
        for comp in model.componentNames:
            component = getattr(model, comp)
            for param_name in component.parameterNames:
                param = getattr(component, param_name)
                full_name = f"{comp}.{param_name}"
                results['parameters'][full_name] = param.values[0]

                # For PhoIndex, extract errors using specific method
                if comp == 'powerlaw' and param_name == 'PhoIndex':
                    if hasattr(param, 'error') and param.error is not None:
                        # Error bounds: [lower, upper]
                        results['errors'][full_name] = [param.error[0], param.error[1]]
                # For other parameters, use generic error extraction
                elif hasattr(param, 'error'):
                    results['errors'][full_name] = param.error

        return results

    def get_model_flux(self,
                      energy_min: float = 0.3,
                      energy_max: float = 10.0,
                      model_num: int = 1) -> Tuple[float, float]:
        """
        Get model flux in specified energy range.

        Parameters
        ----------
        energy_min : float
            Minimum energy (keV)
        energy_max : float
            Maximum energy (keV)
        model_num : int
            Model number

        Returns
        -------
        tuple
            (flux, flux_error) in units of photons/cm^2/s
        """
        xspec.AllModels.calcFlux(f"{energy_min} {energy_max}")
        flux = xspec.AllData(1).flux[0]
        flux_err = xspec.AllData(1).flux[1]
        return flux, flux_err

    @staticmethod
    def clear_session():
        """Clear all data and models from XSPEC."""
        xspec.AllData.clear()
        xspec.AllModels.clear()


def validate_response_files(arf_file: str, rmf_file: str) -> bool:
    """
    Validate that ARF and RMF files exist.

    Parameters
    ----------
    arf_file : str
        Path to ARF file
    rmf_file : str
        Path to RMF file

    Returns
    -------
    bool
        True if both files exist
    """
    arf_exists = Path(arf_file).exists()
    rmf_exists = Path(rmf_file).exists()

    if not arf_exists:
        print(f"Error: ARF file not found: {arf_file}")
    if not rmf_exists:
        print(f"Error: RMF file not found: {rmf_file}")

    return arf_exists and rmf_exists


def get_energy_channels(spectrum_file: str) -> np.ndarray:
    """
    Extract energy channels from a spectrum file.

    Parameters
    ----------
    spectrum_file : str
        Path to spectrum file

    Returns
    -------
    np.ndarray
        Array of energy values
    """
    from astropy.io import fits

    with fits.open(spectrum_file) as hdul:
        # Try to get energy from SPECTRUM extension
        if 'SPECTRUM' in hdul:
            data = hdul['SPECTRUM'].data
            if 'CHANNEL' in data.columns.names:
                return data['CHANNEL']

    return np.array([])

