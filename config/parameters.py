"""
CompPS parameter configurations for simulations.

This module defines parameter sets for CompPS model simulations.
Each parameter set should be a dictionary with 19 parameters as defined
in the CompPS model specification.

CompPS Model Parameters (PyXspec names) with hard limits:
    par1:  kTe - electron temperature (keV) [20, 100000]
    par2:  EleIndex - electron power-law index [0, 5]
    par3:  Gmin - minimum Lorentz factor [-1, 10]
    par4:  Gmax - maximum Lorentz factor [10, 10000]
    par5:  kTbb - temperature of soft photons (keV) [0.001, 10]
    par6:  tau_y - optical depth [0.05, 3]
    par7:  geom - geometry type [-5, 4]
    par8:  HovR_cyl - height-to-radius ratio [0.5, 2]
    par9:  cosIncl - cosine of inclination angle [0.05, 0.95]
    par10: cov_frac - covering factor [0, 1]
    par11: rel_refl - reflection amount [0, 10000]
    par12: Fe_ab_re - iron abundance (solar units) [0.1, 10]
    par13: Me_ab - heavy element abundance (solar units) [0.1, 10]
    par14: xi - disk ionization parameter [0, 100000]
    par15: Tdisk - disk temperature (K) [10000, 1e6]
    par16: Betor10 - reflection emissivity law [-10, 10]
    par17: Rin - inner disk radius (Rg) [6.001, 10000]
    par18: Rout - outer disk radius (Rg) [0, 1e6]
    par19: Redshift - redshift [-0.999, 10]
"""

# Example parameter sets for CompPS simulations
COMPPS_PARAMS = {
    # Basic Maxwellian electron distribution, slab geometry
    'basic_thermal': {
        'kTe': 50.0,         # 50 keV electron temperature
        'EleIndex': 2.0,     # power-law index (not used for Maxwellian)
        'Gmin': -1.0,        # -1 indicates Maxwellian
        'Gmax': 10.0,        # minimum allowed value (not used for Gmin<1)
        'kTbb': 0.5,         # 0.5 keV blackbody seed photons
        'tau_y': 1.0,        # optical depth
        'geom': 1.0,         # slab geometry
        'HovR_cyl': 1.0,     # not used for slab
        'cosIncl': 0.5,      # 60 degree inclination
        'cov_frac': 1.0,     # full covering
        'rel_refl': 0.0,     # no reflection
        'Fe_ab_re': 1.0,     # solar iron abundance
        'Me_ab': 1.0,        # solar metal abundance
        'xi': 0.0,           # neutral reflector
        'Tdisk': 30000.0,    # disk temperature in K
        'Betor10': -10.0,    # non-rotating disk
        'Rin': 6.01,         # inner radius (must be > 6.001)
        'Rout': 100.0,       # outer radius
        'Redshift': 0.0,     # no redshift
    },

    # Higher temperature with reflection
    'hot_with_reflection': {
        'kTe': 100.0,
        'EleIndex': 2.0,
        'Gmin': -1.0,        # -1 indicates Maxwellian
        'Gmax': 10.0,
        'kTbb': 0.3,
        'tau_y': 0.5,
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.707,    # 45 degree inclination
        'cov_frac': 1.0,
        'rel_refl': 1.0,     # full reflection
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 100.0,         # ionized reflector
        'Tdisk': 50000.0,
        'Betor10': -10.0,
        'Rin': 6.01,         # inner radius (must be > 6.001)
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # Spherical geometry
    'spherical': {
        'kTe': 75.0,
        'EleIndex': 2.0,
        'Gmin': -1.0,        # -1 indicates Maxwellian
        'Gmax': 10.0,
        'kTbb': 0.4,
        'tau_y': 2.0,
        'geom': 4.0,         # sphere geometry
        'HovR_cyl': 1.0,
        'cosIncl': 0.866,    # 30 degree inclination
        'cov_frac': 1.0,
        'rel_refl': 0.0,
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 0.0,
        'Tdisk': 30000.0,
        'Betor10': -10.0,
        'Rin': 6.01,         # inner radius (must be > 6.001)
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # Multicolor disk seed photons
    'disk_seed': {
        'kTe': 60.0,
        'EleIndex': 2.0,
        'Gmin': -1.0,        # -1 indicates Maxwellian
        'Gmax': 10.0,
        'kTbb': 0.1,         # low temp seed photons
        'tau_y': 0.8,
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.5,
        'cov_frac': 1.0,
        'rel_refl': 0.5,     # partial reflection
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 50.0,
        'Tdisk': 40000.0,
        'Betor10': -10.0,
        'Rin': 6.01,         # inner radius (must be > 6.001)
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # Power-law electron distribution
    'powerlaw_electrons': {
        'kTe': 20.0,         # minimum allowed (ignored for power-law dist)
        'EleIndex': 2.5,     # power-law index
        'Gmin': 1.5,         # minimum Lorentz factor (>1 = power-law)
        'Gmax': 100.0,       # maximum Lorentz factor
        'kTbb': 0.3,
        'tau_y': 0.5,
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.707,
        'cov_frac': 1.0,
        'rel_refl': 0.0,
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 0.0,
        'Tdisk': 30000.0,
        'Betor10': -10.0,
        'Rin': 6.01,         # inner radius (must be > 6.001)
        'Rout': 100.0,
        'Redshift': 0.0,
    },
}

# Normalization value (to be adjusted based on source flux requirements)
DEFAULT_NORM = 1.0

# Exposure time for simulations (seconds)
DEFAULT_EXPOSURE = 10000.0

# Energy range for simulations (keV)
ENERGY_RANGE = {
    'min': 0.3,
    'max': 10.0,
}

