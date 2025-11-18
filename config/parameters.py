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

# Physics-motivated parameter sets based on AGN observations
# Reference: theory.md - observational constraints from literature
COMPPS_PARAMS = {
    # 1. Typical AGN corona (Ricci et al. 2018 median values, slab geometry)
    'typical_agn_slab': {
        'kTe': 105.0,        # Median observed value (Ricci+ 2018)
        'EleIndex': 2.0,     # Not used for Maxwellian
        'Gmin': -1.0,        # Maxwellian distribution
        'Gmax': 10.0,
        'kTbb': 0.3,         # Typical disk seed photons
        'tau_y': 0.25,       # Median optical depth (slab geometry)
        'geom': 1.0,         # Slab geometry
        'HovR_cyl': 1.0,
        'cosIncl': 0.5,      # 60 deg inclination
        'cov_frac': 1.0,
        'rel_refl': 1.0,     # Standard reflection
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 100.0,         # Moderate ionization
        'Tdisk': 30000.0,
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # 2. Spherical corona geometry (same temp, higher tau)
    'typical_agn_sphere': {
        'kTe': 105.0,        # Same temperature
        'EleIndex': 2.0,
        'Gmin': -1.0,
        'Gmax': 10.0,
        'kTbb': 0.3,
        'tau_y': 0.8,        # Spherical geometry optical depth
        'geom': 4.0,         # Sphere geometry
        'HovR_cyl': 1.0,
        'cosIncl': 0.5,
        'cov_frac': 1.0,
        'rel_refl': 1.0,
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 100.0,
        'Tdisk': 30000.0,
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # 3. High accretion rate AGN (lambda_Edd > 0.1, Ricci+ 2018)
    'high_accretion': {
        'kTe': 100.0,        # Cooler corona (higher accretion)
        'EleIndex': 2.0,
        'Gmin': -1.0,
        'Gmax': 10.0,
        'kTbb': 0.5,         # Higher seed photon temp
        'tau_y': 0.4,        # Higher optical depth
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.5,
        'cov_frac': 1.0,
        'rel_refl': 1.2,     # Higher reflection (low luminosity)
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 200.0,         # More ionized
        'Tdisk': 50000.0,    # Hotter disk
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # 4. Low accretion rate AGN (lambda_Edd <= 0.1, Ricci+ 2018)
    'low_accretion': {
        'kTe': 180.0,        # Hotter corona (lower accretion)
        'EleIndex': 2.0,
        'Gmin': -1.0,
        'Gmax': 10.0,
        'kTbb': 0.2,         # Cooler seed photons
        'tau_y': 0.2,        # Lower optical depth (anticorrelation)
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.5,
        'cov_frac': 1.0,
        'rel_refl': 0.3,     # Lower reflection (high luminosity)
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 50.0,          # Less ionized
        'Tdisk': 20000.0,    # Cooler disk
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # 5. Narrow-Line Seyfert 1 type (steep spectrum, Gamma > 2)
    'nls1_type': {
        'kTe': 80.0,         # Cooler corona
        'EleIndex': 2.0,
        'Gmin': -1.0,
        'Gmax': 10.0,
        'kTbb': 0.8,         # Hot seed photons
        'tau_y': 0.6,        # Higher optical depth
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.5,
        'cov_frac': 1.0,
        'rel_refl': 1.0,
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 300.0,         # Highly ionized
        'Tdisk': 60000.0,    # Hot disk
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # 6. Hard state (very hot corona, upper observed range)
    'hot_corona': {
        'kTe': 200.0,        # Upper end of observed range
        'EleIndex': 2.0,
        'Gmin': -1.0,
        'Gmax': 10.0,
        'kTbb': 0.15,        # Low seed photon temp
        'tau_y': 0.15,       # Very low optical depth (anticorrelation)
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.5,
        'cov_frac': 1.0,
        'rel_refl': 0.3,
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 20.0,
        'Tdisk': 15000.0,
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # 7. High inclination study (edge-on view)
    'high_inclination': {
        'kTe': 105.0,        # Typical temperature
        'EleIndex': 2.0,
        'Gmin': -1.0,
        'Gmax': 10.0,
        'kTbb': 0.3,
        'tau_y': 0.25,
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.1,      # ~84 deg (nearly edge-on)
        'cov_frac': 1.0,
        'rel_refl': 1.0,
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 100.0,
        'Tdisk': 30000.0,
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # 8. Parameter space boundary (near pair production limit)
    'boundary_case': {
        'kTe': 150.0,        # Near runaway pair production boundary
        'EleIndex': 2.0,
        'Gmin': -1.0,
        'Gmax': 10.0,
        'kTbb': 0.25,
        'tau_y': 0.3,
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.707,    # 45 deg
        'cov_frac': 1.0,
        'rel_refl': 0.8,
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 100.0,
        'Tdisk': 30000.0,
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # === Temperature-Optical Depth Anticorrelation Grid ===
    # Systematic exploration of kTe-tau relationship
    # Based on observed anticorrelation: kTe ~ 1/tau

    # 9. Cool corona, high optical depth
    'kTe80_tau0.33': {
        'kTe': 80.0,
        'EleIndex': 2.0,
        'Gmin': -1.0,
        'Gmax': 10.0,
        'kTbb': 0.3,
        'tau_y': 0.33,       # tau scaled as: 105*0.25/80
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.5,
        'cov_frac': 1.0,
        'rel_refl': 1.0,
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 100.0,
        'Tdisk': 30000.0,
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # 10. Median temperature and optical depth
    'kTe105_tau0.25': {
        'kTe': 105.0,
        'EleIndex': 2.0,
        'Gmin': -1.0,
        'Gmax': 10.0,
        'kTbb': 0.3,
        'tau_y': 0.25,       # Observed median (Ricci+ 2018)
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.5,
        'cov_frac': 1.0,
        'rel_refl': 1.0,
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 100.0,
        'Tdisk': 30000.0,
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # 11. Intermediate temperature
    'kTe130_tau0.20': {
        'kTe': 130.0,
        'EleIndex': 2.0,
        'Gmin': -1.0,
        'Gmax': 10.0,
        'kTbb': 0.3,
        'tau_y': 0.20,       # tau scaled as: 105*0.25/130
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.5,
        'cov_frac': 1.0,
        'rel_refl': 1.0,
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 100.0,
        'Tdisk': 30000.0,
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # 12. Hot corona, low optical depth
    'kTe180_tau0.15': {
        'kTe': 180.0,
        'EleIndex': 2.0,
        'Gmin': -1.0,
        'Gmax': 10.0,
        'kTbb': 0.3,
        'tau_y': 0.15,       # tau scaled as: 105*0.25/180
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.5,
        'cov_frac': 1.0,
        'rel_refl': 1.0,
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 100.0,
        'Tdisk': 30000.0,
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # 13. Very hot corona, very low optical depth
    'kTe250_tau0.11': {
        'kTe': 250.0,
        'EleIndex': 2.0,
        'Gmin': -1.0,
        'Gmax': 10.0,
        'kTbb': 0.3,
        'tau_y': 0.11,       # tau scaled as: 105*0.25/250
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.5,
        'cov_frac': 1.0,
        'rel_refl': 1.0,
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 100.0,
        'Tdisk': 30000.0,
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # === Reflection Strength Variation Grid ===
    # Based on observed anticorrelation between reflection and luminosity
    # R ~ 1.2 for low L_X, R ~ 0.3 for high L_X (Balokovic+ 2017)

    # 14. Weak reflection (high luminosity AGN)
    'weak_refl': {
        'kTe': 105.0,        # Typical AGN parameters
        'EleIndex': 2.0,
        'Gmin': -1.0,
        'Gmax': 10.0,
        'kTbb': 0.3,
        'tau_y': 0.25,
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.5,
        'cov_frac': 1.0,
        'rel_refl': 0.3,     # Weak reflection (high L_X)
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 100.0,
        'Tdisk': 30000.0,
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # 15. Medium reflection (intermediate luminosity)
    'medium_refl': {
        'kTe': 105.0,
        'EleIndex': 2.0,
        'Gmin': -1.0,
        'Gmax': 10.0,
        'kTbb': 0.3,
        'tau_y': 0.25,
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.5,
        'cov_frac': 1.0,
        'rel_refl': 0.7,     # Medium reflection
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 100.0,
        'Tdisk': 30000.0,
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },

    # 16. Strong reflection (low luminosity AGN)
    'strong_refl': {
        'kTe': 105.0,
        'EleIndex': 2.0,
        'Gmin': -1.0,
        'Gmax': 10.0,
        'kTbb': 0.3,
        'tau_y': 0.25,
        'geom': 1.0,
        'HovR_cyl': 1.0,
        'cosIncl': 0.5,
        'cov_frac': 1.0,
        'rel_refl': 1.2,     # Strong reflection (low L_X)
        'Fe_ab_re': 1.0,
        'Me_ab': 1.0,
        'xi': 100.0,
        'Tdisk': 30000.0,
        'Betor10': -10.0,
        'Rin': 6.01,
        'Rout': 100.0,
        'Redshift': 0.0,
    },
}

# Normalization value (to be adjusted based on source flux requirements)
DEFAULT_NORM = 1.0

# Exposure time for simulations (seconds)
DEFAULT_EXPOSURE = 100000.0

# Energy range for fitting (keV)
ENERGY_RANGE = {
    'min': 0.3,
    'max': 10.0,
}
