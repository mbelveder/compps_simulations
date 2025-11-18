# CompPS Parameter Sets Comparison


**Default Settings:**
- Normalization: 1.0
- Exposure: 100000 seconds (100 ks)
- Energy Range: 0.3-10.0 keV

---


## CompPS Parameter Definitions

Complete reference for all 19 CompPS model parameters from XSPEC documentation.

| Par | Name | Description |
|-----|------|-------------|
| 1 | `kTe` | Electron temperature (keV) |
| 2 | `EleIndex` | Electron power-law index p [N(γ)=γ^(-p)] |
| 3 | `Gmin` | Minimum Lorentz factor γ |
| 4 | `Gmax` | Maximum Lorentz factor γ |
| | | **Electron distribution rules:** If Gmin or Gmax <1: Maxwellian with kTe. If kTe=0: power-law with EleIndex, Gmin, Gmax. If Gmax<Gmin (both ≥1): cutoff Maxwellian. If kTe≠0 and Gmin,Gmax≥1: hybrid distribution |
| 5 | `kTbb` | Seed photon temperature (keV). If >0: blackbody. If <0: multicolor disk with T_inner=\|kTbb\| |
| 6 | `tau_y` | If >0: vertical optical depth τ. If <0: Compton y-parameter (y=4Θτ). Limits: slab τ<1, sphere τ<3 |
| 7 | `geom` | Geometry: 0=escape probability sphere (fast), 1=slab, 2=cylinder, 3=hemisphere, 4,5=sphere. Negative values: isotropic/homogeneous sources. -5=sphere with eigenfunction distribution |
| 8 | `HovR_cyl` | Height-to-radius ratio (cylinder geometry only) |
| 9 | `cosIncl` | Cosine of inclination angle (if <0: only blackbody emission) |
| 10 | `cov_frac` | Covering factor of cold clouds (dummy parameter for geom=±4,5) |
| 11 | `rel_refl` | Reflection amount R=Ω/(2π). If R<0: only reflection component |
| 12 | `Fe_ab_re` | Iron abundance (solar units) |
| 13 | `Me_ab` | Heavy element abundance (solar units) |
| 14 | `xi` | Disk ionization parameter ξ=L/(nR²) |
| 15 | `Tdisk` | Disk temperature for reflection (K) |
| 16 | `Betor10` | Reflection emissivity law (r^β). -10=non-rotating disk, +10=(1-√(6/r_g))/r_g³ |
| 17 | `Rin` | Inner disk radius (Schwarzschild radii R_g) |
| 18 | `Rout` | Outer disk radius (Schwarzschild radii R_g) |
| 19 | `Redshift` | Source redshift z |
---

## Physics-Motivated AGN Scenarios

| Parameter | typical_agn_slab | typical_agn_sphere | high_accretion | low_accretion | nls1_type | hot_corona | high_inclination | boundary_case |
|-----------|-------|-------|-------|-------|-------|-------|-------|-------|
| **kTe** | 105.00 | 105.00 | 100.00 | 180.00 | 80.00 | 200.00 | 105.00 | 150.00 |
| **tau_y** | 0.25 | 0.80 | 0.40 | 0.20 | 0.60 | 0.15 | 0.25 | 0.30 |
| **kTbb** | 0.30 | 0.30 | 0.50 | 0.20 | 0.80 | 0.15 | 0.30 | 0.25 |
| **geom** | 1.00 | 4.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **cosIncl** | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.10 | 0.71 |
| **rel_refl** | 1.00 | 1.00 | 1.20 | 0.30 | 1.00 | 0.30 | 1.00 | 0.80 |
| **xi** | 100.00 | 100.00 | 200.00 | 50.00 | 300.00 | 20.00 | 100.00 | 100.00 |
| **Tdisk** | 3.00e+04 | 3.00e+04 | 5.00e+04 | 2.00e+04 | 6.00e+04 | 1.50e+04 | 3.00e+04 | 3.00e+04 |
| *Other params* |... |... |... |... |... |... |... |...|
| EleIndex | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| Gmin | -1.00 | -1.00 | -1.00 | -1.00 | -1.00 | -1.00 | -1.00 | -1.00 |
| Gmax | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| HovR_cyl | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| cov_frac | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Fe_ab_re | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Me_ab | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Betor10 | -10.00 | -10.00 | -10.00 | -10.00 | -10.00 | -10.00 | -10.00 | -10.00 |
| Rin | 6.01 | 6.01 | 6.01 | 6.01 | 6.01 | 6.01 | 6.01 | 6.01 |
| Rout | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Redshift | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Temperature-Optical Depth Grid

Systematic exploration of kTe-τ relationship.

| Parameter | kTe80_tau0.33 | kTe105_tau0.25 | kTe130_tau0.20 | kTe180_tau0.15 | kTe250_tau0.11 |
|-----------|-------|-------|-------|-------|-------|
| **kTe** | 80.00 | 105.00 | 130.00 | 180.00 | 250.00 |
| **tau_y** | 0.33 | 0.25 | 0.20 | 0.15 | 0.11 |
| **kTbb** | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 |
| **geom** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **cosIncl** | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| **rel_refl** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **xi** | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| **Tdisk** | 3.00e+04 | 3.00e+04 | 3.00e+04 | 3.00e+04 | 3.00e+04 |
| *Other params* |... |... |... |... |...|
| EleIndex | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| Gmin | -1.00 | -1.00 | -1.00 | -1.00 | -1.00 |
| Gmax | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| HovR_cyl | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| cov_frac | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Fe_ab_re | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Me_ab | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Betor10 | -10.00 | -10.00 | -10.00 | -10.00 | -10.00 |
| Rin | 6.01 | 6.01 | 6.01 | 6.01 | 6.01 |
| Rout | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Redshift | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Reflection Strength Variation Grid


| Parameter | weak_refl | medium_refl | strong_refl |
|-----------|-------|-------|-------|
| **kTe** | 105.00 | 105.00 | 105.00 |
| **tau_y** | 0.25 | 0.25 | 0.25 |
| **kTbb** | 0.30 | 0.30 | 0.30 |
| **geom** | 1.00 | 1.00 | 1.00 |
| **cosIncl** | 0.50 | 0.50 | 0.50 |
| **rel_refl** | 0.30 | 0.70 | 1.20 |
| **xi** | 100.00 | 100.00 | 100.00 |
| **Tdisk** | 3.00e+04 | 3.00e+04 | 3.00e+04 |
| *Other params* |... |... |...|
| EleIndex | 2.00 | 2.00 | 2.00 |
| Gmin | -1.00 | -1.00 | -1.00 |
| Gmax | 10.00 | 10.00 | 10.00 |
| HovR_cyl | 1.00 | 1.00 | 1.00 |
| cov_frac | 1.00 | 1.00 | 1.00 |
| Fe_ab_re | 1.00 | 1.00 | 1.00 |
| Me_ab | 1.00 | 1.00 | 1.00 |
| Betor10 | -10.00 | -10.00 | -10.00 |
| Rin | 6.01 | 6.01 | 6.01 |
| Rout | 100.00 | 100.00 | 100.00 |
| Redshift | 0.0 | 0.0 | 0.0 |

---

*This document was automatically generated from `config/parameters.py`.*
*Total scenarios: 16*
