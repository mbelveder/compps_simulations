# XSPEC CompPS Model Parameter Ranges for AGN

## Summary Table: Typical Parameter Ranges

| Parameter | Typical Range | Median/Mean Value | Source | Notes |
|-----------|---------------|------------------|--------|-------|
| **Plasma Temperature (kT_e)** | 50–300 keV | 105 ± 18 keV (slab) | Ricci et al. 2018 [1] | Slab geometry; varies with Eddington ratio |
| | | 100–200 keV | Multiple sources [1][2] | Hot, optically thin corona |
| | 0.1–2 keV | — | Multiple sources [2][3] | Warm corona (soft X-ray excess) |
| **Optical Depth (τ_e)** | 0.2–1.0 | 0.25 ± 0.06 (slab) | Ricci et al. 2018 [1] | Hot, thin corona; slab geometry |
| | ~0.8 | — | Ricci et al. 2018 [1] | Spherical corona assumption |
| | 10–50 | — | Multiple sources [3][4] | Warm Comptonization (soft excess) |
| **Photon Index (Γ)** | 1.5–2.0 | 1.78 ± 0.10 | Ricci et al. 2017 [1][2] | Median for large AGN sample |
| | 1.65–1.75 | 1.65 ± 0.03 | NuSTAR extragalactic sample [5] | Broad-line AGN average |
| | >2.0 | — | Multiple sources [2] | Narrow-line Seyfert 1 galaxies |
| **High-Energy Cutoff (E_c)** | 50–300 keV | 150–370 keV | Multiple sources [1][2][6] | Depends on Eddington ratio |
| | 160 ± 41 keV | — | λ_Edd > 0.1 | Ricci et al. 2018 [1] |
| | 370 ± 51 keV | — | λ_Edd ≤ 0.1 | Ricci et al. 2018 [1] |
| | 100–200 keV | — | Typical range [2] | Most AGN coronal emission [2] |
| **Compton Parameter (y)** | — | 0.25–0.42 | Various AGN [1][7] | Related to optical depth |
| **Reflection Parameter (R)** | 0.3–1.2 | ~1.0 | Multiple sources [1][8] | Decreases with luminosity |

---

## Detailed Findings by Study

### 1. BAT AGN Spectroscopic Survey – XII (Ricci et al. 2018) [1]

**Sample:** 211 Swift/BAT-selected AGN with hard X-ray data

**Key Results (slab geometry corona):**
- Plasma temperature (slab): **kT_e = 105 ± 18 keV**
- Optical depth (slab): **τ = 0.25 ± 0.06**
- High-energy cutoff dependence on Eddington ratio:
  - λ_Edd > 0.1: **E_C = 160 ± 41 keV** (cooler, higher optical depth)
  - λ_Edd ≤ 0.1: **E_C = 370 ± 51 keV** (hotter, lower optical depth)
- Spherical geometry assumption: **τ = 0.8** for model comparison

**Physical Interpretation:**
- Significant anticorrelation between temperature and optical depth
- Sources cluster near the runaway pair production boundary in temperature-compactness space
- Higher Eddington ratio sources have cooler coronae (radiatively compact)

---

### 2. X-ray Properties of Coronal Emission (Laha et al. 2025) [2]

**Sample:** Large AGN compilation including NuSTAR, XMM-Newton, and Chandra observations

**Reported Ranges:**
- Photon index: **Γ = 1.78 ± 0.10** (median for large AGN samples)
- Optical depth: **τ ≈ 0.25 ± 0.06** (median)
- Plasma temperature: **100–200 keV** (typical for hot corona)
- Warm corona temperature: **0.1–2 keV** (soft X-ray excess component)
- High-energy cutoff: **100–200 keV typical**

**Correlations Identified:**
- Anticorrelation between **kT_e and τ** (confirmed by multiple NuSTAR studies)
- Anticorrelation between **E_C and Γ** (statistically significant, not just parameter degeneracy)
- Narrow-line Seyfert 1 galaxies: **Γ > 2** (cooler or optically thicker corona)

---

### 3. NuSTAR Extragalactic Survey (Balokovic et al. 2017) [5]

**Sample:** NuSTAR-detected AGN combined with Chandra/XMM-Newton

**Average Spectral Properties:**
- Average photon index: **Γ = 1.65 ± 0.03** (whole sample)
- Reflection strength: **R ≈ 1.2** for L_10-40keV < 10^44 erg/s
- Reflection strength: **R ≈ 0.3** for L_10-40keV > 10^44 erg/s
- Significant anticorrelation: **R decreases with luminosity**

---

### 4. Coronal Properties of Low-Accreting AGN (Jana et al. 2023) [8]

**Sample:** 30 Swift/BAT-selected low-accreting AGN (λ_Edd < 10^-3) with XMM-Newton/NuSTAR observations

**Measured Coronal Parameters:**
- Hot electron plasma temperature
- Cutoff energy
- Optical depth
- Photon index

**Key Findings:**
- Relationship between **E_c and kT_e** similar in low-accretion domain to higher-accretion AGN
- Correlations and anticorrelations tested among spectral parameters
- Low-accretion AGN show distinct coronal properties

---

### 5. Broad-band X-ray Spectral Analysis with COMPPS (Lubinski et al. 2016) [7]

**Sample:** Bright AGN observed with hard X-ray satellites

**COMPPS Model Results:**
- Plasma temperature range: **kT_e < 100 keV** (20 of 22 objects)
- Outliers: **kT_e > 200 keV** (2 objects)
- Compton parameter: **y ≈ 0.25–0.42**
- Reflection parameter: **R typically ~1.0**
- Photon index: **Γ ≈ 1.8–1.9** (typical range)
- Expected cutoff energy relation: **E_C = 2–3 kT_e**

---

## Geometric Considerations

### Corona Geometry Effects on Parameter Derivation

| Geometry | Optical Depth | Temperature Range | Notes |
|----------|---------------|-------------------|-------|
| **Spherical** | ~0.8 (typical) | 100–200 keV | Most commonly used for AGN |
| **Slab (disk)** | 0.25 ± 0.06 | Similar to spherical | RICCI et al. 2018 measurement [1] |
| **Warm corona** | 10–50+ | 0.1–2 keV | Responsible for soft X-ray excess |

---

## Accretion Rate Dependence

### Eddington Ratio Effects (λ_Edd = L/L_Edd)

| Eddington Ratio | Typical kT_e | Typical E_c | Typical τ | Reference |
|-----------------|--------------|------------|----------|-----------|
| λ_Edd ≤ 0.1 | ~150–200 keV | 370 ± 51 keV | ~0.25 | Ricci et al. 2018 [1] |
| λ_Edd > 0.1 | ~100 keV | 160 ± 41 keV | Higher | Ricci et al. 2018 [1] |
| λ_Edd < 10^-3 | Measured | Varies | Measured | Jana et al. 2023 [8] |

**Physical Interpretation:**
- Lower accretion rates: **Hotter coronae, lower optical depth**
- Higher accretion rates: **Cooler coronae, higher optical depth**
- Reflects changes in corona geometry and compactness as accretion rate changes

---

## AGN Type Dependencies

### Type 1 vs. Type 2 AGN

- **Type 1 (Unobscured):** Standard CompPS parameters apply
- **Type 2 (Obscured):** CompPS typically combined with torus absorption models
- **Compton-thick AGN:** E_C and temperature measurements more uncertain due to Compton scattering in torus

---

## Key Physical Relationships

### Temperature-Optical Depth Anticorrelation
```
kT_e (keV) ∝ 1/τ

Mathematical relation:
Γ ≈ √[9/4 + (m_e c²)/(kT_e τ(1+τ/3))] - 1/2
```
Sources: [1][2][4]

### Cutoff Energy Relation
```
E_c ≈ 2–3 × kT_e
```
Sources: [2][3][5]

### Compton Parameter
```
y = max(τ, τ²) × (4kT_e)/(m_e c²)
τ = y × m_e c² / (4kT_e)
```
Sources: [1][2]

---

## References

[1] Ricci, C., et al. 2018, MNRAS, 480, 1802 — BAT AGN Spectroscopic Survey – XII. Temperature and optical depth of comptonizing plasma in AGN.

[2] Laha, S., et al. 2025, Frontiers in Astronomy and Space Sciences, 12, 1422652 — X-ray properties of coronal emission in radio quiet AGN.

[3] Tortosa, A., et al. 2018a — Temperature-optical depth anticorrelation in AGN coronae (NuSTAR).

[4] Multiple sources — Warm/optically thick corona parameters for soft X-ray excess.

[5] Balokovic, M., et al. 2017, ApJ, 847, 147 — NuSTAR Extragalactic Surveys: Broad-band X-ray spectroscopy.

[6] Fabian, A. C., et al. 2015, MNRAS, 451, 4375 — Temperature and compactness constraints.

[7] Lubinski, P., et al. 2016, MNRAS, 458, 2454 — Hard X-ray spectra of bright Seyfert galaxies with COMPPS.

[8] Jana, A., et al. 2023, MNRAS, 518, 5729 — Coronal Properties of Low-Accreting AGN using Swift, XMM-Newton, and NuSTAR.

