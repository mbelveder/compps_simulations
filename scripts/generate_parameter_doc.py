#!/usr/bin/env python3
"""
Generate human-readable Markdown documentation from parameters.py.

This script converts the CompPS parameter configurations into a formatted
Markdown table for easy comparison of different scenarios.
"""

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parameters import COMPPS_PARAMS, DEFAULT_NORM, DEFAULT_EXPOSURE, ENERGY_RANGE


def format_parameter_value(value):
    """Format parameter value for display."""
    if isinstance(value, float):
        if value == 0.0:
            return "0.0"
        elif abs(value) >= 1000 or abs(value) < 0.01:
            return f"{value:.2e}"
        else:
            return f"{value:.2f}"
    return str(value)


def generate_parameter_table(params_dict):
    """Generate a markdown table comparing all parameter sets."""

    # Get all parameter names from first scenario
    first_scenario = next(iter(params_dict.values()))
    param_names = list(first_scenario.keys())

    # Header
    md_lines = []
    md_lines.append("# CompPS Parameter Sets Comparison")
    md_lines.append("")
    md_lines.append("")
    md_lines.append(f"**Default Settings:**")
    md_lines.append(f"- Normalization: {DEFAULT_NORM}")
    md_lines.append(f"- Exposure: {DEFAULT_EXPOSURE:.0f} seconds ({DEFAULT_EXPOSURE/1000:.0f} ks)")
    md_lines.append(f"- Energy Range: {ENERGY_RANGE['min']}-{ENERGY_RANGE['max']} keV")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    # Parameter definitions table
    md_lines.append("")
    md_lines.append("## CompPS Parameter Definitions")
    md_lines.append("")
    md_lines.append("Complete reference for all 19 CompPS model parameters from XSPEC documentation.")
    md_lines.append("")
    md_lines.append("| Par | Name | Description |")
    md_lines.append("|-----|------|-------------|")
    md_lines.append("| 1 | `kTe` | Electron temperature (keV) |")
    md_lines.append("| 2 | `EleIndex` | Electron power-law index p [N(γ)=γ^(-p)] |")
    md_lines.append("| 3 | `Gmin` | Minimum Lorentz factor γ |")
    md_lines.append("| 4 | `Gmax` | Maximum Lorentz factor γ |")
    md_lines.append("| | | **Electron distribution rules:** If Gmin or Gmax <1: Maxwellian with kTe. If kTe=0: power-law with EleIndex, Gmin, Gmax. If Gmax<Gmin (both ≥1): cutoff Maxwellian. If kTe≠0 and Gmin,Gmax≥1: hybrid distribution |")
    md_lines.append("| 5 | `kTbb` | Seed photon temperature (keV). If >0: blackbody. If <0: multicolor disk with T_inner=\\|kTbb\\| |")
    md_lines.append("| 6 | `tau_y` | If >0: vertical optical depth τ. If <0: Compton y-parameter (y=4Θτ). Limits: slab τ<1, sphere τ<3 |")
    md_lines.append("| 7 | `geom` | Geometry: 0=escape probability sphere (fast), 1=slab, 2=cylinder, 3=hemisphere, 4,5=sphere. Negative values: isotropic/homogeneous sources. -5=sphere with eigenfunction distribution |")
    md_lines.append("| 8 | `HovR_cyl` | Height-to-radius ratio (cylinder geometry only) |")
    md_lines.append("| 9 | `cosIncl` | Cosine of inclination angle (if <0: only blackbody emission) |")
    md_lines.append("| 10 | `cov_frac` | Covering factor of cold clouds (dummy parameter for geom=±4,5) |")
    md_lines.append("| 11 | `rel_refl` | Reflection amount R=Ω/(2π). If R<0: only reflection component |")
    md_lines.append("| 12 | `Fe_ab_re` | Iron abundance (solar units) |")
    md_lines.append("| 13 | `Me_ab` | Heavy element abundance (solar units) |")
    md_lines.append("| 14 | `xi` | Disk ionization parameter ξ=L/(nR²) |")
    md_lines.append("| 15 | `Tdisk` | Disk temperature for reflection (K) |")
    md_lines.append("| 16 | `Betor10` | Reflection emissivity law (r^β). -10=non-rotating disk, +10=(1-√(6/r_g))/r_g³ |")
    md_lines.append("| 17 | `Rin` | Inner disk radius (Schwarzschild radii R_g) |")
    md_lines.append("| 18 | `Rout` | Outer disk radius (Schwarzschild radii R_g) |")
    md_lines.append("| 19 | `Redshift` | Source redshift z |")
    md_lines.append("---")
    md_lines.append("")
    
    # Create categories
    physics_scenarios = [k for k in params_dict.keys() if not any(x in k for x in ['kTe', '_refl'])]
    temp_tau_scenarios = [k for k in params_dict.keys() if k.startswith('kTe')]
    refl_scenarios = [k for k in params_dict.keys() if '_refl' in k]
    
    # Physics-motivated scenarios
    if physics_scenarios:
        md_lines.append("## Physics-Motivated AGN Scenarios")
        md_lines.append("")
        md_lines.extend(create_comparison_table(params_dict, physics_scenarios, param_names))
        md_lines.append("")
    
    # Temperature-optical depth grid
    if temp_tau_scenarios:
        md_lines.append("## Temperature-Optical Depth Grid")
        md_lines.append("")
        md_lines.append("Systematic exploration of kTe-τ relationship.")
        md_lines.append("")
        md_lines.extend(create_comparison_table(params_dict, temp_tau_scenarios, param_names))
        md_lines.append("")
    
    # Reflection grid
    if refl_scenarios:
        md_lines.append("## Reflection Strength Variation Grid")
        md_lines.append("")
        # md_lines.append("Based on observed anticorrelation between reflection and luminosity.")
        md_lines.append("")
        md_lines.extend(create_comparison_table(params_dict, refl_scenarios, param_names))
        md_lines.append("")
    
    return "\n".join(md_lines)


def create_comparison_table(params_dict, scenario_list, param_names):
    """Create a comparison table for specific scenarios."""
    md_lines = []
    
    # Table header
    header = "| Parameter | " + " | ".join(scenario_list) + " |"
    separator = "|-----------|" + "|".join(["-------" for _ in scenario_list]) + "|"
    
    md_lines.append(header)
    md_lines.append(separator)
    
    # Key parameters to highlight
    key_params = ['kTe', 'tau_y', 'kTbb', 'geom', 'cosIncl', 'rel_refl', 'xi', 'Tdisk']
    
    # Add rows for key parameters first
    for param_name in key_params:
        if param_name in param_names:
            row = f"| **{param_name}** |"
            for scenario in scenario_list:
                value = params_dict[scenario][param_name]
                formatted = format_parameter_value(value)
                row += f" {formatted} |"
            md_lines.append(row)
    
    # Add separator
    md_lines.append(f"| *Other params* |" + " |".join(["..." for _ in scenario_list]) + "|")
    
    # Add remaining parameters
    other_params = [p for p in param_names if p not in key_params]
    for param_name in other_params:
        row = f"| {param_name} |"
        for scenario in scenario_list:
            value = params_dict[scenario][param_name]
            formatted = format_parameter_value(value)
            row += f" {formatted} |"
        md_lines.append(row)
    
    return md_lines


def get_parameter_descriptions():
    """Return descriptions for each parameter."""
    return {
        'kTe': 'Electron temperature (keV)',
        'EleIndex': 'Electron power-law index',
        'Gmin': 'Minimum Lorentz factor (-1 for Maxwellian)',
        'Gmax': 'Maximum Lorentz factor',
        'kTbb': 'Seed photon temperature (keV)',
        'tau_y': 'Optical depth (y-parameter)',
        'geom': 'Geometry (1=slab, 4=sphere)',
        'HovR_cyl': 'Height-to-radius ratio (cylinder)',
        'cosIncl': 'Cosine of inclination angle',
        'cov_frac': 'Covering factor',
        'rel_refl': 'Relative reflection strength',
        'Fe_ab_re': 'Iron abundance (solar units)',
        'Me_ab': 'Heavy element abundance (solar)',
        'xi': 'Disk ionization parameter',
        'Tdisk': 'Disk temperature (K)',
        'Betor10': 'Reflection emissivity law',
        'Rin': 'Inner disk radius (Rg)',
        'Rout': 'Outer disk radius (Rg)',
        'Redshift': 'Redshift'
    }


def generate_quick_reference():
    """Generate a quick reference section."""
    md_lines = []
    md_lines.append("## Quick Reference")
    md_lines.append("")
    md_lines.append("### Parameter Ranges from Observations")
    md_lines.append("")
    md_lines.append("| Parameter | Observed Range | Typical/Median | Reference |")
    md_lines.append("|-----------|----------------|----------------|-----------|")
    md_lines.append("| kTe | 50-300 keV | 105 ± 18 keV | Ricci+ 2018 |")
    md_lines.append("| τ (slab) | 0.2-1.0 | 0.25 ± 0.06 | Ricci+ 2018 |")
    md_lines.append("| τ (sphere) | ~0.8 | 0.8 | Ricci+ 2018 |")
    md_lines.append("| Γ | 1.5-2.0 | 1.78 ± 0.10 | Multiple |")
    md_lines.append("| E_cutoff | 50-300 keV | 150-370 keV | Ricci+ 2018 |")
    md_lines.append("| Reflection R | 0.3-1.2 | ~1.0 | Balokovic+ 2017 |")
    md_lines.append("")
    md_lines.append("### Key Relationships")
    md_lines.append("")
    md_lines.append("- **Temperature-Optical Depth**: kTe ∝ 1/τ (anticorrelation)")
    md_lines.append("- **Eddington Ratio**: Higher λ_Edd → cooler, denser corona")
    md_lines.append("- **Reflection-Luminosity**: R ~ 1.2 (low L_X), R ~ 0.3 (high L_X)")
    md_lines.append("- **Cutoff Energy**: E_c ≈ 2-3 × kTe")
    md_lines.append("")
    
    return "\n".join(md_lines)


def main():
    """Generate the parameter documentation."""
    
    output_file = Path(__file__).parent.parent / "config" / "parameter_sets.md"
    
    print(f"Generating parameter documentation...")
    print(f"Found {len(COMPPS_PARAMS)} parameter sets")
    
    # Generate content
    content = generate_parameter_table(COMPPS_PARAMS)
    # content += "\n" + generate_quick_reference()
    
    # Add footer
    content += "\n---\n\n"
    content += "*This document was automatically generated from `config/parameters.py`.*\n"
    content += f"*Total scenarios: {len(COMPPS_PARAMS)}*\n"
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"✓ Documentation saved to: {output_file}")
    print(f"  - {len(COMPPS_PARAMS)} parameter sets documented")
    
    # Count categories
    physics_count = len([k for k in COMPPS_PARAMS.keys() if not any(x in k for x in ['kTe', '_refl'])])
    temp_tau_count = len([k for k in COMPPS_PARAMS.keys() if k.startswith('kTe')])
    refl_count = len([k for k in COMPPS_PARAMS.keys() if '_refl' in k])
    
    print(f"  - {physics_count} physics-motivated scenarios")
    print(f"  - {temp_tau_count} temperature-optical depth grid points")
    print(f"  - {refl_count} reflection variation scenarios")


if __name__ == "__main__":
    main()

