#!/usr/bin/env python3
"""
kTbb variation study execution script.

This script runs the full pipeline for the kTbb variation study:
1. Simulate spectra using CompPS model with kTbb variants
2. Fit spectra with absorbed powerlaw
3. Analyze results
4. Generate kTbb-specific plots (spectra grids and photon index distributions)
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parameters import (
    COMPPS_PARAMS, generate_ktbb_variants, DEFAULT_NORM, DEFAULT_EXPOSURE, ENERGY_RANGE
)
from src.simulator import SpectrumSimulator
from src.fitter import SpectrumFitter
from src.analysis import ResultsAnalyzer
from src.plotting import SpectrumPlotter
from src.xcm_plotter import plot_xcm_grid_by_ktbb


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run kTbb variation study pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full kTbb study pipeline
  python scripts/run_ktbb_study.py --arf nustar.arf --rmf nustar.rmf

  # Only simulate (no fitting)
  python scripts/run_ktbb_study.py --arf nustar.arf --rmf nustar.rmf --simulate-only

  # Only fit existing spectra
  python scripts/run_ktbb_study.py --fit-only

  # Only analyze and plot existing results
  python scripts/run_ktbb_study.py --analyze-only
        """
    )

    parser.add_argument('--arf', type=str,
                       help='ARF filename (in data/response/)')
    parser.add_argument('--rmf', type=str,
                       help='RMF filename (in data/response/)')

    parser.add_argument('--exposure', type=float, default=DEFAULT_EXPOSURE,
                       help=f'Exposure time in seconds (default: {DEFAULT_EXPOSURE})')
    parser.add_argument('--norm', type=float, default=DEFAULT_NORM,
                       help=f'Model normalization (default: {DEFAULT_NORM})')

    parser.add_argument('--simulate-only', action='store_true',
                       help='Only simulate spectra, skip fitting')
    parser.add_argument('--fit-only', action='store_true',
                       help='Only fit existing spectra')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze existing results')

    parser.add_argument('--simulated-dir', type=str, default='data/simulated',
                       help='Directory for simulated spectra')
    parser.add_argument('--results-dir', type=str, default='data/results',
                       help='Directory for results')
    parser.add_argument('--response-dir', type=str, default='data/response',
                       help='Directory containing ARF/RMF files')

    parser.add_argument('--energy-min', type=float, default=ENERGY_RANGE['min'],
                       help=f'Minimum energy for fitting (keV, default: {ENERGY_RANGE["min"]})')
    parser.add_argument('--energy-max', type=float, default=ENERGY_RANGE['max'],
                       help=f'Maximum energy for fitting (keV, default: {ENERGY_RANGE["max"]})')

    parser.add_argument('--stat-method', type=str, default='cstat',
                       choices=['chi', 'cstat', 'pgstat'],
                       help='Fit statistic method (default: cstat)')

    parser.add_argument('--ktbb-values', type=str,
                       default='0.1,0.3,0.5,0.8',
                       help='Comma-separated list of kTbb values (default: 0.1,0.3,0.5,0.8)')

    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')

    args = parser.parse_args()

    # Parse kTbb values from comma-separated string
    try:
        ktbb_values = [float(x.strip()) for x in args.ktbb_values.split(',')]
    except ValueError:
        parser.error(f"Invalid kTbb values format: {args.ktbb_values}. "
                    "Expected comma-separated floats (e.g., 0.1,0.3,0.5,0.8)")

    # Validate arguments
    if not (args.simulate_only or args.fit_only or args.analyze_only):
        # Full pipeline - need ARF and RMF
        if not args.arf or not args.rmf:
            parser.error("--arf and --rmf are required for simulation")

    # Generate kTbb variants with specified values
    COMPPS_PARAMS_KTBB_STUDY = generate_ktbb_variants(
        ktbb_values=ktbb_values
    )

    print("=" * 70)
    print("kTbb VARIATION STUDY PIPELINE")
    print("=" * 70)
    print(f"kTbb values: {ktbb_values}")
    print(f"Total scenarios: {len(COMPPS_PARAMS_KTBB_STUDY)}")
    print(f"  (16 base scenarios × {len(ktbb_values)} kTbb values)")

    # ========== SIMULATION ==========
    if not (args.fit_only or args.analyze_only):
        print("\n" + "=" * 70)
        print("STEP 1: SIMULATING SPECTRA")
        print("=" * 70)

        simulator = SpectrumSimulator(
            arf_file=args.arf,
            rmf_file=args.rmf,
            output_dir=args.simulated_dir,
            response_dir=args.response_dir
        )

        spectrum_files = simulator.simulate_multiple(
            scenarios=COMPPS_PARAMS_KTBB_STUDY,
            exposure=args.exposure,
            norm=args.norm
        )

        simulator.save_simulation_log()
        summary = simulator.get_simulation_summary()
        print(f"\nSimulation Summary:")
        print(f"  Total simulations: {summary['num_simulations']}")
        print(f"  Output directory: {summary['output_directory']}")

        if args.simulate_only:
            print("\n" + "=" * 70)
            print("Simulation complete. Exiting (--simulate-only specified).")
            return 0

    # ========== FITTING ==========
    if not args.analyze_only:
        print("\n" + "=" * 70)
        print("STEP 2: FITTING SPECTRA")
        print("=" * 70)

        fitter = SpectrumFitter(
            output_dir=args.results_dir,
            energy_range=(args.energy_min, args.energy_max),
            stat_method=args.stat_method
        )

        # Find spectra to fit
        from src.simulator import find_simulated_spectra
        if args.fit_only:
            # Fit all existing spectra (filter for kTbb variants)
            all_spectra = find_simulated_spectra(args.simulated_dir)
            # Filter to only kTbb variant spectra
            import re
            ktbb_pattern = re.compile(r'_ktbb[\d.]+_')
            spectrum_files = [
                s for s in all_spectra if ktbb_pattern.search(Path(s).name)
            ]
            if not spectrum_files:
                print("Error: No kTbb variant spectra found")
                return 1
        # else: spectrum_files already set from simulation

        fit_results = fitter.fit_multiple(spectrum_files)

        fitter.save_fit_log()
        summary = fitter.get_fit_summary()
        print(f"\nFit Summary:")
        print(f"  Total fits: {summary['num_fits']}")
        print(f"  Successful: {summary['num_successful']}")
        print(f"  Success rate: {summary['success_rate']:.1%}")

        if summary['photon_index']['mean'] is not None:
            print(f"  Photon Index: {summary['photon_index']['mean']:.3f} "
                  f"± {summary['photon_index']['std']:.3f}")
        if summary['nH']['mean'] is not None:
            print(f"  nH: {summary['nH']['mean']:.4f} "
                  f"± {summary['nH']['std']:.4f} × 10²² cm⁻²")

    # ========== ANALYSIS ==========
    print("\n" + "=" * 70)
    print("STEP 3: ANALYZING RESULTS")
    print("=" * 70)

    analyzer = ResultsAnalyzer(
        simulated_dir=args.simulated_dir,
        results_dir=args.results_dir
    )

    # Generate text report
    report = analyzer.generate_summary_report(output_file="ktbb_study_analysis_report.txt")
    print(report)

    # Export to CSV
    analyzer.export_to_csv(filename="ktbb_study_combined_results.csv")
    analyzer.export_comparison_to_csv(filename="ktbb_study_parameter_comparison.csv")

    # ========== PLOTTING ==========
    if not args.no_plots:
        print("\n" + "=" * 70)
        print("STEP 4: GENERATING PLOTS")
        print("=" * 70)

        plotter = SpectrumPlotter(output_dir=args.results_dir)

        # Generate kTbb-specific photon index distribution plots
        df = analyzer.load_all_results()
        if not df.empty and 'fit_powerlaw.PhoIndex' in df.columns:
            print("\nGenerating photon index distribution plots by kTbb...")
            saved_plots = plotter.plot_photon_index_distribution_by_ktbb(
                df,
                ktbb_values=ktbb_values,
                output_prefix='photon_index_dist',
                show=False
            )
            if saved_plots:
                print(f"Saved {len(saved_plots)} photon index distribution plots:")
                for plot_file in saved_plots:
                    print(f"  - {plot_file}")

            # Generate kTbb vs photon index comparison plot
            print("\nGenerating kTbb vs photon index comparison plot...")
            plotter.plot_ktbb_vs_photon_index(
                df,
                save_as='ktbb_vs_photon_index.png',
                show=False
            )

        # Generate kTbb-specific spectra grid plots
        print("\nGenerating spectra grid plots by kTbb...")
        saved_grids = plot_xcm_grid_by_ktbb(
            results_dir=args.results_dir,
            ktbb_values=ktbb_values,
            output_prefix='spectra_grid',
            figsize=(30, 20),
            dpi=300
        )
        if saved_grids:
            print(f"\nSaved {len(saved_grids)} spectra grid plots:")
            for plot_file in saved_grids:
                print(f"  - {plot_file}")

    print("\n" + "=" * 70)
    print("kTbb STUDY PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nResults saved in: {args.results_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

