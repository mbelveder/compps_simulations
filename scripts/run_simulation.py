#!/usr/bin/env python3
"""
Main execution script for CompPS simulations.

This script orchestrates the complete workflow:
1. Simulate spectra using CompPS model
2. Fit spectra with absorbed powerlaw
3. Analyze results
4. Generate plots
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parameters import COMPPS_PARAMS, DEFAULT_NORM, DEFAULT_EXPOSURE, ENERGY_RANGE
from src.simulator import SpectrumSimulator
from src.fitter import SpectrumFitter
from src.analysis import ResultsAnalyzer
from src.plotting import SpectrumPlotter


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run CompPS simulations and fitting pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with all scenarios
  python scripts/run_simulation.py --arf nustar.arf --rmf nustar.rmf --all

  # Run specific scenarios
  python scripts/run_simulation.py --arf nustar.arf --rmf nustar.rmf \\
      --scenarios basic_thermal hot_with_reflection

  # Only simulate (no fitting)
  python scripts/run_simulation.py --arf nustar.arf --rmf nustar.rmf \\
      --all --simulate-only

  # Only fit existing spectra
  python scripts/run_simulation.py --fit-only

  # Only analyze existing results
  python scripts/run_simulation.py --analyze-only
        """
    )

    parser.add_argument('--arf', type=str,
                       help='ARF filename (in data/response/)')
    parser.add_argument('--rmf', type=str,
                       help='RMF filename (in data/response/)')

    parser.add_argument('--scenarios', nargs='+',
                       help='Specific scenarios to run (from config.parameters)')
    parser.add_argument('--all', action='store_true',
                       help='Run all scenarios defined in config')

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

    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')

    args = parser.parse_args()

    # Validate arguments
    if not (args.simulate_only or args.fit_only or args.analyze_only):
        # Full pipeline - need ARF and RMF
        if not args.arf or not args.rmf:
            parser.error("--arf and --rmf are required for simulation")

        if not (args.all or args.scenarios):
            parser.error("Either --all or --scenarios must be specified")

    # Determine which scenarios to run
    if args.all:
        scenarios_to_run = COMPPS_PARAMS
    elif args.scenarios:
        scenarios_to_run = {
            name: COMPPS_PARAMS[name]
            for name in args.scenarios
            if name in COMPPS_PARAMS
        }
        if not scenarios_to_run:
            print("Error: No valid scenarios found")
            return 1
    else:
        scenarios_to_run = {}

    print("=" * 70)
    print("CompPS SIMULATION AND FITTING PIPELINE")
    print("=" * 70)

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
            scenarios=scenarios_to_run,
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
            energy_range=(args.energy_min, args.energy_max)
        )

        # Find spectra to fit
        from src.simulator import find_simulated_spectra
        if args.fit_only:
            # Fit all existing spectra
            spectrum_files = find_simulated_spectra(args.simulated_dir)
            if not spectrum_files:
                print("Error: No simulated spectra found")
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
    report = analyzer.generate_summary_report(output_file="analysis_report.txt")
    print(report)

    # Export to CSV
    analyzer.export_to_csv()
    analyzer.export_comparison_to_csv()

    # ========== PLOTTING ==========
    if not args.no_plots:
        print("\n" + "=" * 70)
        print("STEP 4: GENERATING PLOTS")
        print("=" * 70)

        plotter = SpectrumPlotter(output_dir=args.results_dir)
        saved_plots = plotter.create_summary_plots(analyzer)

        if saved_plots:
            print(f"\nSaved {len(saved_plots)} plots:")
            for plot_file in saved_plots:
                print(f"  - {plot_file}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nResults saved in: {args.results_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

