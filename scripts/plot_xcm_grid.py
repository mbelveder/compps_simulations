#!/usr/bin/env python3
"""
Command-line script to generate a grid plot of all XCM spectra.

This script finds all .xcm files in the results directory and creates
a grid plot showing fitted spectra and residuals for each scenario.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xcm_plotter import plot_xcm_grid


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate grid plot of XCM spectra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all .xcm files in default results directory
  python scripts/plot_xcm_grid.py

  # Specify custom results directory
  python scripts/plot_xcm_grid.py --results-dir data/results

  # Specify custom output filename
  python scripts/plot_xcm_grid.py --output my_spectra_grid.png

  # Custom figure size and DPI
  python scripts/plot_xcm_grid.py --figsize 24 24 --dpi 200
        """
    )

    parser.add_argument('--results-dir', type=str, default='data/results',
                       help='Directory containing .xcm files (default: data/results)')
    parser.add_argument('--output', type=str, default='spectra_grid.png',
                       help='Output filename (default: spectra_grid.png)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[20, 20],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Figure size in inches (default: 20 20)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Resolution in DPI (default: 150)')

    args = parser.parse_args()

    print("=" * 70)
    print("XCM SPECTRA GRID PLOTTER")
    print("=" * 70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output file: {args.output}")
    print(f"Figure size: {args.figsize[0]} x {args.figsize[1]} inches")
    print(f"DPI: {args.dpi}")
    print()

    try:
        output_path = plot_xcm_grid(
            results_dir=args.results_dir,
            output_file=args.output,
            figsize=tuple(args.figsize),
            dpi=args.dpi
        )

        if output_path:
            print()
            print("=" * 70)
            print("SUCCESS")
            print("=" * 70)
            print(f"Grid plot saved to: {output_path}")
            return 0
        else:
            print()
            print("=" * 70)
            print("NO FILES FOUND")
            print("=" * 70)
            print("No .xcm files found in the specified directory.")
            return 1

    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(f"Failed to create grid plot: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

