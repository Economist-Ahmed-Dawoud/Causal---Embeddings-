"""
Master Pipeline Script for Heterogeneous Treatment Effects Analysis
====================================================================

Version: 1.0.0
Date: 2025-11-20

This script runs the complete HTE analysis pipeline:
1. Data generation with heterogeneous effects (3 scenarios)
2. Bambi hierarchical models (random intercepts, slopes, interactions)
3. DoubleML HTE methods (causal forest, meta-learners)
4. Validation and comparison
5. Summary report generation

Usage:
------
# Run all scenarios
python run_hte_pipeline.py --all

# Run specific scenario
python run_hte_pipeline.py --scenario scenario1

# Run only specific steps
python run_hte_pipeline.py --scenario scenario1 --steps data bambi

# Skip data generation (if already done)
python run_hte_pipeline.py --scenario scenario1 --skip-data

Author: Part 2 - HTE Analysis
"""

import subprocess
import argparse
import sys
from pathlib import Path
import time

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCENARIOS = ['scenario1', 'scenario2', 'scenario3']
SCENARIO_NAMES = {
    'scenario1': 'Ability-Based (Linear)',
    'scenario2': 'Market Demand (Non-Linear)',
    'scenario3': 'Multi-Dimensional (Realistic)'
}

STEPS = ['data', 'bambi', 'doubleml', 'validation']

# ==============================================================================
# PIPELINE FUNCTIONS
# ==============================================================================

def run_command(command: list, description: str):
    """Run a command and handle errors."""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70)
    print(f"Command: {' '.join(command)}\n")

    start_time = time.time()

    try:
        result = subprocess.run(command, check=True, capture_output=False)
        elapsed_time = time.time() - start_time
        print(f"\n✓ {description} completed in {elapsed_time:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {description} FAILED after {elapsed_time:.1f}s")
        print(f"Error: {e}")
        return False

def run_data_generation():
    """Step 1: Generate data with heterogeneous effects."""
    src_dir = Path(__file__).parent / 'src'
    command = ['python', str(src_dir / 'data_generation_hte.py')]
    return run_command(command, "Data Generation (All Scenarios)")

def run_bambi(scenario: str):
    """Step 2: Run Bambi hierarchical models."""
    src_dir = Path(__file__).parent / 'src'
    command = ['python', str(src_dir / 'bambi_hierarchical_models.py'), '--scenario', scenario]
    return run_command(command, f"Bambi Hierarchical Models ({SCENARIO_NAMES[scenario]})")

def run_doubleml(scenario: str):
    """Step 3: Run DoubleML HTE methods."""
    src_dir = Path(__file__).parent / 'src'
    command = ['python', str(src_dir / 'doubleml_hte_methods.py'), '--scenario', scenario]
    return run_command(command, f"DoubleML HTE Methods ({SCENARIO_NAMES[scenario]})")

def run_validation(scenario: str):
    """Step 4: Run validation and comparison."""
    src_dir = Path(__file__).parent / 'src'
    command = ['python', str(src_dir / 'validation_metrics.py'), '--scenario', scenario]
    return run_command(command, f"Validation & Comparison ({SCENARIO_NAMES[scenario]})")

def generate_summary_report():
    """Generate final summary report."""
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)

    results_dir = Path(__file__).parent / 'results'

    # Collect all metrics
    summary = []

    for scenario in SCENARIOS:
        metrics_path = results_dir / f'validation_metrics_{scenario}.json'
        if metrics_path.exists():
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                summary.append({
                    'scenario': SCENARIO_NAMES[scenario],
                    'metrics': metrics
                })

    if summary:
        print("\n✓ Summary report generated")
        print("\nSCENARIO COMPARISON:")
        print("-"*70)
        for item in summary:
            print(f"\n{item['scenario']}:")
            if isinstance(item['metrics'], list):
                for method_metrics in item['metrics']:
                    if 'method' in method_metrics:
                        print(f"  {method_metrics['method']}:")
                        print(f"    RMSE: {method_metrics.get('rmse', 'N/A'):.4f}")
                        print(f"    Rank Corr: {method_metrics.get('spearman_correlation', 'N/A'):.4f}")
        print("-"*70)
    else:
        print("\n  No validation metrics found. Run validation first.")

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main(args):
    """Run the complete HTE analysis pipeline."""

    print("\n" + "#"*70)
    print("# HETEROGENEOUS TREATMENT EFFECTS ANALYSIS PIPELINE")
    print("#"*70)
    print(f"\nConfiguration:")
    print(f"  Scenarios: {', '.join(args.scenarios) if args.scenarios else 'All'}")
    print(f"  Steps: {', '.join(args.steps) if args.steps else 'All'}")
    print(f"  Skip data generation: {args.skip_data}")

    # Determine scenarios to run
    scenarios_to_run = args.scenarios if args.scenarios else SCENARIOS

    # Determine steps to run
    steps_to_run = args.steps if args.steps else STEPS

    # Track success
    all_success = True

    # Step 1: Data generation (runs once for all scenarios)
    if 'data' in steps_to_run and not args.skip_data:
        success = run_data_generation()
        all_success = all_success and success
        if not success:
            print("\n✗ Data generation failed. Exiting...")
            sys.exit(1)

    # Steps 2-4: Run for each scenario
    for scenario in scenarios_to_run:
        print(f"\n\n{'#'*70}")
        print(f"# PROCESSING: {SCENARIO_NAMES[scenario]}")
        print(f"{'#'*70}")

        # Step 2: Bambi
        if 'bambi' in steps_to_run:
            success = run_bambi(scenario)
            all_success = all_success and success
            if not success and not args.continue_on_error:
                print(f"\n✗ Bambi failed for {scenario}. Exiting...")
                sys.exit(1)

        # Step 3: DoubleML
        if 'doubleml' in steps_to_run:
            success = run_doubleml(scenario)
            all_success = all_success and success
            if not success and not args.continue_on_error:
                print(f"\n✗ DoubleML failed for {scenario}. Exiting...")
                sys.exit(1)

        # Step 4: Validation
        if 'validation' in steps_to_run:
            success = run_validation(scenario)
            all_success = all_success and success
            if not success and not args.continue_on_error:
                print(f"\n✗ Validation failed for {scenario}. Exiting...")
                sys.exit(1)

    # Final summary report
    if 'validation' in steps_to_run and all_success:
        generate_summary_report()

    # Final status
    print("\n\n" + "#"*70)
    if all_success:
        print("# PIPELINE COMPLETED SUCCESSFULLY!")
        print("#"*70)
        print("\nResults location:")
        print(f"  - Data: {Path(__file__).parent / 'data'}")
        print(f"  - Results: {Path(__file__).parent / 'results'}")
        print(f"  - Figures: {Path(__file__).parent / 'results' / 'figures'}")
        print(f"  - CATE Predictions: {Path(__file__).parent / 'results' / 'cate_predictions'}")
    else:
        print("# PIPELINE COMPLETED WITH ERRORS")
        print("#"*70)
        print("\nSome steps failed. Check logs above for details.")

    print("\n")

# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run HTE analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline for all scenarios
  python run_hte_pipeline.py --all

  # Run specific scenario
  python run_hte_pipeline.py --scenario scenario1

  # Run multiple scenarios
  python run_hte_pipeline.py --scenario scenario1 scenario2

  # Run only specific steps
  python run_hte_pipeline.py --scenario scenario1 --steps bambi doubleml

  # Skip data generation (if already done)
  python run_hte_pipeline.py --all --skip-data

  # Continue on errors
  python run_hte_pipeline.py --all --continue-on-error
        """
    )

    parser.add_argument('--all', action='store_true',
                        help='Run all scenarios')
    parser.add_argument('--scenario', dest='scenarios', nargs='+',
                        choices=SCENARIOS + ['all'],
                        help='Specific scenario(s) to run')
    parser.add_argument('--steps', nargs='+',
                        choices=STEPS,
                        help='Specific steps to run (default: all)')
    parser.add_argument('--skip-data', action='store_true',
                        help='Skip data generation step')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue pipeline even if a step fails')

    args = parser.parse_args()

    # Handle --all flag
    if args.all or (args.scenarios and 'all' in args.scenarios):
        args.scenarios = SCENARIOS

    # Validate arguments
    if not args.scenarios:
        parser.print_help()
        print("\nError: Must specify --all or --scenario")
        sys.exit(1)

    main(args)
