"""
Validation Metrics and Comparison Utilities
============================================

Version: 1.0.0
Date: 2025-11-20

This script provides comprehensive validation and comparison utilities for HTE methods:
1. Oracle validation against ground truth CATE
2. Cross-method agreement and comparison
3. Calibration analysis
4. Policy evaluation (AUTOC, QINI curves)

Author: Part 2 - HTE Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Tuple
import json

# ==============================================================================
# ORACLE VALIDATION METRICS
# ==============================================================================

def compute_oracle_metrics(true_cate: np.ndarray, pred_cate: np.ndarray) -> Dict:
    """
    Compute comprehensive oracle validation metrics.

    Parameters:
    -----------
    true_cate : np.ndarray
        Ground truth CATE
    pred_cate : np.ndarray
        Predicted CATE

    Returns:
    --------
    metrics : dict
        MSE, RMSE, MAE, rank correlation, calibration metrics
    """
    # Point metrics
    mse = np.mean((true_cate - pred_cate)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_cate - pred_cate))

    # Correlation metrics
    pearson_corr, pearson_p = pearsonr(true_cate, pred_cate)
    spearman_corr, spearman_p = spearmanr(true_cate, pred_cate)

    # ATE metrics
    true_ate = true_cate.mean()
    pred_ate = pred_cate.mean()
    ate_bias = pred_ate - true_ate

    # Calibration (quintile analysis)
    calibration = compute_calibration_by_quintile(true_cate, pred_cate)

    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'pearson_correlation': float(pearson_corr),
        'pearson_pvalue': float(pearson_p),
        'spearman_correlation': float(spearman_corr),
        'spearman_pvalue': float(spearman_p),
        'true_ate': float(true_ate),
        'pred_ate': float(pred_ate),
        'ate_bias': float(ate_bias),
        'ate_bias_pct': float(ate_bias / true_ate * 100) if true_ate != 0 else 0,
        'calibration_by_quintile': calibration
    }

    return metrics

def compute_calibration_by_quintile(true_cate: np.ndarray, pred_cate: np.ndarray) -> Dict:
    """
    Assess calibration by dividing into quintiles based on predicted CATE.

    Well-calibrated methods should have predicted and true CATE align within quintiles.

    Returns:
    --------
    calibration : dict
        Mean true vs predicted CATE for each quintile
    """
    # Quintiles based on predicted CATE
    quintiles = pd.qcut(pred_cate, q=5, labels=False, duplicates='drop')

    calibration = {}
    for q in range(int(quintiles.max()) + 1):
        mask = quintiles == q
        if mask.sum() > 0:
            calibration[f'Q{q+1}'] = {
                'mean_pred_cate': float(pred_cate[mask].mean()),
                'mean_true_cate': float(true_cate[mask].mean()),
                'count': int(mask.sum())
            }

    return calibration

# ==============================================================================
# CROSS-METHOD COMPARISON
# ==============================================================================

def compare_methods(cate_predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Compute pairwise correlations and agreement across methods.

    Parameters:
    -----------
    cate_predictions : dict
        Keys are method names, values are CATE prediction arrays

    Returns:
    --------
    comparison_df : pd.DataFrame
        Pairwise correlation matrix
    """
    methods = list(cate_predictions.keys())
    n_methods = len(methods)

    # Create correlation matrix
    corr_matrix = np.zeros((n_methods, n_methods))

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i <= j:
                corr, _ = pearsonr(cate_predictions[method1], cate_predictions[method2])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

    comparison_df = pd.DataFrame(corr_matrix, index=methods, columns=methods)

    return comparison_df

# ==============================================================================
# POLICY EVALUATION (AUTOC, QINI)
# ==============================================================================

def compute_autoc_curve(true_cate: np.ndarray, pred_cate: np.ndarray,
                         n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Area Under the TOC (Targeting Operator Characteristic) curve.

    The TOC curve shows cumulative gain from targeting individuals
    with highest predicted CATE.

    Returns:
    --------
    fractions : np.ndarray
        Fraction of population targeted (0 to 1)
    cumulative_gains : np.ndarray
        Cumulative gain at each fraction
    autoc : float
        Area under TOC curve (normalized to [0, 1])
    """
    # Sort by predicted CATE (descending)
    sorted_idx = np.argsort(pred_cate)[::-1]
    sorted_true_cate = true_cate[sorted_idx]

    # Compute cumulative gains
    fractions = np.linspace(0, 1, n_points)
    cumulative_gains = []

    for frac in fractions:
        n_treated = int(frac * len(sorted_true_cate))
        if n_treated == 0:
            cumulative_gains.append(0)
        else:
            gain = sorted_true_cate[:n_treated].sum()
            cumulative_gains.append(gain)

    cumulative_gains = np.array(cumulative_gains)

    # Normalize by random allocation
    random_gain = true_cate.sum()
    cumulative_gains_norm = cumulative_gains / random_gain if random_gain != 0 else cumulative_gains

    # Compute AUTOC (area under curve)
    autoc = np.trapz(cumulative_gains_norm, fractions)

    return fractions, cumulative_gains_norm, autoc

def compute_qini_coefficient(true_cate: np.ndarray, pred_cate: np.ndarray) -> float:
    """
    Compute QINI coefficient for uplift modeling performance.

    QINI measures how much better the targeting policy is compared to random.

    Returns:
    --------
    qini : float
        QINI coefficient (higher is better)
    """
    # Sort by predicted CATE
    sorted_idx = np.argsort(pred_cate)[::-1]
    sorted_true_cate = true_cate[sorted_idx]

    # Cumulative gains
    cumulative_gains = np.cumsum(sorted_true_cate)

    # Random allocation baseline
    random_cumulative = np.cumsum(true_cate)

    # QINI: Area between actual and random
    qini = (cumulative_gains - random_cumulative).sum() / len(true_cate)

    return float(qini)

# ==============================================================================
# SUBGROUP ANALYSIS
# ==============================================================================

def analyze_heterogeneity_by_subgroups(df: pd.DataFrame,
                                        pred_cate_col: str,
                                        group_cols: List[str]) -> pd.DataFrame:
    """
    Analyze predicted CATE heterogeneity by subgroups.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with CATE predictions and grouping variables
    pred_cate_col : str
        Column name for predicted CATE
    group_cols : list
        Columns to group by

    Returns:
    --------
    subgroup_df : pd.DataFrame
        Mean CATE by subgroup
    """
    results = []

    for col in group_cols:
        group_stats = df.groupby(col)[pred_cate_col].agg([
            ('mean_cate', 'mean'),
            ('std_cate', 'std'),
            ('count', 'count')
        ]).reset_index()

        group_stats['group_variable'] = col
        group_stats = group_stats.rename(columns={col: 'group_value'})
        results.append(group_stats)

    subgroup_df = pd.concat(results, ignore_index=True)

    return subgroup_df

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_cate_comparison(true_cate: np.ndarray,
                          pred_cate_dict: Dict[str, np.ndarray],
                          output_path: Path,
                          scenario: str):
    """
    Create comprehensive CATE comparison plots.

    Plots:
    1. True vs Predicted scatter for each method
    2. Distribution comparison
    3. Calibration plots
    """
    n_methods = len(pred_cate_dict)
    fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))

    if n_methods == 1:
        axes = axes.reshape(-1, 1)

    methods = list(pred_cate_dict.keys())

    for i, method in enumerate(methods):
        pred_cate = pred_cate_dict[method]

        # Row 1: Scatter plot
        ax1 = axes[0, i]
        ax1.scatter(true_cate, pred_cate, alpha=0.3, s=10)
        ax1.plot([true_cate.min(), true_cate.max()],
                 [true_cate.min(), true_cate.max()],
                 'r--', lw=2, label='Perfect calibration')
        ax1.set_xlabel('True CATE ($)')
        ax1.set_ylabel('Predicted CATE ($)')
        ax1.set_title(f'{method}\n(Correlation: {pearsonr(true_cate, pred_cate)[0]:.3f})')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Row 2: Distribution comparison
        ax2 = axes[1, i]
        ax2.hist(true_cate, bins=30, alpha=0.5, label='True CATE', density=True)
        ax2.hist(pred_cate, bins=30, alpha=0.5, label='Predicted CATE', density=True)
        ax2.set_xlabel('CATE ($)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Distribution Comparison')
        ax2.legend()
        ax2.grid(alpha=0.3)

    plt.suptitle(f'CATE Comparison: {scenario.upper()}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ CATE comparison plot saved to {output_path}")

def plot_autoc_curves(true_cate: np.ndarray,
                       pred_cate_dict: Dict[str, np.ndarray],
                       output_path: Path,
                       scenario: str):
    """
    Plot AUTOC curves for all methods.
    """
    plt.figure(figsize=(10, 6))

    # Oracle (perfect ranking)
    oracle_fractions, oracle_gains, oracle_autoc = compute_autoc_curve(true_cate, true_cate)
    plt.plot(oracle_fractions, oracle_gains, 'k--', lw=2, label=f'Oracle (AUTOC={oracle_autoc:.3f})')

    # Each method
    for method, pred_cate in pred_cate_dict.items():
        fractions, gains, autoc = compute_autoc_curve(true_cate, pred_cate)
        plt.plot(fractions, gains, lw=2, label=f'{method} (AUTOC={autoc:.3f})')

    # Random baseline
    plt.plot([0, 1], [0, 1], 'gray', linestyle=':', lw=1, label='Random')

    plt.xlabel('Fraction of Population Targeted')
    plt.ylabel('Normalized Cumulative Gain')
    plt.title(f'AUTOC Curves: {scenario.upper()}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ AUTOC curves saved to {output_path}")

def plot_method_correlation_heatmap(comparison_df: pd.DataFrame,
                                     output_path: Path,
                                     scenario: str):
    """
    Plot heatmap of method correlations.
    """
    plt.figure(figsize=(8, 7))
    sns.heatmap(comparison_df, annot=True, fmt='.3f', cmap='coolwarm',
                center=0.5, vmin=0, vmax=1, square=True, cbar_kws={'label': 'Correlation'})
    plt.title(f'Method Agreement: {scenario.upper()}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Method correlation heatmap saved to {output_path}")

# ==============================================================================
# MAIN VALIDATION FUNCTION
# ==============================================================================

def run_comprehensive_validation(scenario: str):
    """
    Run comprehensive validation for a given scenario.

    Loads CATE predictions from both Bambi and DoubleML,
    computes all validation metrics, and generates comparison plots.
    """
    print("\n" + "#"*70)
    print(f"# COMPREHENSIVE VALIDATION: {scenario.upper()}")
    print("#"*70)

    # Paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results'
    cate_dir = results_dir / 'cate_predictions'
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(exist_ok=True, parents=True)

    # Load Bambi predictions
    print("\n[1/4] Loading Bambi CATE predictions...")
    bambi_path = cate_dir / f'bambi_cate_predictions_{scenario}.csv'
    if bambi_path.exists():
        bambi_df = pd.read_csv(bambi_path)
        print(f"✓ Loaded Bambi predictions: {len(bambi_df)} observations")
    else:
        print(f"  Warning: Bambi predictions not found at {bambi_path}")
        bambi_df = None

    # Load DoubleML predictions
    print("\n[2/4] Loading DoubleML CATE predictions...")
    doubleml_path = cate_dir / f'doubleml_cate_predictions_{scenario}.csv'
    if doubleml_path.exists():
        doubleml_df = pd.read_csv(doubleml_path)
        print(f"✓ Loaded DoubleML predictions: {len(doubleml_df)} observations")
    else:
        print(f"  Warning: DoubleML predictions not found at {doubleml_path}")
        doubleml_df = None

    # Merge if both exist
    if bambi_df is not None and doubleml_df is not None:
        # Merge on freelancer_id
        merged_df = bambi_df.merge(doubleml_df, on='freelancer_id', suffixes=('_bambi', '_doubleml'))
        true_cate = merged_df['true_cate_bambi'].values  # Should be the same in both

        # Collect all predictions
        pred_cate_dict = {
            'Bambi (Model 3)': merged_df['pred_cate_model3'].values,
            'Causal Forest': merged_df['cate_causal_forest'].values,
            'T-Learner': merged_df['cate_t_learner'].values,
            'X-Learner': merged_df['cate_x_learner'].values,
            'S-Learner': merged_df['cate_s_learner'].values
        }

        # Compute metrics for all
        print("\n[3/4] Computing validation metrics...")
        all_metrics = {}
        for method, pred_cate in pred_cate_dict.items():
            metrics = compute_oracle_metrics(true_cate, pred_cate)
            metrics['method'] = method
            all_metrics[method] = metrics
            print(f"\n  {method}:")
            print(f"    RMSE: {metrics['rmse']:.4f}")
            print(f"    Rank correlation: {metrics['spearman_correlation']:.4f}")
            print(f"    ATE bias: ${metrics['ate_bias']:.2f}")

        # Save metrics
        metrics_df = pd.DataFrame(all_metrics).T
        metrics_path = results_dir / f'validation_metrics_{scenario}.json'
        metrics_df.to_json(metrics_path, orient='records', indent=2)
        print(f"\n✓ Metrics saved to {metrics_path}")

        # Cross-method comparison
        comparison_df = compare_methods(pred_cate_dict)
        print(f"\n  Cross-method correlation:")
        print(comparison_df)

        # Create plots
        print("\n[4/4] Creating validation plots...")

        # CATE comparison
        plot_cate_comparison(true_cate, pred_cate_dict,
                              figures_dir / f'cate_comparison_{scenario}.png',
                              scenario)

        # AUTOC curves
        plot_autoc_curves(true_cate, pred_cate_dict,
                           figures_dir / f'autoc_curves_{scenario}.png',
                           scenario)

        # Method correlation heatmap
        plot_method_correlation_heatmap(comparison_df,
                                         figures_dir / f'method_correlation_{scenario}.png',
                                         scenario)

        print("\n" + "="*70)
        print("VALIDATION COMPLETE!")
        print("="*70)

    else:
        print("\n  Error: Missing predictions. Please run Bambi and DoubleML first.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run comprehensive validation')
    parser.add_argument('--scenario', type=str, default='scenario1',
                        choices=['scenario1', 'scenario2', 'scenario3'],
                        help='Which scenario to validate')
    args = parser.parse_args()

    run_comprehensive_validation(scenario=args.scenario)
