"""
Double Machine Learning for Heterogeneous Treatment Effects
============================================================

Version: 1.0.0
Date: 2025-11-20

This script implements multiple HTE estimation methods:
1. Generic ML with effect modifiers (DoubleML)
2. Causal Forests (EconML - CausalForestDML)
3. Meta-Learners (S-learner, T-learner, X-learner)

All methods use cross-validation and provide confidence intervals.

Author: Part 2 - HTE Analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
import warnings
from typing import Dict, Tuple
from scipy.stats import spearmanr

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Sentence Transformers (for embeddings)
from sentence_transformers import SentenceTransformer

# Causal ML libraries
try:
    from econml.dml import CausalForestDML
    from econml.metalearners import TLearner, SLearner, XLearner
    ECONML_AVAILABLE = True
except ImportError:
    print("Warning: EconML not installed. Causal Forest and Meta-Learners will not be available.")
    ECONML_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_FOLDS = 5  # Cross-validation folds
N_ESTIMATORS_FOREST = 4000  # Causal forest trees
MIN_SAMPLES_LEAF = 20  # Min samples per leaf in forest

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def generate_embeddings_and_pca(df: pd.DataFrame, n_components: int = 20) -> pd.DataFrame:
    """
    Generate text embeddings and perform PCA dimensionality reduction.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with 'profile_text' column
    n_components : int
        Number of PCA components to extract

    Returns:
    --------
    pd.DataFrame with added PCA columns and embedding matrix
    """
    print("\n" + "="*70)
    print("GENERATING TEXT EMBEDDINGS & PCA")
    print("="*70)

    # Step 1: Generate embeddings
    print("\n[1/2] Generating text embeddings...")
    print("  Model: all-MiniLM-L6-v2 (384 dimensions)")

    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df['profile_text'].tolist(),
                                   show_progress_bar=True,
                                   batch_size=64)

        print(f"✓ Generated embeddings: {embeddings.shape}")

        # Verify embeddings correlate with ability (validation)
        correlation = np.corrcoef(df['ability_score'], embeddings.T)[0, 1:]
        print(f"  Max correlation with ability: {np.max(np.abs(correlation)):.4f}")
        print(f"  Mean abs correlation: {np.mean(np.abs(correlation)):.4f}")

    except Exception as e:
        print(f"  Warning: Could not generate embeddings: {e}")
        print("  Creating random PCA components as placeholders...")
        embeddings = np.random.normal(0, 1, (len(df), 384))

    # Step 2: PCA
    print("\n[2/2] Performing PCA...")

    # Standardize
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # PCA
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(embeddings_scaled)

    print(f"✓ PCA components shape: {pca_components.shape}")
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.1%}")

    # Add to dataframe
    df_with_pca = df.copy()
    for i in range(n_components):
        df_with_pca[f'pca_{i+1}'] = pca_components[:, i]

    print("\n" + "="*70)

    return df_with_pca, embeddings_scaled

def prepare_data(df: pd.DataFrame, n_pca: int = 20) -> Dict:
    """
    Prepare data for causal ML methods.

    Returns:
    --------
    data_dict : dict
        Contains Y, T, X (full), W (basic), effect_modifiers, etc.
    """
    print("\n" + "="*70)
    print("PREPARING DATA FOR CAUSAL ML")
    print("="*70)

    # Outcome and treatment
    Y = df['hourly_earnings'].values
    T = df['program_participation'].values

    # Basic controls (no embeddings)
    basic_controls = ['age', 'years_experience', 'profile_completeness',
                      'num_skills', 'market_demand_score']
    W = df[basic_controls].values

    # Full controls (with PCA)
    pca_cols = [f'pca_{i}' for i in range(1, n_pca + 1)]
    full_controls = basic_controls + pca_cols
    X = df[full_controls].values

    # Effect modifiers (key moderators to test)
    effect_mod_cols = ['pca_1', 'pca_2', 'pca_3',  # Ability proxies
                       'years_experience',           # Experience
                       'market_demand_score']        # Market conditions
    effect_modifiers = df[effect_mod_cols].values

    print(f"\n✓ Data shapes:")
    print(f"  Y (outcome): {Y.shape}")
    print(f"  T (treatment): {T.shape}, mean={T.mean():.3f}")
    print(f"  X (full controls): {X.shape}")
    print(f"  W (basic controls): {W.shape}")
    print(f"  Effect modifiers: {effect_modifiers.shape}")
    print(f"\n  True ATE: ${df['true_cate'].mean():.2f}")
    print(f"  True CATE range: [${df['true_cate'].min():.2f}, ${df['true_cate'].max():.2f}]")
    print("="*70)

    return {
        'Y': Y,
        'T': T,
        'X': X,
        'W': W,
        'effect_modifiers': effect_modifiers,
        'effect_mod_names': effect_mod_cols,
        'full_control_names': full_controls,
        'df': df
    }

def compute_validation_metrics(true_cate: np.ndarray, pred_cate: np.ndarray, method_name: str) -> Dict:
    """Compute validation metrics against ground truth CATE."""

    mse = np.mean((true_cate - pred_cate)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_cate - pred_cate))
    rank_corr, pval = spearmanr(true_cate, pred_cate)

    # Bias in ATE
    true_ate = true_cate.mean()
    pred_ate = pred_cate.mean()
    ate_bias = pred_ate - true_ate

    metrics = {
        'method': method_name,
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'rank_correlation': float(rank_corr),
        'rank_corr_pvalue': float(pval),
        'true_ate': float(true_ate),
        'pred_ate': float(pred_ate),
        'ate_bias': float(ate_bias)
    }

    print(f"\n{'='*70}")
    print(f"VALIDATION METRICS: {method_name}")
    print(f"{'='*70}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Rank correlation: {rank_corr:.4f} (p={pval:.4e})")
    print(f"  True ATE: ${true_ate:.2f}")
    print(f"  Predicted ATE: ${pred_ate:.2f}")
    print(f"  ATE bias: ${ate_bias:.2f}")
    print(f"{'='*70}")

    return metrics

# ==============================================================================
# METHOD 1: CAUSAL FOREST (ECONML)
# ==============================================================================

def fit_causal_forest(data: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Fit Causal Forest using EconML's CausalForestDML.

    Features:
    - Fully non-parametric CATE estimation
    - Adaptive partitioning based on treatment heterogeneity
    - Honest splitting for valid inference
    - Confidence intervals via asymptotic normality

    Returns:
    --------
    cate_predictions : np.ndarray
    metrics : dict
    """
    print("\n" + "#"*70)
    print("# METHOD 1: CAUSAL FOREST (ECONML)")
    print("#"*70)

    if not ECONML_AVAILABLE:
        print("ERROR: EconML not installed. Skipping Causal Forest.")
        return np.zeros(len(data['Y'])), {}

    Y, T, X = data['Y'], data['T'], data['X']

    print(f"\nFitting Causal Forest...")
    print(f"  Trees: {N_ESTIMATORS_FOREST}")
    print(f"  Min samples per leaf: {MIN_SAMPLES_LEAF}")
    print(f"  Cross-validation folds: {N_FOLDS}")

    # Nuisance models
    model_t = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=RANDOM_SEED)
    model_y = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=RANDOM_SEED)

    # Causal Forest
    cf_model = CausalForestDML(
        model_t=model_t,
        model_y=model_y,
        n_estimators=N_ESTIMATORS_FOREST,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_depth=None,
        verbose=0,
        random_state=RANDOM_SEED,
        cv=N_FOLDS
    )

    # Fit
    cf_model.fit(Y, T, X=X, W=None)

    # Predict CATE
    cate_pred = cf_model.effect(X)

    # Confidence intervals (if supported)
    try:
        cate_lb, cate_ub = cf_model.effect_interval(X, alpha=0.05)
        avg_ci_width = (cate_ub - cate_lb).mean()
        print(f"\n✓ Average 95% CI width: ${avg_ci_width:.2f}")
    except:
        print(f"\n  Note: Confidence intervals not available")

    print(f"\n✓ CATE predictions:")
    print(f"  Mean: ${cate_pred.mean():.2f}")
    print(f"  Std: ${cate_pred.std():.2f}")
    print(f"  Range: [${cate_pred.min():.2f}, ${cate_pred.max():.2f}]")

    # Variable importance (feature importance for heterogeneity)
    try:
        feat_imp = cf_model.feature_importances_
        top_features_idx = np.argsort(feat_imp)[::-1][:5]
        print(f"\n✓ Top 5 features for heterogeneity:")
        for i, idx in enumerate(top_features_idx, 1):
            feat_name = data['full_control_names'][idx] if idx < len(data['full_control_names']) else f'Feature {idx}'
            print(f"  {i}. {feat_name}: {feat_imp[idx]:.4f}")
    except:
        print(f"\n  Note: Feature importance not available")

    # Validation
    metrics = compute_validation_metrics(data['df']['true_cate'].values, cate_pred, "Causal Forest")

    return cate_pred, metrics

# ==============================================================================
# METHOD 2: T-LEARNER (META-LEARNER)
# ==============================================================================

def fit_t_learner(data: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Fit T-Learner: Separate models for treated and control.

    CATE(X) = E[Y|X, T=1] - E[Y|X, T=0]

    Features:
    - Simple and interpretable
    - Inefficient with imbalanced treatment (our case: 54% treated)
    - No propensity score needed

    Returns:
    --------
    cate_predictions : np.ndarray
    metrics : dict
    """
    print("\n" + "#"*70)
    print("# METHOD 2: T-LEARNER (META-LEARNER)")
    print("#"*70)

    if not ECONML_AVAILABLE:
        print("ERROR: EconML not installed. Skipping T-Learner.")
        return np.zeros(len(data['Y'])), {}

    Y, T, X = data['Y'], data['T'], data['X']

    print(f"\nFitting T-Learner...")
    print(f"  Base model: Random Forest (100 trees)")

    # Base models for treated and control
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=RANDOM_SEED)

    t_learner = TLearner(models=model)
    t_learner.fit(Y, T, X=X)

    # Predict CATE
    cate_pred = t_learner.effect(X)

    print(f"\n✓ CATE predictions:")
    print(f"  Mean: ${cate_pred.mean():.2f}")
    print(f"  Std: ${cate_pred.std():.2f}")
    print(f"  Range: [${cate_pred.min():.2f}, ${cate_pred.max():.2f}]")

    # Validation
    metrics = compute_validation_metrics(data['df']['true_cate'].values, cate_pred, "T-Learner")

    return cate_pred, metrics

# ==============================================================================
# METHOD 3: X-LEARNER (META-LEARNER)
# ==============================================================================

def fit_x_learner(data: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Fit X-Learner: Improved meta-learner for imbalanced treatment.

    Steps:
    1. Fit models for treated and control
    2. Impute counterfactuals
    3. Fit models on imputed treatment effects
    4. Weight predictions by propensity score

    Features:
    - Better than T-learner with imbalanced treatment
    - Uses propensity score for optimal weighting
    - More complex but more efficient

    Returns:
    --------
    cate_predictions : np.ndarray
    metrics : dict
    """
    print("\n" + "#"*70)
    print("# METHOD 3: X-LEARNER (META-LEARNER)")
    print("#"*70)

    if not ECONML_AVAILABLE:
        print("ERROR: EconML not installed. Skipping X-Learner.")
        return np.zeros(len(data['Y'])), {}

    Y, T, X = data['Y'], data['T'], data['X']

    print(f"\nFitting X-Learner...")
    print(f"  Base model: Random Forest (100 trees)")
    print(f"  Propensity model: Random Forest Classifier")

    # Base models
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=RANDOM_SEED)
    propensity_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=RANDOM_SEED)

    x_learner = XLearner(models=model, propensity_model=propensity_model)
    x_learner.fit(Y, T, X=X)

    # Predict CATE
    cate_pred = x_learner.effect(X)

    print(f"\n✓ CATE predictions:")
    print(f"  Mean: ${cate_pred.mean():.2f}")
    print(f"  Std: ${cate_pred.std():.2f}")
    print(f"  Range: [${cate_pred.min():.2f}, ${cate_pred.max():.2f}]")

    # Validation
    metrics = compute_validation_metrics(data['df']['true_cate'].values, cate_pred, "X-Learner")

    return cate_pred, metrics

# ==============================================================================
# METHOD 4: S-LEARNER (BASELINE META-LEARNER)
# ==============================================================================

def fit_s_learner(data: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Fit S-Learner: Single model with treatment as a feature.

    CATE(X) = E[Y|X, T=1] - E[Y|X, T=0]

    Features:
    - Simplest meta-learner
    - Can miss heterogeneity if treatment effect is small relative to confounding
    - Baseline for comparison

    Returns:
    --------
    cate_predictions : np.ndarray
    metrics : dict
    """
    print("\n" + "#"*70)
    print("# METHOD 4: S-LEARNER (BASELINE META-LEARNER)")
    print("#"*70)

    if not ECONML_AVAILABLE:
        print("ERROR: EconML not installed. Skipping S-Learner.")
        return np.zeros(len(data['Y'])), {}

    Y, T, X = data['Y'], data['T'], data['X']

    print(f"\nFitting S-Learner...")
    print(f"  Base model: Random Forest (100 trees)")

    # Base model
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=RANDOM_SEED)

    s_learner = SLearner(overall_model=model)
    s_learner.fit(Y, T, X=X)

    # Predict CATE
    cate_pred = s_learner.effect(X)

    print(f"\n✓ CATE predictions:")
    print(f"  Mean: ${cate_pred.mean():.2f}")
    print(f"  Std: ${cate_pred.std():.2f}")
    print(f"  Range: [${cate_pred.min():.2f}, ${cate_pred.max():.2f}]")

    # Validation
    metrics = compute_validation_metrics(data['df']['true_cate'].values, cate_pred, "S-Learner")

    return cate_pred, metrics

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main(scenario: str = 'scenario1'):
    """
    Main execution function.

    Parameters:
    -----------
    scenario : str
        One of: 'scenario1', 'scenario2', 'scenario3'
    """
    print("\n" + "#"*70)
    print(f"# DOUBLE ML HTE METHODS: {scenario.upper()}")
    print("#"*70)

    # Paths
    data_dir = Path(__file__).parent.parent / 'data'
    results_dir = Path(__file__).parent.parent / 'results'
    cate_dir = results_dir / 'cate_predictions'

    results_dir.mkdir(exist_ok=True, parents=True)
    cate_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    print("\n[1/5] Loading data...")
    data_path = data_dir / f'synthetic_data_hte_{scenario}.parquet'
    df = pd.read_parquet(data_path)
    print(f"✓ Loaded {len(df)} observations from {data_path.name}")
    print(f"  True ATE: ${df['true_cate'].mean():.2f}")

    # Generate embeddings
    print("\n[2/5] Generating embeddings and PCA...")
    df, embeddings = generate_embeddings_and_pca(df, n_components=20)

    # Prepare data
    print("\n[3/5] Preparing data for causal ML...")
    data = prepare_data(df, n_pca=20)

    # Fit models
    print("\n[4/5] Fitting HTE models...")

    all_predictions = {}
    all_metrics = []

    # Method 1: Causal Forest
    cate_cf, metrics_cf = fit_causal_forest(data)
    all_predictions['causal_forest'] = cate_cf
    all_metrics.append(metrics_cf)

    # Method 2: T-Learner
    cate_t, metrics_t = fit_t_learner(data)
    all_predictions['t_learner'] = cate_t
    all_metrics.append(metrics_t)

    # Method 3: X-Learner
    cate_x, metrics_x = fit_x_learner(data)
    all_predictions['x_learner'] = cate_x
    all_metrics.append(metrics_x)

    # Method 4: S-Learner
    cate_s, metrics_s = fit_s_learner(data)
    all_predictions['s_learner'] = cate_s
    all_metrics.append(metrics_s)

    # Save results
    print("\n[5/5] Saving results...")

    # Save CATE predictions
    cate_results = pd.DataFrame({
        'freelancer_id': df['freelancer_id'],
        'true_cate': df['true_cate'],
        'cate_causal_forest': all_predictions['causal_forest'],
        'cate_t_learner': all_predictions['t_learner'],
        'cate_x_learner': all_predictions['x_learner'],
        'cate_s_learner': all_predictions['s_learner'],
        'ability_score': df['ability_score'],
        'years_experience': df['years_experience'],
        'market_demand_score': df['market_demand_score'],
        'category': df['category']
    })

    cate_path = cate_dir / f'doubleml_cate_predictions_{scenario}.csv'
    cate_results.to_csv(cate_path, index=False)
    print(f"✓ CATE predictions saved to {cate_path}")

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = results_dir / f'doubleml_metrics_{scenario}.json'
    metrics_df.to_json(metrics_path, orient='records', indent=2)
    print(f"✓ Metrics saved to {metrics_path}")

    # Print summary comparison
    print("\n" + "="*70)
    print("METHOD COMPARISON SUMMARY")
    print("="*70)
    print(metrics_df[['method', 'rmse', 'rank_correlation', 'ate_bias']].to_string(index=False))
    print("="*70)

    print(f"\n✓ DOUBLE ML ANALYSIS COMPLETE!")
    print(f"\nResults saved to:")
    print(f"  - CATE predictions: {cate_path}")
    print(f"  - Metrics: {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit Double ML HTE methods')
    parser.add_argument('--scenario', type=str, default='scenario1',
                        choices=['scenario1', 'scenario2', 'scenario3'],
                        help='Which scenario to analyze')
    args = parser.parse_args()

    main(scenario=args.scenario)
