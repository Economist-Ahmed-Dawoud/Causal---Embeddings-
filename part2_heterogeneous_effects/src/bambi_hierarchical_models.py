"""
Bayesian Hierarchical Models for Heterogeneous Treatment Effects
==================================================================

Version: 1.0.0
Date: 2025-11-20

This script implements three Bayesian hierarchical models using Bambi:
1. Random Intercepts: Partial pooling by group (city, category, education)
2. Random Slopes: Treatment effect varies by category
3. Cross-Level Interactions: Continuous moderators (ability proxy, experience, demand)

Author: Part 2 - HTE Analysis
"""

import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import json
import argparse
from typing import Dict, Tuple

warnings.filterwarnings('ignore')

# Matplotlib settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# MCMC Parameters
N_DRAWS = 2000
N_TUNE = 1000
N_CHAINS = 2
TARGET_ACCEPT = 0.95

# ==============================================================================
# EMBEDDING GENERATION
# ==============================================================================

def generate_embeddings_and_pca(df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
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
    pd.DataFrame with added PCA columns
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

    return df_with_pca

# ==============================================================================
# MODEL 1: RANDOM INTERCEPTS (PARTIAL POOLING BY GROUP)
# ==============================================================================

def fit_model1_random_intercepts(df: pd.DataFrame, n_pca: int = 10) -> Tuple[bmb.Model, az.InferenceData]:
    """
    Model 1: Random Intercepts (Partial Pooling)

    Formula:
    hourly_earnings ~ program_participation + age + years_experience +
                      (1 | city) + (1 | category) + (1 | education_level) +
                      pca_1 + ... + pca_n

    Key Features:
    - Group-level intercepts for city, category, education
    - Fixed effect for treatment (ATE)
    - Embeddings control for confounding

    Returns:
    --------
    model : bambi.Model
    idata : arviz.InferenceData
    """
    print("\n" + "="*70)
    print("MODEL 1: RANDOM INTERCEPTS (PARTIAL POOLING)")
    print("="*70)

    # Build formula
    pca_terms = " + ".join([f"pca_{i}" for i in range(1, n_pca + 1)])
    formula = f"""
    hourly_earnings ~
        program_participation +
        age + years_experience +
        (1 | city) +
        (1 | category) +
        (1 | education_level) +
        {pca_terms}
    """

    print("\nFormula:")
    print(formula)

    # Fit model
    print("\nFitting model...")
    print(f"  Draws: {N_DRAWS}, Tune: {N_TUNE}, Chains: {N_CHAINS}")
    print(f"  Target accept: {TARGET_ACCEPT}")

    model = bmb.Model(formula, df, dropna=True)
    print("\nModel specification:")
    print(model)

    idata = model.fit(
        draws=N_DRAWS,
        tune=N_TUNE,
        chains=N_CHAINS,
        target_accept=TARGET_ACCEPT,
        random_seed=RANDOM_SEED
    )

    print("\n✓ Model fitted successfully!")

    # Convergence diagnostics
    print("\nConvergence Diagnostics:")
    rhat = az.rhat(idata)
    print(f"  R-hat values:")
    print(f"    Mean: {rhat.to_array().mean().values:.4f}")
    print(f"    Max: {rhat.to_array().max().values:.4f}")
    print(f"    % > 1.01: {(rhat.to_array() > 1.01).mean().values * 100:.1f}%")

    ess = az.ess(idata)
    print(f"  Effective sample size:")
    print(f"    Mean: {ess.to_array().mean().values:.0f}")
    print(f"    Min: {ess.to_array().min().values:.0f}")

    # Treatment effect estimate
    ate_posterior = idata.posterior['program_participation'].values.flatten()
    ate_mean = ate_posterior.mean()
    ate_hdi = az.hdi(idata, var_names=['program_participation'], hdi_prob=0.94)

    print(f"\nTreatment Effect (ATE):")
    print(f"  Posterior mean: ${ate_mean:.2f}")
    print(f"  94% HDI: [${ate_hdi['program_participation'].values[0]:.2f}, ${ate_hdi['program_participation'].values[1]:.2f}]")

    return model, idata

# ==============================================================================
# MODEL 2: RANDOM SLOPES (TREATMENT EFFECT VARIES BY CATEGORY)
# ==============================================================================

def fit_model2_random_slopes(df: pd.DataFrame, n_pca: int = 10) -> Tuple[bmb.Model, az.InferenceData]:
    """
    Model 2: Random Slopes (Effect Modification)

    Formula:
    hourly_earnings ~ program_participation + age + years_experience +
                      (1 + program_participation | category) +
                      (1 | city) + (1 | education_level) +
                      pca_1 + ... + pca_n

    Key Features:
    - Category-specific treatment effects (random slopes)
    - Captures heterogeneity across job categories
    - Correlation between baseline and treatment effect

    Returns:
    --------
    model : bambi.Model
    idata : arviz.InferenceData
    """
    print("\n" + "="*70)
    print("MODEL 2: RANDOM SLOPES (EFFECT MODIFICATION)")
    print("="*70)

    # Build formula
    pca_terms = " + ".join([f"pca_{i}" for i in range(1, n_pca + 1)])
    formula = f"""
    hourly_earnings ~
        program_participation +
        age + years_experience +
        (1 + program_participation | category) +
        (1 | city) +
        (1 | education_level) +
        {pca_terms}
    """

    print("\nFormula:")
    print(formula)

    # Fit model
    print("\nFitting model...")
    print(f"  Draws: {N_DRAWS}, Tune: {N_TUNE}, Chains: {N_CHAINS}")
    print(f"  Target accept: {TARGET_ACCEPT}")

    model = bmb.Model(formula, df, dropna=True)
    print("\nModel specification:")
    print(model)

    idata = model.fit(
        draws=N_DRAWS,
        tune=N_TUNE,
        chains=N_CHAINS,
        target_accept=TARGET_ACCEPT,
        random_seed=RANDOM_SEED
    )

    print("\n✓ Model fitted successfully!")

    # Convergence diagnostics
    print("\nConvergence Diagnostics:")
    rhat = az.rhat(idata)
    print(f"  R-hat values:")
    print(f"    Mean: {rhat.to_array().mean().values:.4f}")
    print(f"    Max: {rhat.to_array().max().values:.4f}")
    print(f"    % > 1.01: {(rhat.to_array() > 1.01).mean().values * 100:.1f}%")

    ess = az.ess(idata)
    print(f"  Effective sample size:")
    print(f"    Mean: {ess.to_array().mean().values:.0f}")
    print(f"    Min: {ess.to_array().min().values:.0f}")

    # Treatment effect estimates
    ate_posterior = idata.posterior['program_participation'].values.flatten()
    ate_mean = ate_posterior.mean()
    ate_hdi = az.hdi(idata, var_names=['program_participation'], hdi_prob=0.94)

    print(f"\nOverall Treatment Effect (ATE):")
    print(f"  Posterior mean: ${ate_mean:.2f}")
    print(f"  94% HDI: [${ate_hdi['program_participation'].values[0]:.2f}, ${ate_hdi['program_participation'].values[1]:.2f}]")

    # Category-specific effects (if available)
    print(f"\nCategory-Specific Treatment Effects:")
    print(f"  See posterior plots for detailed estimates by category")

    return model, idata

# ==============================================================================
# MODEL 3: CROSS-LEVEL INTERACTIONS (CONTINUOUS MODERATORS)
# ==============================================================================

def fit_model3_interactions(df: pd.DataFrame, n_pca: int = 10) -> Tuple[bmb.Model, az.InferenceData]:
    """
    Model 3: Cross-Level Interactions (Continuous Moderation)

    Formula:
    hourly_earnings ~ program_participation +
                      program_participation:pca_1 +
                      program_participation:years_experience +
                      program_participation:market_demand_score +
                      age + years_experience + market_demand_score +
                      (1 | city) + (1 | category) +
                      pca_1 + ... + pca_n

    Key Features:
    - Continuous effect modification
    - Individual-level CATE predictions
    - Tests ability, experience, demand as moderators

    Returns:
    --------
    model : bambi.Model
    idata : arviz.InferenceData
    """
    print("\n" + "="*70)
    print("MODEL 3: CROSS-LEVEL INTERACTIONS (CONTINUOUS MODERATORS)")
    print("="*70)

    # Build formula
    pca_terms = " + ".join([f"pca_{i}" for i in range(1, n_pca + 1)])
    formula = f"""
    hourly_earnings ~
        program_participation +
        program_participation:pca_1 +
        program_participation:years_experience +
        program_participation:market_demand_score +
        age + years_experience + market_demand_score +
        (1 | city) +
        (1 | category) +
        {pca_terms}
    """

    print("\nFormula:")
    print(formula)

    # Fit model
    print("\nFitting model...")
    print(f"  Draws: {N_DRAWS}, Tune: {N_TUNE}, Chains: {N_CHAINS}")
    print(f"  Target accept: {TARGET_ACCEPT}")

    model = bmb.Model(formula, df, dropna=True)
    print("\nModel specification:")
    print(model)

    idata = model.fit(
        draws=N_DRAWS,
        tune=N_TUNE,
        chains=N_CHAINS,
        target_accept=TARGET_ACCEPT,
        random_seed=RANDOM_SEED
    )

    print("\n✓ Model fitted successfully!")

    # Convergence diagnostics
    print("\nConvergence Diagnostics:")
    rhat = az.rhat(idata)
    print(f"  R-hat values:")
    print(f"    Mean: {rhat.to_array().mean().values:.4f}")
    print(f"    Max: {rhat.to_array().max().values:.4f}")
    print(f"    % > 1.01: {(rhat.to_array() > 1.01).mean().values * 100:.1f}%")

    ess = az.ess(idata)
    print(f"  Effective sample size:")
    print(f"    Mean: {ess.to_array().mean().values:.0f}")
    print(f"    Min: {ess.to_array().min().values:.0f}")

    # Treatment effect and moderation
    ate_posterior = idata.posterior['program_participation'].values.flatten()
    ate_mean = ate_posterior.mean()
    ate_hdi = az.hdi(idata, var_names=['program_participation'], hdi_prob=0.94)

    print(f"\nBase Treatment Effect:")
    print(f"  Posterior mean: ${ate_mean:.2f}")
    print(f"  94% HDI: [${ate_hdi['program_participation'].values[0]:.2f}, ${ate_hdi['program_participation'].values[1]:.2f}]")

    # Effect moderation coefficients
    interaction_vars = ['program_participation:pca_1',
                        'program_participation:years_experience',
                        'program_participation:market_demand_score']

    print(f"\nEffect Moderation (Interaction Coefficients):")
    for var in interaction_vars:
        try:
            post = idata.posterior[var].values.flatten()
            hdi = az.hdi(idata, var_names=[var], hdi_prob=0.94)
            print(f"  {var}:")
            print(f"    Mean: {post.mean():.4f}")
            print(f"    94% HDI: [{hdi[var].values[0]:.4f}, {hdi[var].values[1]:.4f}]")
        except:
            print(f"  {var}: Not available in posterior")

    return model, idata

# ==============================================================================
# CATE PREDICTION
# ==============================================================================

def predict_cate_from_model3(model: bmb.Model, idata: az.InferenceData, df: pd.DataFrame) -> np.ndarray:
    """
    Predict individual-level CATE from Model 3 (interactions).

    CATE(X) = β_D + β_{D×pca1} × pca1 + β_{D×exp} × exp + β_{D×demand} × demand

    Returns:
    --------
    cate_predictions : np.ndarray (shape: n_samples)
        Posterior mean CATE for each individual
    """
    print("\n" + "="*70)
    print("PREDICTING INDIVIDUAL-LEVEL CATE (MODEL 3)")
    print("="*70)

    # Extract posterior means
    beta_D = idata.posterior['program_participation'].values.flatten().mean()

    try:
        beta_D_pca1 = idata.posterior['program_participation:pca_1'].values.flatten().mean()
    except:
        beta_D_pca1 = 0
        print("  Warning: program_participation:pca_1 not found, using 0")

    try:
        beta_D_exp = idata.posterior['program_participation:years_experience'].values.flatten().mean()
    except:
        beta_D_exp = 0
        print("  Warning: program_participation:years_experience not found, using 0")

    try:
        beta_D_demand = idata.posterior['program_participation:market_demand_score'].values.flatten().mean()
    except:
        beta_D_demand = 0
        print("  Warning: program_participation:market_demand_score not found, using 0")

    # Compute CATE
    cate = (beta_D +
            beta_D_pca1 * df['pca_1'] +
            beta_D_exp * df['years_experience'] +
            beta_D_demand * df['market_demand_score'])

    print(f"\n✓ Predicted CATE statistics:")
    print(f"  Mean: ${cate.mean():.2f}")
    print(f"  Std: ${cate.std():.2f}")
    print(f"  Range: [${cate.min():.2f}, ${cate.max():.2f}]")

    return cate.values

# ==============================================================================
# VALIDATION METRICS
# ==============================================================================

def compute_validation_metrics(true_cate: np.ndarray, pred_cate: np.ndarray) -> Dict:
    """Compute validation metrics against ground truth CATE."""
    from scipy.stats import spearmanr

    mse = np.mean((true_cate - pred_cate)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_cate - pred_cate))
    rank_corr, pval = spearmanr(true_cate, pred_cate)

    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'rank_correlation': float(rank_corr),
        'rank_corr_pvalue': float(pval)
    }

    print("\n" + "="*70)
    print("VALIDATION METRICS (vs Ground Truth CATE)")
    print("="*70)
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Rank correlation (Spearman): {rank_corr:.4f} (p={pval:.4e})")
    print("="*70)

    return metrics

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def create_diagnostic_plots(idata: az.InferenceData, model_name: str, output_dir: Path):
    """Create diagnostic plots for model assessment."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Trace plots
    print(f"\n  Creating trace plots...")
    az.plot_trace(idata, var_names=['~'], compact=True, figsize=(12, 8))
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_trace.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Forest plot
    print(f"  Creating forest plot...")
    az.plot_forest(idata, var_names=['program_participation'], combined=True, hdi_prob=0.94)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_forest.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Posterior plot
    print(f"  Creating posterior plot...")
    az.plot_posterior(idata, var_names=['program_participation'], hdi_prob=0.94)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_posterior.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Plots saved to {output_dir}")

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
    print(f"# BAYESIAN HIERARCHICAL MODELS: {scenario.upper()}")
    print("#"*70)

    # Paths
    data_dir = Path(__file__).parent.parent / 'data'
    results_dir = Path(__file__).parent.parent / 'results'
    figures_dir = results_dir / 'figures'
    cate_dir = results_dir / 'cate_predictions'

    results_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)
    cate_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    print("\n[1/6] Loading data...")
    data_path = data_dir / f'synthetic_data_hte_{scenario}.parquet'
    df = pd.read_parquet(data_path)
    print(f"✓ Loaded {len(df)} observations from {data_path.name}")
    print(f"  True ATE: ${df['true_cate'].mean():.2f}")

    # Generate embeddings
    print("\n[2/6] Generating embeddings and PCA...")
    df = generate_embeddings_and_pca(df, n_components=10)

    # Fit models
    print("\n[3/6] Fitting Model 1 (Random Intercepts)...")
    model1, idata1 = fit_model1_random_intercepts(df, n_pca=10)

    print("\n[4/6] Fitting Model 2 (Random Slopes)...")
    model2, idata2 = fit_model2_random_slopes(df, n_pca=10)

    print("\n[5/6] Fitting Model 3 (Interactions)...")
    model3, idata3 = fit_model3_interactions(df, n_pca=10)

    # Predict CATE from Model 3
    print("\n[6/6] Predicting CATE and validating...")
    cate_pred = predict_cate_from_model3(model3, idata3, df)
    metrics = compute_validation_metrics(df['true_cate'].values, cate_pred)

    # Save results
    print("\nSaving results...")

    # Save CATE predictions
    cate_results = pd.DataFrame({
        'freelancer_id': df['freelancer_id'],
        'true_cate': df['true_cate'],
        'pred_cate_model3': cate_pred,
        'ability_score': df['ability_score'],
        'years_experience': df['years_experience'],
        'market_demand_score': df['market_demand_score'],
        'category': df['category']
    })
    cate_path = cate_dir / f'bambi_cate_predictions_{scenario}.csv'
    cate_results.to_csv(cate_path, index=False)
    print(f"✓ CATE predictions saved to {cate_path}")

    # Save metrics
    metrics['scenario'] = scenario
    metrics['true_ate'] = float(df['true_cate'].mean())
    metrics['model1_ate'] = float(idata1.posterior['program_participation'].values.flatten().mean())
    metrics['model2_ate'] = float(idata2.posterior['program_participation'].values.flatten().mean())
    metrics['model3_ate'] = float(idata3.posterior['program_participation'].values.flatten().mean())

    metrics_path = results_dir / f'bambi_metrics_{scenario}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")

    # Create diagnostic plots
    print("\nCreating diagnostic plots...")
    create_diagnostic_plots(idata1, f'model1_{scenario}', figures_dir)
    create_diagnostic_plots(idata2, f'model2_{scenario}', figures_dir)
    create_diagnostic_plots(idata3, f'model3_{scenario}', figures_dir)

    # Save InferenceData objects
    print("\nSaving InferenceData objects...")
    idata1.to_netcdf(results_dir / f'model1_{scenario}_idata.nc')
    idata2.to_netcdf(results_dir / f'model2_{scenario}_idata.nc')
    idata3.to_netcdf(results_dir / f'model3_{scenario}_idata.nc')
    print(f"✓ InferenceData saved")

    print("\n" + "="*70)
    print("BAMBI ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  - CATE predictions: {cate_path}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - InferenceData: {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit Bayesian hierarchical models for HTE')
    parser.add_argument('--scenario', type=str, default='scenario1',
                        choices=['scenario1', 'scenario2', 'scenario3'],
                        help='Which scenario to analyze')
    args = parser.parse_args()

    main(scenario=args.scenario)
