# Part 2: Heterogeneous Treatment Effects Analysis

This folder contains a complete **heterogeneous treatment effects (HTE)** study extending the causal inference analysis from Part 1.

## Overview

While Part 1 demonstrated that text embeddings can control for unobserved confounding and recover average treatment effects (ATE), Part 2 investigates **whether treatment effects vary across individuals** and **which characteristics moderate these effects**.

## Key Features

- **Three heterogeneity scenarios**: Ability-based, market-demand-based, and multi-dimensional
- **Bayesian hierarchical models**: Partial pooling with random effects (Bambi/PyMC)
- **Machine learning HTE methods**: Causal forests, meta-learners, DoubleML
- **Ground truth validation**: Direct comparison against known CATEs
- **Rich visualizations**: Interactive dashboards and publication-ready plots

## Methods Implemented

### 1. Bayesian Hierarchical Models (Bambi)
- Random intercepts by group (city, category, education)
- Random slopes for effect modification
- Cross-level interactions with continuous moderators
- Full posterior distributions with uncertainty quantification

### 2. Double Machine Learning for HTE
- Generic ML with effect modifiers
- Causal forests (via EconML)
- Meta-learners (S-learner, T-learner, X-learner)
- Honest inference with confidence intervals

## Folder Structure

```
part2_heterogeneous_effects/
├── data/                               # Generated datasets
│   ├── synthetic_data_hte_scenario1.parquet
│   ├── synthetic_data_hte_scenario2.parquet
│   └── synthetic_data_hte_scenario3.parquet
├── src/                                # Source code
│   ├── data_generation_hte.py          # Generate data with heterogeneity
│   ├── bambi_hierarchical_models.py    # Bayesian HTE estimation
│   ├── doubleml_hte_methods.py         # DML + causal forest + meta-learners
│   ├── validation_metrics.py           # Validation and comparison
│   └── visualization_utils.py          # Plotting utilities
├── notebooks/                          # Analysis notebooks
│   ├── 01_data_exploration_hte.ipynb
│   ├── 02_bayesian_hte_analysis.ipynb
│   ├── 03_doubleml_hte_analysis.ipynb
│   └── 04_comparison_validation.ipynb
├── results/                            # Outputs
│   ├── cate_predictions/
│   ├── figures/
│   └── tables/
└── docs/                               # Documentation
    ├── RESEARCH_DESIGN.md              # Detailed research design
    └── RESULTS_SUMMARY.md              # Final results (generated)
```

## Quick Start

### 1. Generate Data with Heterogeneous Effects

```bash
cd part2_heterogeneous_effects
python src/data_generation_hte.py
```

This creates three datasets with different heterogeneity patterns.

### 2. Run Bayesian HTE Analysis

```bash
python src/bambi_hierarchical_models.py --scenario 1
```

### 3. Run Double ML HTE Analysis

```bash
python src/doubleml_hte_methods.py --scenario 1
```

### 4. Explore Results in Notebooks

```bash
jupyter notebook notebooks/01_data_exploration_hte.ipynb
```

## Research Questions

1. **Does the treatment effect vary significantly across individuals?**
2. **Which characteristics moderate the treatment effect?**
3. **Can we identify optimal targeting rules?**
4. **Do text embeddings capture effect modification?**
5. **How do Bayesian and ML approaches compare?**

## Three Heterogeneity Scenarios

### Scenario 1: Ability-Based (Linear)
- Treatment effects increase with ability (skill complementarity)
- Tests detection of smooth, monotonic heterogeneity
- CATE: τ(X) = 3.0 + 2.5 × ability_score

### Scenario 2: Market Demand (Non-Linear)
- Treatment effects amplified in extreme demand conditions
- Tests detection of U-shaped heterogeneity
- CATE: τ(X) = 5.0 + 8.0 × (market_demand - 0.5)²

### Scenario 3: Multi-Dimensional (Realistic)
- Complex interactions of ability, demand, experience, education
- Tests realistic, multi-variate heterogeneity
- CATE combines multiple mechanisms (complementarity, demand, diminishing returns)

## Key Dependencies

```
bambi>=0.13.0
pymc>=5.0.0
doubleml>=0.7.0
econml>=0.15.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Validation Strategy

1. **Oracle validation**: Compare against true CATE (MSE, rank correlation)
2. **Cross-method validation**: Compare Bayesian vs ML estimates
3. **Policy validation**: AUTOC curves, QINI coefficients
4. **Calibration checks**: Predicted vs true CATE by quintile

## Expected Outputs

- Individual-level CATE predictions for all 5,000 freelancers
- Group-level treatment effects with uncertainty intervals
- Variable importance scores (which features drive heterogeneity?)
- Targeting rules for optimal program allocation
- Comprehensive validation metrics
- Publication-ready visualizations

## Citation

If you use this code, please cite:

```
[Your paper citation here]
```

## License

[Your license here]

## Contact

[Your contact information]
