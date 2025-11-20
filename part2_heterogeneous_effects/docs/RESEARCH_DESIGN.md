# Part 2: Heterogeneous Treatment Effects Study
## Research Design Document

---

## I. EXECUTIVE SUMMARY

This study extends our causal inference analysis of a job training program for Egyptian freelancers by investigating **heterogeneous treatment effects (HTE)**. While Part 1 established that text embeddings can successfully control for unobserved confounding and recover average treatment effects, Part 2 asks: **Does the program work differently for different people?**

We employ two complementary methodological approaches:
1. **Bayesian Hierarchical Models (Bambi/PyMC)**: Partial pooling with random effects to model group-level heterogeneity
2. **Double Machine Learning HTE (DoubleML)**: Causal forests and meta-learners for flexible, individualized effect estimation

---

## II. RESEARCH QUESTIONS

### Primary Questions:
1. **Does the treatment effect vary significantly across individuals?**
   - Is there evidence of effect heterogeneity beyond sampling variability?
   - What is the distribution of individual-level treatment effects (CATE)?

2. **Which characteristics moderate the treatment effect?**
   - Do certain demographic, human capital, or platform characteristics predict larger/smaller effects?
   - Can we identify key effect modifiers?

3. **Can we identify optimal targeting rules?**
   - Which subgroups benefit most from the program?
   - For whom is the program least effective or potentially harmful?

### Secondary Questions:
4. **Do text embeddings capture effect modification?**
   - Beyond confounding control, do embeddings predict heterogeneous responses?
   - What dimensions of writing quality moderate treatment effects?

5. **How do HTE estimates compare across methods?**
   - Do Bayesian and ML approaches identify similar heterogeneity patterns?
   - What are the trade-offs in precision, interpretability, and validity?

---

## III. THEORETICAL FRAMEWORK

### A. Sources of Heterogeneity

We posit **three distinct mechanisms** of effect heterogeneity:

#### 1. **Skill Complementarity** (Ability-Based)
- **Hypothesis**: Treatment effects vary by baseline ability
- **Direction**: Ambiguous
  - *Floor effects*: Low-ability workers have more room for improvement → larger effects
  - *Ceiling effects*: High-ability workers already near optimal → smaller effects
  - *Complementarity*: High-ability workers better exploit training → larger effects
- **Variables**: `ability_score` (ground truth), text embeddings (proxy)

#### 2. **Labor Market Constraints** (Demand-Based)
- **Hypothesis**: Treatment effects vary by market conditions
- **Direction**: Positive moderation expected
  - High-demand markets: More opportunities to apply new skills → larger effects
  - Low-demand markets: Limited job availability limits skill utilization → smaller effects
- **Variables**: `market_demand_score`, `category`

#### 3. **Human Capital Accumulation** (Experience-Based)
- **Hypothesis**: Treatment effects vary by career stage
- **Direction**: Negative moderation expected
  - Early career: Foundation-building, steeper learning curves → larger effects
  - Late career: Diminishing returns to additional training → smaller effects
- **Variables**: `years_experience`, `age`, `total_jobs`

### B. Conceptual Model

```
Individual Characteristics (X)
  ├─ Ability (latent)
  ├─ Demographics (age, education, city)
  ├─ Human Capital (experience, skills, certifications)
  └─ Market Context (demand, category)
        ↓
Effect Modification (X × D)
        ↓
Heterogeneous Outcomes (Y)
```

**Key Insight**: Traditional causal inference methods estimate average effects, pooling over individual variation. HTE methods **disaggregate** this average to reveal **distributional heterogeneity**.

---

## IV. DATA GENERATING PROCESS WITH HETEROGENEITY

### A. Heterogeneity Specifications

We implement **three data generating scenarios** to test method robustness:

#### **Scenario 1: Ability-Based Heterogeneity (Linear)**
```python
# Treatment effect increases with ability (skill complementarity)
τ(X) = 3.0 + 2.5 × ability_score
# Range: [3.0 - 7.5, 3.0 + 7.5] = [-4.5, 10.5] for ability ∈ [-3, 3]
# ATE: 3.0
```

**Rationale**: Tests if methods can detect smooth, monotonic heterogeneity. High-ability workers benefit more from training (complementarity dominates ceiling effects).

#### **Scenario 2: Market Demand Heterogeneity (Non-Linear)**
```python
# Treatment effect amplified in high-demand markets
τ(X) = 5.0 + 8.0 × (market_demand_score - 0.5)²
# Range: [3.0, 7.0] for demand ∈ [0, 1]
# ATE: ~5.7
```

**Rationale**: Tests detection of non-monotonic patterns. U-shaped effect where extreme demand conditions (very low or very high) amplify treatment effectiveness.

#### **Scenario 3: Multi-Dimensional Heterogeneity (Realistic)**
```python
# Complex interaction of ability, demand, and experience
τ(X) = 5.0
     + 1.5 × ability_score                           # Complementarity
     + 3.0 × (market_demand_score > 0.7)             # High-demand boost
     - 0.15 × years_experience                       # Diminishing returns
     + 2.0 × (education_level == 'Bachelor')         # Education premium
     - 1.0 × (ability_score > 1.0) × (experience > 15) # Ceiling effect for experts
# Range: [-3, 13]
# ATE: ~5.0
```

**Rationale**: Realistic scenario combining multiple mechanisms. Tests methods' ability to capture complex, multi-variate heterogeneity.

### B. Ground Truth CATE

For each scenario, we **save the true CATE** for every individual:
```python
data['true_cate_scenario1'] = 3.0 + 2.5 * data['ability_score']
data['true_cate_scenario2'] = 5.0 + 8.0 * (data['market_demand_score'] - 0.5)**2
data['true_cate_scenario3'] = [complex formula above]
```

This enables **direct validation** of HTE estimates against ground truth.

---

## V. METHODS

### A. Bayesian Hierarchical Models (Bambi)

#### **Model 1: Random Intercepts (Partial Pooling by Group)**

```python
# Groups: city (16 levels), category (14 levels), education (6 levels)
formula = """
hourly_earnings ~
    program_participation +                    # Fixed effect (ATE)
    age + years_experience +                   # Fixed controls
    (1 | city) +                               # Random intercept by city
    (1 | category) +                           # Random intercept by category
    (1 | education_level) +                    # Random intercept by education
    pca_1 + pca_2 + ... + pca_10              # Embedding controls
"""
```

**Key Features:**
- **Partial pooling**: Group-level effects shrink toward global mean (James-Stein estimator)
- **Uncertainty quantification**: Full posterior distributions for each group
- **Handles small groups**: Borrows strength across groups to stabilize estimates

**Output:**
- Group-specific average treatment effects: $\tau_g = \beta_D + \alpha_g$
- Posterior SDs quantify uncertainty
- Variance components: $\sigma^2_{\text{city}}, \sigma^2_{\text{category}}, \sigma^2_{\text{education}}$

#### **Model 2: Random Slopes (Effect Modification)**

```python
formula = """
hourly_earnings ~
    program_participation +
    age + years_experience +
    (1 + program_participation | category) +   # Random slope: effect varies by category
    (1 | city) +
    (1 | education_level) +
    pca_1 + pca_2 + ... + pca_10
"""
```

**Key Features:**
- **Effect modification**: Each category has its own treatment effect
- **Correlation structure**: Models correlation between baseline and treatment effect
- **Flexible heterogeneity**: Continuous distribution of effects across groups

**Output:**
- Category-specific CATEs: $\tau_c = \beta_D + \beta_{D,c}$
- Correlation: $\rho(\alpha_c, \beta_{D,c})$ (e.g., do high-earning categories benefit more?)

#### **Model 3: Cross-Level Interactions (Continuous Moderators)**

```python
formula = """
hourly_earnings ~
    program_participation +
    program_participation:pca_1 +              # Interaction with ability proxy
    program_participation:years_experience +   # Interaction with experience
    program_participation:market_demand_score +# Interaction with demand
    age + years_experience + market_demand_score +
    (1 | city) + (1 | category) +
    pca_1 + pca_2 + ... + pca_10
"""
```

**Key Features:**
- **Continuous moderation**: Effects vary smoothly with covariates
- **Interpretable coefficients**: $\beta_{D \times X}$ quantifies effect modification
- **Individualized prediction**: CATE(X) = $\beta_D + \beta_{D \times X} \cdot X$

**Output:**
- Marginal effect modification: $\frac{\partial \tau}{\partial X}$
- Individual-level CATE predictions with credible intervals

---

### B. Double Machine Learning for HTE (DoubleML)

#### **Method 1: Generic Machine Learning with Effect Modifiers**

```python
from doubleml import DoubleMLPLR, DoubleMLCATEs

# Specify effect modifiers
effect_modifiers = ['ability_score', 'market_demand_score', 'years_experience',
                     'pca_1', 'pca_2', 'pca_3']

# Fit nuisance functions
dml_cate = DoubleMLCATEs(
    data,
    ml_g=RandomForestRegressor(),  # E[Y|X,D]
    ml_m=RandomForestClassifier(), # E[D|X]
    effect_modifiers=effect_modifiers
)
```

**Key Features:**
- **Two-stage residualization**: Remove confounding before estimating heterogeneity
- **Neyman orthogonality**: Robust to nuisance function misspecification
- **Flexible ML**: Random forests, gradient boosting, neural networks

**Output:**
- Effect modifier coefficients: $\gamma_k$ in $\tau(X) = \tau_0 + \sum_k \gamma_k X_k$
- Best linear projection of CATE onto modifiers

#### **Method 2: Causal Forest (Generalized Random Forest)**

```python
from econml.dml import CausalForestDML

cf_model = CausalForestDML(
    model_t=RandomForestClassifier(),  # Propensity model
    model_y=RandomForestRegressor(),   # Outcome model
    n_estimators=4000,
    min_samples_leaf=20,
    max_depth=None,
    honest=True                        # Honest splitting for valid inference
)

cf_model.fit(Y, T, X=X, W=W)
cate_pred = cf_model.effect(X)
```

**Key Features:**
- **Fully non-parametric**: No functional form assumptions on $\tau(X)$
- **Adaptive partitioning**: Finds splits that maximize treatment effect heterogeneity
- **Valid inference**: Honest forests provide confidence intervals for CATE
- **Variable importance**: Identifies which features drive heterogeneity

**Output:**
- Individual-level CATE predictions: $\hat{\tau}(X_i)$ for all i
- 95% confidence intervals via asymptotic normality
- Feature importance scores

#### **Method 3: Meta-Learners**

```python
from econml.metalearners import TLearner, XLearner

# T-Learner: Separate models for treated and control
t_learner = TLearner(models=RandomForestRegressor())
t_learner.fit(Y, T, X=X)
cate_t = t_learner.effect(X)

# X-Learner: Imputes counterfactuals, weighted by propensity
x_learner = XLearner(models=RandomForestRegressor(),
                      propensity_model=RandomForestClassifier())
x_learner.fit(Y, T, X=X)
cate_x = x_learner.effect(X)
```

**Key Features:**
- **T-Learner**: Simple, interpretable, but inefficient with imbalanced treatment
- **X-Learner**: Better performance with imbalanced data (our case: 54% treated)
- **S-Learner**: Single model with treatment indicator (baseline comparison)

**Output:**
- Individual-level CATE predictions
- Comparison of meta-learner architectures

---

## VI. VALIDATION STRATEGY

### A. Oracle Validation (Ground Truth)

Since we have true CATE, we assess:

1. **Mean Squared Error (MSE)**:
   $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (\hat{\tau}(X_i) - \tau(X_i))^2$$

2. **Rank Correlation** (Spearman's $\rho$):
   - Do methods correctly rank individuals by treatment effect?
   - More robust to scale differences than MSE

3. **Subgroup Calibration**:
   - Divide individuals into quintiles by predicted CATE
   - Compare mean predicted CATE vs mean true CATE per quintile
   - Well-calibrated methods should align closely

### B. Cross-Method Validation

Compare estimates across methods:

1. **Agreement**: Correlation between Bayesian and ML CATE predictions
2. **Uncertainty**: Do Bayesian credible intervals contain ML point estimates?
3. **Qualitative patterns**: Do methods identify the same effect modifiers?

### C. Policy Validation (AUTOC/QINI Curves)

Simulate targeting policies:

1. **Area Under TOC (AUTOC)**:
   - Rank individuals by predicted CATE (high to low)
   - Calculate cumulative gain from treating top k%
   - Compare to random allocation and oracle ranking

2. **QINI Coefficient**:
   - Measures uplift modeling performance
   - Higher values = better identification of high-effect individuals

---

## VII. ANALYSIS PLAN

### Phase 1: Data Generation & Descriptive Analysis
1. Generate data with three heterogeneity scenarios (n=5,000 each)
2. Descriptive statistics and balance checks
3. Visualize true CATE distributions

### Phase 2: Bayesian Hierarchical Estimation
1. Fit Models 1-3 in Bambi (2,000 samples, 1,000 tuning)
2. Convergence diagnostics (R-hat, ESS, trace plots)
3. Extract group-level and individual-level CATE estimates
4. Visualize posterior distributions and random effects

### Phase 3: Double ML HTE Estimation
1. Fit generic ML with effect modifiers (5-fold CV)
2. Fit causal forest (4,000 trees, honest splitting)
3. Fit meta-learners (T, S, X learners)
4. Extract individual-level CATE predictions and confidence intervals

### Phase 4: Validation & Comparison
1. Oracle validation: MSE, rank correlation, calibration plots
2. Cross-method comparison: Agreement, uncertainty quantification
3. Policy validation: AUTOC, QINI curves
4. Sensitivity analysis: Vary hyperparameters, test robustness

### Phase 5: Interpretation & Visualization
1. Variable importance analysis (which features matter most?)
2. Partial dependence plots (how do effects vary with key modifiers?)
3. Subgroup profiling (who benefits most/least?)
4. Interactive dashboards (Plotly/Streamlit for exploration)

---

## VIII. EXPECTED CONTRIBUTIONS

### Methodological:
1. **Benchmark HTE methods**: Direct comparison of Bayesian vs ML approaches with ground truth
2. **Embedding-based heterogeneity**: Test if text embeddings capture effect modification (novel!)
3. **Multi-dimensional heterogeneity**: Realistic scenario with multiple simultaneous moderators

### Substantive:
1. **Targeting insights**: Identify optimal subgroups for program allocation
2. **Mechanism discovery**: Distinguish ability vs demand vs experience mechanisms
3. **Efficiency gains**: Quantify welfare improvements from personalized vs uniform treatment

### Practical:
1. **Reusable pipeline**: Modular code for HTE analysis in similar settings
2. **Visualization toolkit**: Intuitive plots for communicating heterogeneity to stakeholders
3. **Decision support**: Translate CATE estimates into actionable targeting rules

---

## IX. LIMITATIONS & FUTURE WORK

### Limitations:
1. **Synthetic data**: Results may not generalize to real-world settings
2. **Positivity**: Strong overlap, but real data often has violations
3. **Functional form**: Linear/polynomial heterogeneity may be simplistic
4. **Computational cost**: Bayesian methods with large n or many groups

### Future Extensions:
1. **External validity**: Apply to real Upwork/freelance data
2. **Dynamic treatment effects**: How do effects evolve over time?
3. **Multiple treatments**: Compare different program variants
4. **Deep learning**: Neural networks for high-dimensional heterogeneity

---

## X. SOFTWARE & COMPUTATIONAL REQUIREMENTS

### Core Libraries:
- **Bambi** (0.13+): Bayesian modeling with partial pooling
- **PyMC** (5.0+): Backend for MCMC sampling
- **DoubleML** (0.7+): DML with HTE support
- **EconML** (0.15+): Causal forests, meta-learners (Microsoft)
- **scikit-learn** (1.3+): Nuisance models (RF, GBM)
- **pandas**, **numpy**: Data manipulation
- **matplotlib**, **seaborn**, **plotly**: Visualization

### Computational Needs:
- **Bayesian models**: ~10-30 minutes per model (depends on sampling)
- **Causal forest**: ~5-15 minutes (4,000 trees, n=5,000)
- **Memory**: ~8 GB RAM (embeddings + multiple models)
- **Parallelization**: Use joblib (sklearn), PyMC cores (Bambi)

---

## XI. FILE STRUCTURE

```
part2_heterogeneous_effects/
├── data/
│   ├── synthetic_data_hte_scenario1.parquet
│   ├── synthetic_data_hte_scenario2.parquet
│   └── synthetic_data_hte_scenario3.parquet
├── src/
│   ├── data_generation_hte.py          # Generate data with heterogeneity
│   ├── bambi_hierarchical_models.py    # Bayesian HTE estimation
│   ├── doubleml_hte_methods.py         # DML + causal forest + meta-learners
│   ├── validation_metrics.py           # MSE, rank correlation, AUTOC
│   └── visualization_utils.py          # Plotting functions
├── notebooks/
│   ├── 01_data_exploration_hte.ipynb   # EDA for heterogeneity
│   ├── 02_bayesian_hte_analysis.ipynb  # Bambi results
│   ├── 03_doubleml_hte_analysis.ipynb  # ML results
│   └── 04_comparison_validation.ipynb  # Cross-method comparison
├── results/
│   ├── cate_predictions/               # Saved CATE estimates
│   ├── figures/                        # Publication-ready plots
│   └── tables/                         # LaTeX tables
└── docs/
    ├── RESEARCH_DESIGN.md              # This document
    └── RESULTS_SUMMARY.md              # Final report (to be generated)
```

---

## XII. TIMELINE

- **Week 1**: Data generation + EDA (Phase 1)
- **Week 2**: Bayesian models (Phase 2)
- **Week 3**: Double ML methods (Phase 3)
- **Week 4**: Validation + visualization (Phases 4-5)

**Total**: ~4 weeks for full implementation and analysis

---

## XIII. REFERENCES

### Methodological:
- Künzel et al. (2019): "Metalearners for estimating heterogeneous treatment effects using machine learning"
- Wager & Athey (2018): "Estimation and inference of heterogeneous treatment effects using random forests"
- Chernozhukov et al. (2018): "Generic machine learning inference on heterogeneous treatment effects in randomized experiments"
- Gelman & Hill (2007): *Data Analysis Using Regression and Multilevel/Hierarchical Models*

### Software:
- Bambi docs: https://bambinos.github.io/bambi/
- DoubleML docs: https://docs.doubleml.org/
- EconML docs: https://econml.azurewebsites.net/

---

**Document prepared by:** Claude (AI Assistant)
**Date:** 2025-11-20
**Version:** 1.0 (Initial Research Design)
