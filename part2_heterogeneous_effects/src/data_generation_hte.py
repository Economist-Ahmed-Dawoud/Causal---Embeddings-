"""
Heterogeneous Treatment Effects Data Generation
================================================

Version: 1.0.0
Date: 2025-11-20

This script generates synthetic data with THREE heterogeneity scenarios:
1. Ability-Based Heterogeneity (Linear)
2. Market Demand Heterogeneity (Non-Linear)
3. Multi-Dimensional Heterogeneity (Realistic)

Each scenario includes GROUND TRUTH CATE for validation.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy import stats
import random
import warnings
from typing import Dict, List, Tuple
import json
from pathlib import Path
import sys

# Add parent directory to path to import from Part 1
sys.path.append(str(Path(__file__).parent.parent.parent))

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

VERSION = "1.0.0"
np.random.seed(42)
random.seed(42)

# Core parameters
N_SAMPLES = 5000
RANDOM_SEED = 42

# Standardized Egyptian cities
EGYPTIAN_CITIES = [
    'Cairo', 'Alexandria', 'Giza', 'Shubra El-Kheima', 'Port Said',
    'Suez', 'Luxor', 'Mansoura', 'El-Mahalla El-Kubra', 'Tanta',
    'Asyut', 'Ismailia', 'Fayyum', 'Zagazig', 'Aswan', 'Damietta'
]

# Standardized categories
CATEGORIES = [
    'Web Development', 'Graphic Design', 'Translation',
    'Administrative Support', 'Mobile App Development', 'Content Writing',
    'Digital Marketing', 'Video Editing', 'Data Entry', 'SEO Specialist',
    'UI/UX Design', 'Social Media Management', 'Accounting', 'Customer Support'
]

EDUCATION_LEVELS = [
    'High School', 'Some College', 'Bachelor\'s Degree',
    'Master\'s Degree', 'Professional Certificate', 'Self-Taught'
]

# Market demand scores (0-1 scale)
CATEGORY_DEMAND = {
    'Web Development': 0.90, 'Mobile App Development': 0.85, 'UI/UX Design': 0.80,
    'Digital Marketing': 0.75, 'SEO Specialist': 0.70, 'Graphic Design': 0.65,
    'Content Writing': 0.60, 'Translation': 0.55, 'Video Editing': 0.60,
    'Data Entry': 0.40, 'Administrative Support': 0.45, 'Customer Support': 0.50,
    'Social Media Management': 0.65, 'Accounting': 0.55
}

# Category effects on earnings ($)
CATEGORY_EFFECTS = {
    'Web Development': 8, 'Mobile App Development': 10, 'UI/UX Design': 7,
    'Digital Marketing': 5, 'SEO Specialist': 4, 'Graphic Design': 4,
    'Content Writing': 3, 'Translation': 3, 'Video Editing': 5,
    'Data Entry': -2, 'Administrative Support': -1, 'Customer Support': 0,
    'Social Media Management': 4, 'Accounting': 6
}

# Skill areas by category (for text generation)
SKILL_AREAS = {
    'Web Development': ['React', 'Node.js', 'Python', 'Full-stack', 'Frontend', 'Backend'],
    'Graphic Design': ['Photoshop', 'Illustrator', 'InDesign', 'branding', 'logo design'],
    'Translation': ['English-Arabic', 'French-Arabic', 'technical translation', 'localization'],
    'Administrative Support': ['scheduling', 'email management', 'data organization'],
    'Mobile App Development': ['iOS', 'Android', 'React Native', 'Flutter', 'Swift'],
    'Content Writing': ['SEO writing', 'blog posts', 'copywriting', 'technical writing'],
    'Digital Marketing': ['Facebook Ads', 'Google Ads', 'email marketing', 'analytics'],
    'Video Editing': ['Premiere Pro', 'After Effects', 'Final Cut Pro', 'motion graphics'],
    'Data Entry': ['Excel', 'data processing', 'accuracy', 'speed'],
    'SEO Specialist': ['keyword research', 'on-page SEO', 'link building', 'analytics'],
    'UI/UX Design': ['Figma', 'user research', 'wireframing', 'prototyping'],
    'Social Media Management': ['Instagram', 'Facebook', 'content strategy', 'engagement'],
    'Accounting': ['QuickBooks', 'bookkeeping', 'financial reporting', 'tax preparation'],
    'Customer Support': ['Zendesk', 'customer service', 'problem-solving', 'communication']
}

# ==============================================================================
# TEXT TEMPLATES (Copied from Part 1)
# ==============================================================================

HIGH_ABILITY_TEMPLATES = [
    "As an accomplished {category} specialist with {exp:.0f} years of expertise, I deliver transformative solutions that drive measurable ROI for my clients. Based in {city}, Egypt, I leverage cutting-edge methodologies and industry best practices to exceed expectations.",
    "I am a results-driven {category} professional operating from {city}, bringing {exp:.0f} years of proven track record in delivering high-impact projects. My approach combines technical excellence with client-centric communication, ensuring seamless collaboration.",
    "Distinguished {category} expert with {exp:.0f} years of progressive experience, I specialize in architecting innovative solutions that align with business objectives. Located in {city}, Egypt, my work ethic centers on precision and reliability.",
    "Elite {category} consultant based in {city}, offering {exp:.0f} years of specialized expertise. I consistently exceed client expectations through meticulous attention to detail, strategic problem-solving, and unwavering commitment to excellence.",
    "Seasoned {category} professional with {exp:.0f} years of demonstrable success in delivering enterprise-grade solutions. From {city}, Egypt, I bring a sophisticated blend of technical mastery and business acumen to every engagement.",
]

MED_ABILITY_TEMPLATES = [
    "I'm a {category} freelancer from {city} with {exp:.0f} years of experience. I work hard to deliver quality results and maintain good communication with clients throughout projects.",
    "Experienced {category} professional based in {city}, Egypt. With {exp:.0f} years in the field, I offer reliable service and timely delivery on all projects.",
    "I specialize in {category} and have been working in this field for {exp:.0f} years. Located in {city}, I'm committed to meeting client requirements and deadlines.",
    "{exp:.0f}-year {category} professional from {city}. I focus on delivering solid work that meets client expectations and project specifications.",
    "Based in {city}, I offer {category} services with {exp:.0f} years of practical experience. I'm dedicated to producing quality work and maintaining professional standards.",
]

LOW_ABILITY_TEMPLATES = [
    "Hi, I'm from {city} and I do {category}. I have {exp:.0f} years experience and I'm looking for work.",
    "I am a {category} worker in {city}. I have {exp:.0f} years experience. I can do the job.",
    "{category} freelancer from {city}. {exp:.0f} years working. Ready to start.",
    "I do {category} work. Based in {city}. {exp:.0f} years of experience. Contact me for projects.",
    "Looking for {category} projects. From {city}, Egypt. {exp:.0f} years in field. Available now.",
]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def generate_profile_text(ability: float, category: str, experience: float, city: str) -> str:
    """Generate ability-aware profile text."""
    # Temporarily allow randomness for template selection
    current_np_state = np.random.get_state()
    current_random_state = random.getstate()

    np.random.seed()
    random.seed()

    skill_area = random.choice(SKILL_AREAS.get(category, ['professional services']))

    if ability > 1.0:
        template = random.choice(HIGH_ABILITY_TEMPLATES)
    elif ability < -1.0:
        template = random.choice(LOW_ABILITY_TEMPLATES)
    else:
        template = random.choice(MED_ABILITY_TEMPLATES)

    text = template.format(category=category, exp=experience, city=city, skill_area=skill_area)

    # Restore original random state for reproducibility
    np.random.set_state(current_np_state)
    random.setstate(current_random_state)

    return text

def calculate_balance_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate standardized mean differences for balance assessment."""
    covariates = ['age', 'years_experience', 'profile_completeness',
                  'num_skills', 'portfolio_items', 'ability_score', 'market_demand_score']

    balance_stats = []
    for var in covariates:
        treated = df[df['program_participation'] == 1][var]
        control = df[df['program_participation'] == 0][var]

        mean_diff = treated.mean() - control.mean()
        pooled_std = np.sqrt((treated.std()**2 + control.std()**2) / 2)
        smd = mean_diff / pooled_std if pooled_std > 0 else 0

        balance_stats.append({
            'variable': var,
            'treated_mean': treated.mean(),
            'control_mean': control.mean(),
            'mean_diff': mean_diff,
            'pooled_std': pooled_std,
            'smd': smd,
            'balanced': abs(smd) < 0.1
        })

    return pd.DataFrame(balance_stats)

# ==============================================================================
# HETEROGENEOUS TREATMENT EFFECT FUNCTIONS
# ==============================================================================

def compute_cate_scenario1(ability_score: np.ndarray) -> np.ndarray:
    """
    Scenario 1: Ability-Based Heterogeneity (Linear)

    Mechanism: Skill complementarity
    Treatment effects increase with ability (high-ability workers benefit more)

    CATE(X) = 3.0 + 2.5 × ability_score
    Range: [-4.5, 10.5] for ability ∈ [-3, 3]
    ATE: 3.0 (approximately, depends on ability distribution)
    """
    cate = 3.0 + 2.5 * ability_score
    return cate

def compute_cate_scenario2(market_demand_score: np.ndarray) -> np.ndarray:
    """
    Scenario 2: Market Demand Heterogeneity (Non-Linear)

    Mechanism: Labor market constraints
    Treatment effects amplified in high-demand markets (U-shaped)

    CATE(X) = 5.0 + 8.0 × (market_demand - 0.5)²
    Range: [5.0, 7.0] for demand ∈ [0, 1]
    ATE: ~5.7 (approximately)
    """
    cate = 5.0 + 8.0 * (market_demand_score - 0.5)**2
    return cate

def compute_cate_scenario3(
    ability_score: np.ndarray,
    market_demand_score: np.ndarray,
    years_experience: np.ndarray,
    education_numeric: np.ndarray
) -> np.ndarray:
    """
    Scenario 3: Multi-Dimensional Heterogeneity (Realistic)

    Mechanisms: Multiple simultaneous moderators
    1. Ability complementarity (+1.5 per SD)
    2. High-demand market boost (+3.0 if demand > 0.7)
    3. Diminishing returns with experience (-0.15 per year)
    4. Education premium (+2.0 for Bachelor's, education_numeric ∈ [0.5, 1.5])
    5. Ceiling effect for high-ability experts (-1.0 if ability > 1 AND experience > 15)

    CATE(X) = 5.0 + [complex interactions]
    Range: [-3, 13] approximately
    ATE: ~5.0
    """
    cate = 5.0  # Base effect

    # 1. Ability complementarity (continuous)
    cate += 1.5 * ability_score

    # 2. High-demand market boost (discrete)
    cate += 3.0 * (market_demand_score > 0.7)

    # 3. Diminishing returns with experience (continuous)
    cate -= 0.15 * years_experience

    # 4. Education premium (discrete: Bachelor's level has education_numeric ∈ [0.5, 1.5])
    bachelors_indicator = ((education_numeric >= 0.5) & (education_numeric <= 1.5)).astype(float)
    cate += 2.0 * bachelors_indicator

    # 5. Ceiling effect for high-ability experts (interaction)
    ceiling_indicator = ((ability_score > 1.0) & (years_experience > 15)).astype(float)
    cate -= 1.0 * ceiling_indicator

    return cate

# ==============================================================================
# DATA GENERATION FUNCTION
# ==============================================================================

def generate_data_with_hte(scenario: str = 'scenario1') -> pd.DataFrame:
    """
    Generate complete synthetic dataset with heterogeneous treatment effects.

    Parameters:
    -----------
    scenario : str
        One of: 'scenario1', 'scenario2', 'scenario3'

    Returns:
    --------
    pd.DataFrame with true CATE column
    """
    print(f"\n{'='*70}")
    print(f"GENERATING DATA: {scenario.upper()}")
    print(f"{'='*70}\n")

    # Reset seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # 1. LATENT CONFOUNDER (Unobserved ability)
    ability_score = np.random.normal(0, 1, N_SAMPLES)
    print(f"  ✓ Latent ability: μ={ability_score.mean():.4f}, σ={ability_score.std():.4f}")

    # 2. DEMOGRAPHICS
    age = np.random.uniform(18, 50, N_SAMPLES)
    years_experience = (age - 18) * 0.6 + np.random.normal(0, 2, N_SAMPLES)
    years_experience = np.clip(years_experience, 0, None)
    years_experience = np.minimum(years_experience, age - 18)

    # Education (correlated with ability)
    education_numeric = ability_score * 0.3 + np.random.normal(0, 1, N_SAMPLES)
    education_level = pd.cut(
        education_numeric,
        bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, 2.0, np.inf],
        labels=EDUCATION_LEVELS
    ).astype(str)

    city = np.random.choice(EGYPTIAN_CITIES, N_SAMPLES, replace=True)
    category = np.random.choice(CATEGORIES, N_SAMPLES, replace=True)
    print(f"  ✓ Demographics: age={age.mean():.1f}±{age.std():.1f}, exp={years_experience.mean():.1f}±{years_experience.std():.1f}")

    # 3. PLATFORM METRICS (Correlated with ability)
    profile_completeness = np.clip(
        50 + ability_score * 15 + np.random.normal(0, 10, N_SAMPLES),
        0, 100
    )
    num_skills = np.round(
        5 + ability_score * 2 + years_experience * 0.3 + np.random.normal(0, 2, N_SAMPLES)
    ).astype(int)
    num_skills = np.clip(num_skills, 1, 50)

    portfolio_items = np.round(
        3 + ability_score * 1.5 + years_experience * 0.4 + np.random.normal(0, 2, N_SAMPLES)
    ).astype(int)
    portfolio_items = np.clip(portfolio_items, 0, 100)

    certifications = np.round(
        ability_score * 0.8 + np.random.normal(0, 1, N_SAMPLES)
    ).astype(int)
    certifications = np.clip(certifications, 0, 20)

    total_jobs = np.round(
        10 + years_experience * 5 + ability_score * 8 + np.random.normal(0, 10, N_SAMPLES)
    ).astype(int)
    total_jobs = np.clip(total_jobs, 0, 1000)

    success_rate = np.clip(
        60 + ability_score * 10 + years_experience * 1.5 + np.random.normal(0, 8, N_SAMPLES),
        0, 100
    )
    response_rate = np.clip(
        50 + ability_score * 12 + np.random.normal(0, 15, N_SAMPLES),
        0, 100
    )
    print(f"  ✓ Platform metrics: completeness={profile_completeness.mean():.1f}%, skills={num_skills.mean():.1f}")

    # 4. MARKET FACTORS
    market_demand_score = np.array([CATEGORY_DEMAND[cat] + np.random.normal(0, 0.1) for cat in category])
    market_demand_score = np.clip(market_demand_score, 0, 1)
    print(f"  ✓ Market demand: μ={market_demand_score.mean():.3f}")

    # 5. TREATMENT ASSIGNMENT (Strong selection bias - same as Part 1)
    treatment_propensity = expit(
        1.2 * ability_score +
        0.4 * years_experience / 10 +
        0.2 * (profile_completeness - 75) / 25 +
        np.random.normal(0, 0.5, N_SAMPLES)
    )
    treatment_propensity = np.clip(treatment_propensity, 0.01, 0.99)

    program_participation = (np.random.uniform(0, 1, N_SAMPLES) < treatment_propensity).astype(int)
    print(f"  ✓ Treatment: participation={program_participation.mean():.1%}, propensity∈[{treatment_propensity.min():.3f}, {treatment_propensity.max():.3f}]")

    # 6. COMPUTE TRUE CATE (Scenario-specific)
    if scenario == 'scenario1':
        true_cate = compute_cate_scenario1(ability_score)
        scenario_name = "Ability-Based (Linear)"
    elif scenario == 'scenario2':
        true_cate = compute_cate_scenario2(market_demand_score)
        scenario_name = "Market Demand (Non-Linear)"
    elif scenario == 'scenario3':
        true_cate = compute_cate_scenario3(ability_score, market_demand_score,
                                           years_experience, education_numeric)
        scenario_name = "Multi-Dimensional (Realistic)"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    true_ate = true_cate.mean()
    print(f"\n  TRUE CATE STATISTICS ({scenario_name}):")
    print(f"    ATE (mean CATE): ${true_ate:.2f}")
    print(f"    CATE range: [${true_cate.min():.2f}, ${true_cate.max():.2f}]")
    print(f"    CATE std: ${true_cate.std():.2f}")
    print(f"    CATE Q25/Q50/Q75: ${np.percentile(true_cate, 25):.2f} / ${np.percentile(true_cate, 50):.2f} / ${np.percentile(true_cate, 75):.2f}")

    # 7. OUTCOME WITH HETEROGENEOUS EFFECTS
    category_effect = np.array([CATEGORY_EFFECTS[cat] for cat in category])

    # Outcome uses INDIVIDUAL-SPECIFIC treatment effects
    hourly_earnings = (
        10 +  # Baseline
        true_cate * program_participation +  # HETEROGENEOUS EFFECT (not constant!)
        8.0 * ability_score +  # STRONG CONFOUNDING
        0.5 * years_experience +
        category_effect +
        3.0 * market_demand_score +
        0.05 * profile_completeness +
        np.random.normal(0, 2, N_SAMPLES)
    )
    hourly_earnings = np.clip(hourly_earnings, 3, 80)

    treated_earnings = hourly_earnings[program_participation == 1].mean()
    control_earnings = hourly_earnings[program_participation == 0].mean()
    naive_ate = treated_earnings - control_earnings

    print(f"\n  ✓ Earnings: μ=${hourly_earnings.mean():.2f}")
    print(f"    Treated: ${treated_earnings:.2f}")
    print(f"    Control: ${control_earnings:.2f}")
    print(f"    Naive ATE: ${naive_ate:.2f} (true ATE: ${true_ate:.2f}, bias: ${naive_ate - true_ate:.2f})")

    # 8. GENERATE TEXT PROFILES
    print("\n  Generating text profiles...")
    profile_texts = [
        generate_profile_text(ability_score[i], category[i], years_experience[i], city[i])
        for i in range(N_SAMPLES)
    ]
    print(f"  ✓ Generated {len(profile_texts):,} profiles")

    # 9. CREATE DATAFRAME
    df = pd.DataFrame({
        'freelancer_id': [f'EGY_{i+1:05d}' for i in range(N_SAMPLES)],
        'ability_score': ability_score,
        'age': age,
        'years_experience': years_experience,
        'education_level': education_level,
        'education_numeric': education_numeric,  # Save for scenario3 validation
        'city': city,
        'category': category,
        'profile_completeness': profile_completeness,
        'num_skills': num_skills,
        'portfolio_items': portfolio_items,
        'certifications': certifications,
        'total_jobs': total_jobs,
        'success_rate': success_rate,
        'response_rate': response_rate,
        'market_demand_score': market_demand_score,
        'treatment_propensity': treatment_propensity,
        'program_participation': program_participation,
        'true_cate': true_cate,  # GROUND TRUTH CATE
        'hourly_earnings': hourly_earnings,
        'profile_text': profile_texts
    })

    return df

# ==============================================================================
# QUALITY REPORT
# ==============================================================================

def generate_quality_report(df: pd.DataFrame, scenario: str) -> Dict:
    """Generate comprehensive data quality report."""
    print("\n" + "="*70)
    print(f"DATA QUALITY REPORT: {scenario.upper()}")
    print("="*70)

    report = {
        'scenario': scenario,
        'version': VERSION,
        'n_samples': len(df),
        'seed': RANDOM_SEED
    }

    # 1. Basic statistics
    print("\n1. BASIC STATISTICS")
    print(f"   Samples: {len(df):,}")
    print(f"   Variables: {len(df.columns)}")
    print(f"   Missing values: {df.isnull().sum().sum()}")

    report['basic_stats'] = {
        'n_samples': len(df),
        'n_variables': len(df.columns),
        'missing_values': int(df.isnull().sum().sum())
    }

    # 2. Treatment assignment
    print("\n2. TREATMENT ASSIGNMENT")
    treat_rate = df['program_participation'].mean()
    print(f"   Treatment rate: {treat_rate:.1%}")
    print(f"   Propensity range: [{df['treatment_propensity'].min():.3f}, {df['treatment_propensity'].max():.3f}]")

    report['treatment'] = {
        'treatment_rate': float(treat_rate),
        'propensity_min': float(df['treatment_propensity'].min()),
        'propensity_max': float(df['treatment_propensity'].max()),
        'propensity_mean': float(df['treatment_propensity'].mean())
    }

    # 3. CATE statistics
    print("\n3. TRUE CATE DISTRIBUTION")
    print(f"   ATE (mean CATE): ${df['true_cate'].mean():.2f}")
    print(f"   CATE range: [${df['true_cate'].min():.2f}, ${df['true_cate'].max():.2f}]")
    print(f"   CATE std: ${df['true_cate'].std():.2f}")
    print(f"   CATE quartiles:")
    print(f"     Q25: ${df['true_cate'].quantile(0.25):.2f}")
    print(f"     Q50: ${df['true_cate'].quantile(0.50):.2f}")
    print(f"     Q75: ${df['true_cate'].quantile(0.75):.2f}")

    report['cate_distribution'] = {
        'ate': float(df['true_cate'].mean()),
        'cate_min': float(df['true_cate'].min()),
        'cate_max': float(df['true_cate'].max()),
        'cate_std': float(df['true_cate'].std()),
        'cate_q25': float(df['true_cate'].quantile(0.25)),
        'cate_q50': float(df['true_cate'].quantile(0.50)),
        'cate_q75': float(df['true_cate'].quantile(0.75))
    }

    # 4. Outcome statistics
    print("\n4. OUTCOME STATISTICS")
    treated_earnings = df[df['program_participation'] == 1]['hourly_earnings'].mean()
    control_earnings = df[df['program_participation'] == 0]['hourly_earnings'].mean()
    naive_ate = treated_earnings - control_earnings
    true_ate = df['true_cate'].mean()
    bias = naive_ate - true_ate

    print(f"   Overall mean: ${df['hourly_earnings'].mean():.2f}")
    print(f"   Treated mean: ${treated_earnings:.2f}")
    print(f"   Control mean: ${control_earnings:.2f}")
    print(f"   Naive ATE: ${naive_ate:.2f}")
    print(f"   True ATE: ${true_ate:.2f}")
    print(f"   Naive bias: ${bias:.2f} ({bias/true_ate*100:.1f}%)")

    report['outcomes'] = {
        'overall_mean': float(df['hourly_earnings'].mean()),
        'treated_mean': float(treated_earnings),
        'control_mean': float(control_earnings),
        'naive_ate': float(naive_ate),
        'true_ate': float(true_ate),
        'bias': float(bias),
        'bias_pct': float(bias / true_ate * 100) if true_ate != 0 else 0
    }

    # 5. Balance statistics
    print("\n5. COVARIATE BALANCE")
    balance_df = calculate_balance_statistics(df)
    print(balance_df.to_string(index=False))

    report['balance'] = balance_df.to_dict('records')

    # 6. Confounding verification
    print("\n6. CONFOUNDING STRUCTURE")
    corr_ability_earnings = df['ability_score'].corr(df['hourly_earnings'])
    corr_ability_treatment = df['ability_score'].corr(df['program_participation'])
    print(f"   Corr(ability, earnings): {corr_ability_earnings:.3f}")
    print(f"   Corr(ability, treatment): {corr_ability_treatment:.3f}")

    report['confounding'] = {
        'corr_ability_earnings': float(corr_ability_earnings),
        'corr_ability_treatment': float(corr_ability_treatment)
    }

    print("\n" + "="*70)

    return report

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Generate all three scenarios."""

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Results directory for reports
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True, parents=True)

    scenarios = ['scenario1', 'scenario2', 'scenario3']
    scenario_names = {
        'scenario1': 'Ability-Based (Linear)',
        'scenario2': 'Market Demand (Non-Linear)',
        'scenario3': 'Multi-Dimensional (Realistic)'
    }

    for scenario in scenarios:
        print(f"\n\n{'#'*70}")
        print(f"# {scenario_names[scenario]}")
        print(f"{'#'*70}")

        # Generate data
        df = generate_data_with_hte(scenario)

        # Generate quality report
        report = generate_quality_report(df, scenario)

        # Save data
        output_path = output_dir / f'synthetic_data_hte_{scenario}.parquet'
        df.to_parquet(output_path, index=False)
        print(f"\n✓ Data saved to: {output_path}")

        # Save report
        report_path = results_dir / f'data_quality_report_{scenario}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Report saved to: {report_path}")

    print(f"\n\n{'='*70}")
    print("ALL SCENARIOS GENERATED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\nData files:")
    for scenario in scenarios:
        print(f"  - {output_dir}/synthetic_data_hte_{scenario}.parquet")
    print(f"\nQuality reports:")
    for scenario in scenarios:
        print(f"  - {results_dir}/data_quality_report_{scenario}.json")

if __name__ == "__main__":
    main()
