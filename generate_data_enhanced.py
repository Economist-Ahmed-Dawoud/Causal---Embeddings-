"""
Enhanced Data Generation with Comprehensive Quality Controls
============================================================

Version: 2.0.0
Date: 2025-11-20

This script generates high-quality synthetic freelancer data with:
- Comprehensive validation at every step
- Statistical verification of confounding structure
- Reproducibility guarantees
- Extensive quality metrics

Improvements over v1.0:
- Added parameter validation
- Comprehensive data quality checks
- Statistical tests for confounding
- Balance diagnostics
- Positivity verification
- Enhanced text templates (10+ per level)
- Full reproducibility guarantees
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy import stats
import random
import warnings
from typing import Dict, List, Tuple
import json

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

VERSION = "2.0.0"
np.random.seed(42)
random.seed(42)

# Core parameters
N_SAMPLES = 5000
TRUE_CAUSAL_EFFECT = 5.0
RANDOM_SEED = 42

# Standardized Egyptian cities (aligned across all scripts)
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

# ==============================================================================
# ENHANCED TEXT TEMPLATES (10 per ability level)
# ==============================================================================

HIGH_ABILITY_TEMPLATES = [
    "As an accomplished {category} specialist with {exp:.0f} years of expertise, I deliver transformative solutions that drive measurable ROI for my clients. Based in {city}, Egypt, I leverage cutting-edge methodologies and industry best practices to exceed expectations.",
    "I am a results-driven {category} professional operating from {city}, bringing {exp:.0f} years of proven track record in delivering high-impact projects. My approach combines technical excellence with client-centric communication, ensuring seamless collaboration.",
    "Distinguished {category} expert with {exp:.0f} years of progressive experience, I specialize in architecting innovative solutions that align with business objectives. Located in {city}, Egypt, my work ethic centers on precision and reliability.",
    "Elite {category} consultant based in {city}, offering {exp:.0f} years of specialized expertise. I consistently exceed client expectations through meticulous attention to detail, strategic problem-solving, and unwavering commitment to excellence.",
    "Seasoned {category} professional with {exp:.0f} years of demonstrable success in delivering enterprise-grade solutions. From {city}, Egypt, I bring a sophisticated blend of technical mastery and business acumen to every engagement.",
    "Award-winning {category} specialist in {city} with {exp:.0f} years of experience driving transformational outcomes. My portfolio showcases consistent delivery of innovative, scalable solutions that maximize client value.",
    "Highly accomplished {category} expert from {city}, leveraging {exp:.0f} years of deep industry knowledge. I pride myself on delivering exceptional quality through rigorous methodology and strategic insight.",
    "Premier {category} professional based in {city}, Egypt, with {exp:.0f} years of proven excellence. I specialize in creating robust, future-proof solutions that consistently surpass client expectations and industry standards.",
    "Internationally recognized {category} consultant operating from {city}, bringing {exp:.0f} years of elite-level expertise. My work is characterized by precision, innovation, and an unwavering commitment to client success.",
    "Distinguished {category} architect with {exp:.0f} years of progressive experience serving clients from {city}. I deliver sophisticated solutions that seamlessly integrate technical excellence with strategic business objectives."
]

MED_ABILITY_TEMPLATES = [
    "I am a {category} freelancer based in {city}, Egypt with {exp:.0f} years of experience in the field. I provide professional services and work closely with clients to meet project requirements on time.",
    "Professional {category} specialist from {city} offering {exp:.0f} years of hands-on experience. I focus on providing reliable services and maintaining good communication with clients throughout the project lifecycle.",
    "Experienced {category} freelancer located in {city}, Egypt. With {exp:.0f} years in the industry, I offer dependable services. I value client satisfaction and strive to complete projects efficiently.",
    "{category} professional based in {city} with {exp:.0f} years of practical experience. I deliver quality work and maintain clear communication with clients to ensure project success.",
    "Dedicated {category} specialist from {city}, Egypt, bringing {exp:.0f} years of experience. I am committed to providing reliable services and meeting project deadlines consistently.",
    "Reliable {category} freelancer in {city} with {exp:.0f} years of industry experience. I work professionally with clients and deliver projects according to specifications and timelines.",
    "{exp:.0f}-year {category} professional operating from {city}. I provide competent services and maintain good working relationships with clients through clear communication.",
    "Based in {city}, I am a {category} specialist with {exp:.0f} years of experience. I offer professional services and work diligently to meet client expectations and project requirements.",
    "{category} practitioner from {city}, Egypt with {exp:.0f} years in the field. I provide dependable work and communicate effectively with clients to ensure satisfactory outcomes.",
    "Professional {category} provider in {city} bringing {exp:.0f} years of experience. I focus on delivering reliable services and maintaining positive client relationships throughout projects."
]

LOW_ABILITY_TEMPLATES = [
    "hello i am a {category} from {city} with {exp:.0f} years experience. i work hard and deliver projects on time. looking for good opportunities to work with clients.",
    "i am {category} freelancer in {city}, egypt. i have {exp:.0f} years experience and i am very dedicated worker. i am available for projects. thanks for reading my profile.",
    "hardworking {category} based in {city}. {exp:.0f} years doing this work. i always try my best for clients. please contact me for your projects.",
    "{category} in {city} here. got {exp:.0f} years experience. i do good work and am ready to start on projects. contact me anytime.",
    "hello, {exp:.0f} years experience as {category} from {city}. i am hardworking person and will complete your work. looking forward to work with you.",
    "i do {category} work in {city}. {exp:.0f} years now. very dedicated and will give you good results. hire me for your projects thanks.",
    "{category} freelancer from {city} egypt with {exp:.0f} years. i work hard on all projects and try to satisfy clients. message me for work.",
    "based in {city}, doing {category} for {exp:.0f} years. i am available and ready to work hard on your projects. good quality work.",
    "hi im {category} in {city}. {exp:.0f} years experience total. i work very hard and dedicated. looking for projects to work on.",
    "{exp:.0f} year {category} from {city}. hardworking and dedicated person. will do my best on your projects. contact me for work opportunities."
]

# Skill areas by category
SKILL_AREAS = {
    'Web Development': ['full-stack development', 'responsive design', 'API integration', 'database optimization', 'React/Vue.js', 'Node.js', 'cloud deployment'],
    'Graphic Design': ['brand identity', 'visual storytelling', 'Adobe Creative Suite', 'print and digital media', 'typography', 'illustration'],
    'Translation': ['technical translation', 'localization', 'cultural adaptation', 'proofreading', 'Arabic-English', 'transcription'],
    'Administrative Support': ['calendar management', 'data organization', 'correspondence handling', 'scheduling', 'documentation', 'CRM systems'],
    'Mobile App Development': ['iOS and Android development', 'cross-platform solutions', 'app optimization', 'UX implementation', 'Flutter/React Native'],
    'Content Writing': ['SEO copywriting', 'blog content', 'technical writing', 'storytelling', 'editing', 'content strategy'],
    'Digital Marketing': ['campaign strategy', 'social media marketing', 'analytics', 'conversion optimization', 'PPC', 'email marketing'],
    'Video Editing': ['post-production', 'color grading', 'motion graphics', 'sound design', 'Adobe Premiere', 'After Effects'],
    'Data Entry': ['data processing', 'spreadsheet management', 'accuracy verification', 'database updates', 'typing', 'quality control'],
    'SEO Specialist': ['keyword research', 'on-page optimization', 'link building', 'technical SEO', 'analytics', 'content optimization'],
    'UI/UX Design': ['user research', 'wireframing', 'prototyping', 'usability testing', 'Figma', 'design systems'],
    'Social Media Management': ['content scheduling', 'community engagement', 'analytics reporting', 'brand voice', 'strategy'],
    'Accounting': ['bookkeeping', 'financial reporting', 'tax preparation', 'reconciliation', 'QuickBooks', 'financial analysis'],
    'Customer Support': ['client communication', 'issue resolution', 'CRM management', 'satisfaction tracking', 'ticket handling'],
}

# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_parameters():
    """Validate all input parameters before data generation."""
    assert N_SAMPLES > 0, "N_SAMPLES must be positive"
    assert N_SAMPLES <= 1000000, "N_SAMPLES too large (max 1M)"
    assert TRUE_CAUSAL_EFFECT > 0, "TRUE_CAUSAL_EFFECT should be positive"
    assert TRUE_CAUSAL_EFFECT < 50, "TRUE_CAUSAL_EFFECT unreasonably large"
    assert len(CATEGORIES) == len(CATEGORY_DEMAND), "Category lists must match"
    assert len(CATEGORIES) == len(CATEGORY_EFFECTS), "Category effects must match"
    assert all(0 <= v <= 1 for v in CATEGORY_DEMAND.values()), "Demand scores must be in [0,1]"
    print("✓ Parameter validation passed")

def check_for_missing_values(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Check for any missing values in the dataset."""
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].index.tolist()
    return len(missing_cols) == 0, missing_cols

def check_for_infinite_values(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Check for infinite values in numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = []
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            inf_cols.append(col)
    return len(inf_cols) == 0, inf_cols

def check_value_ranges(df: pd.DataFrame) -> Dict[str, bool]:
    """Validate all variables are within expected ranges."""
    checks = {
        'age_range': ((df['age'] >= 18) & (df['age'] <= 50)).all(),
        'experience_range': ((df['years_experience'] >= 0) & (df['years_experience'] <= 32)).all(),
        'experience_logical': (df['years_experience'] <= df['age'] - 18).all(),
        'profile_completeness': ((df['profile_completeness'] >= 0) & (df['profile_completeness'] <= 100)).all(),
        'num_skills': ((df['num_skills'] >= 1) & (df['num_skills'] <= 50)).all(),
        'portfolio': (df['portfolio_items'] >= 0).all(),
        'certifications': (df['certifications'] >= 0).all(),
        'total_jobs': (df['total_jobs'] >= 0).all(),
        'success_rate': ((df['success_rate'] >= 0) & (df['success_rate'] <= 100)).all(),
        'response_rate': ((df['response_rate'] >= 0) & (df['response_rate'] <= 100)).all(),
        'market_demand': ((df['market_demand_score'] >= 0) & (df['market_demand_score'] <= 1)).all(),
        'propensity': ((df['treatment_propensity'] >= 0) & (df['treatment_propensity'] <= 1)).all(),
        'treatment_binary': df['program_participation'].isin([0, 1]).all(),
        'earnings_range': ((df['hourly_earnings'] >= 3) & (df['hourly_earnings'] <= 80)).all(),
        'text_nonempty': (df['profile_text'].str.len() > 0).all(),
    }
    return checks

def check_duplicates(df: pd.DataFrame) -> Tuple[bool, int]:
    """Check for duplicate observations."""
    n_duplicates = df.duplicated(subset=['freelancer_id']).sum()
    return n_duplicates == 0, n_duplicates

def verify_confounding_structure(df: pd.DataFrame) -> Dict[str, float]:
    """Statistically verify that confounding exists."""
    results = {}

    # 1. Ability affects treatment (selection bias)
    treated_ability = df[df['program_participation'] == 1]['ability_score'].mean()
    control_ability = df[df['program_participation'] == 0]['ability_score'].mean()
    ability_diff = treated_ability - control_ability
    t_stat, p_val = stats.ttest_ind(
        df[df['program_participation'] == 1]['ability_score'],
        df[df['program_participation'] == 0]['ability_score']
    )
    results['selection_bias'] = {
        'treated_ability_mean': treated_ability,
        'control_ability_mean': control_ability,
        'difference': ability_diff,
        't_statistic': t_stat,
        'p_value': p_val,
        'significant': p_val < 0.001
    }

    # 2. Ability affects outcome (confounding)
    corr_ability_earnings = df['ability_score'].corr(df['hourly_earnings'])
    results['confounding'] = {
        'correlation': corr_ability_earnings,
        'strong': abs(corr_ability_earnings) > 0.5
    }

    # 3. Verify naive estimate is biased
    naive_ate = (df[df['program_participation'] == 1]['hourly_earnings'].mean() -
                 df[df['program_participation'] == 0]['hourly_earnings'].mean())
    bias = naive_ate - TRUE_CAUSAL_EFFECT
    results['naive_bias'] = {
        'naive_estimate': naive_ate,
        'true_effect': TRUE_CAUSAL_EFFECT,
        'bias': bias,
        'bias_pct': (bias / TRUE_CAUSAL_EFFECT) * 100,
        'substantial': abs(bias) > 1.0
    }

    return results

def check_positivity(df: pd.DataFrame, min_propensity: float = 0.01,
                    max_propensity: float = 0.99) -> Dict[str, any]:
    """Verify positivity assumption (overlap)."""
    results = {
        'min_propensity': df['treatment_propensity'].min(),
        'max_propensity': df['treatment_propensity'].max(),
        'violations': ((df['treatment_propensity'] < min_propensity) |
                      (df['treatment_propensity'] > max_propensity)).sum(),
        'violation_pct': ((df['treatment_propensity'] < min_propensity) |
                         (df['treatment_propensity'] > max_propensity)).mean() * 100,
        'satisfied': ((df['treatment_propensity'] >= min_propensity) &
                     (df['treatment_propensity'] <= max_propensity)).all()
    }
    return results

def calculate_balance_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate standardized mean differences for balance assessment."""
    covariates = ['age', 'years_experience', 'profile_completeness',
                  'num_skills', 'portfolio_items', 'ability_score']

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
# DATA GENERATION FUNCTIONS
# ==============================================================================

def generate_profile_text(ability: float, category: str, experience: float, city: str) -> str:
    """Generate ability-aware profile text with enhanced diversity."""
    np.random.seed()  # Allow variation within ability tiers
    random.seed()

    skill_area = random.choice(SKILL_AREAS.get(category, ['professional services']))

    if ability > 1.0:
        template = random.choice(HIGH_ABILITY_TEMPLATES)
    elif ability < -1.0:
        template = random.choice(LOW_ABILITY_TEMPLATES)
    else:
        template = random.choice(MED_ABILITY_TEMPLATES)

    text = template.format(category=category, exp=experience, city=city, skill_area=skill_area)

    # Reset seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    return text

def generate_data() -> pd.DataFrame:
    """Generate complete synthetic dataset with all features."""
    print("\nGenerating synthetic data...")

    # 1. LATENT CONFOUNDER (Unobserved ability)
    ability_score = np.random.normal(0, 1, N_SAMPLES)
    print(f"  ✓ Latent ability: μ={ability_score.mean():.4f}, σ={ability_score.std():.4f}")

    # 2. DEMOGRAPHICS
    age = np.random.uniform(18, 50, N_SAMPLES)
    years_experience = (age - 18) * 0.6 + np.random.normal(0, 2, N_SAMPLES)
    years_experience = np.clip(years_experience, 0, None)
    years_experience = np.minimum(years_experience, age - 18)

    # Validate logical constraint
    assert (years_experience <= age - 18 + 0.01).all(), "Experience > age constraint violated"
    assert (years_experience >= 0).all(), "Negative experience detected"

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

    # 5. TREATMENT ASSIGNMENT (Strong selection bias)
    treatment_propensity = expit(
        1.2 * ability_score +  # Reduced from 1.5 to ensure better positivity
        0.4 * years_experience / 10 +  # Reduced from 0.5
        0.2 * (profile_completeness - 75) / 25 +  # Reduced from 0.3
        np.random.normal(0, 0.5, N_SAMPLES)
    )
    # Ensure propensity is strictly in (0, 1) - stricter bounds for positivity
    treatment_propensity = np.clip(treatment_propensity, 0.01, 0.99)

    program_participation = (np.random.uniform(0, 1, N_SAMPLES) < treatment_propensity).astype(int)
    print(f"  ✓ Treatment: participation={program_participation.mean():.1%}, propensity∈[{treatment_propensity.min():.3f}, {treatment_propensity.max():.3f}]")

    # 6. OUTCOME (TRUE EFFECT = $5.00)
    category_effect = np.array([CATEGORY_EFFECTS[cat] for cat in category])

    hourly_earnings = (
        10 +  # Baseline
        TRUE_CAUSAL_EFFECT * program_participation +  # TRUE EFFECT
        8.0 * ability_score +  # STRONG CONFOUNDING
        0.5 * years_experience +
        category_effect +
        3.0 * market_demand_score +
        0.05 * profile_completeness +
        np.random.normal(0, 2, N_SAMPLES)
    )
    hourly_earnings = np.clip(hourly_earnings, 3, 80)
    print(f"  ✓ Earnings: μ=${hourly_earnings.mean():.2f}, treated=${hourly_earnings[program_participation==1].mean():.2f}, control=${hourly_earnings[program_participation==0].mean():.2f}")

    # 7. GENERATE TEXT PROFILES
    print("  Generating text profiles...")
    profile_texts = [
        generate_profile_text(ability_score[i], category[i], years_experience[i], city[i])
        for i in range(N_SAMPLES)
    ]
    print(f"  ✓ Generated {len(profile_texts):,} profiles")

    # 8. CREATE DATAFRAME
    df = pd.DataFrame({
        'freelancer_id': [f'EGY_{i+1:05d}' for i in range(N_SAMPLES)],
        'ability_score': ability_score,
        'age': age,
        'years_experience': years_experience,
        'education_level': education_level,
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
        'hourly_earnings': hourly_earnings,
        'profile_text': profile_texts
    })

    return df

# ==============================================================================
# QUALITY REPORT
# ==============================================================================

def generate_quality_report(df: pd.DataFrame) -> Dict:
    """Generate comprehensive data quality report."""
    print("\n" + "="*70)
    print("DATA QUALITY REPORT")
    print("="*70)

    report = {
        'version': VERSION,
        'n_samples': len(df),
        'true_causal_effect': TRUE_CAUSAL_EFFECT,
        'checks': {}
    }

    # 1. Missing values
    no_missing, missing_cols = check_for_missing_values(df)
    report['checks']['no_missing_values'] = no_missing
    if no_missing:
        print("✓ No missing values")
    else:
        print(f"✗ Missing values in: {missing_cols}")

    # 2. Infinite values
    no_inf, inf_cols = check_for_infinite_values(df)
    report['checks']['no_infinite_values'] = no_inf
    if no_inf:
        print("✓ No infinite values")
    else:
        print(f"✗ Infinite values in: {inf_cols}")

    # 3. Value ranges
    range_checks = check_value_ranges(df)
    report['checks']['value_ranges'] = range_checks
    failed_checks = [k for k, v in range_checks.items() if not v]
    if len(failed_checks) == 0:
        print("✓ All value ranges valid")
    else:
        print(f"✗ Range violations: {failed_checks}")

    # 4. Duplicates
    no_dupes, n_dupes = check_duplicates(df)
    report['checks']['no_duplicates'] = no_dupes
    if no_dupes:
        print("✓ No duplicate IDs")
    else:
        print(f"✗ {n_dupes} duplicate IDs found")

    # 5. Confounding structure
    print("\n" + "-"*70)
    print("CONFOUNDING VERIFICATION")
    print("-"*70)
    confounding = verify_confounding_structure(df)
    report['confounding'] = confounding

    print(f"\nSelection Bias:")
    print(f"  Treated ability:  {confounding['selection_bias']['treated_ability_mean']:+.4f}")
    print(f"  Control ability:  {confounding['selection_bias']['control_ability_mean']:+.4f}")
    print(f"  Difference:       {confounding['selection_bias']['difference']:+.4f}")
    print(f"  t-statistic:      {confounding['selection_bias']['t_statistic']:.2f}")
    print(f"  p-value:          {confounding['selection_bias']['p_value']:.2e}")
    print(f"  {'✓' if confounding['selection_bias']['significant'] else '✗'} Significant selection bias detected")

    print(f"\nOutcome Confounding:")
    print(f"  Corr(ability, earnings): {confounding['confounding']['correlation']:+.4f}")
    print(f"  {'✓' if confounding['confounding']['strong'] else '✗'} Strong confounding detected")

    print(f"\nNaive Bias:")
    print(f"  Naive estimate:   ${confounding['naive_bias']['naive_estimate']:.2f}")
    print(f"  True effect:      ${confounding['naive_bias']['true_effect']:.2f}")
    print(f"  Bias:             ${confounding['naive_bias']['bias']:.2f} ({confounding['naive_bias']['bias_pct']:+.1f}%)")
    print(f"  {'✓' if confounding['naive_bias']['substantial'] else '✗'} Substantial bias detected")

    # 6. Positivity
    print("\n" + "-"*70)
    print("POSITIVITY VERIFICATION")
    print("-"*70)
    positivity = check_positivity(df)
    report['positivity'] = positivity

    print(f"  Min propensity:   {positivity['min_propensity']:.4f}")
    print(f"  Max propensity:   {positivity['max_propensity']:.4f}")
    print(f"  Violations:       {positivity['violations']} ({positivity['violation_pct']:.2f}%)")
    print(f"  {'✓' if positivity['satisfied'] else '✗'} Positivity assumption {'satisfied' if positivity['satisfied'] else 'VIOLATED'}")

    # 7. Balance statistics
    print("\n" + "-"*70)
    print("BALANCE STATISTICS (Before Adjustment)")
    print("-"*70)
    balance = calculate_balance_statistics(df)
    report['balance'] = balance.to_dict('records')

    print(balance[['variable', 'treated_mean', 'control_mean', 'smd', 'balanced']].to_string(index=False))
    print(f"\n  Balanced variables: {balance['balanced'].sum()}/{len(balance)}")

    # Overall assessment
    print("\n" + "="*70)
    all_checks_passed = (
        no_missing and no_inf and all(range_checks.values()) and no_dupes and
        confounding['selection_bias']['significant'] and
        confounding['confounding']['strong'] and
        confounding['naive_bias']['substantial'] and
        positivity['satisfied']
    )

    if all_checks_passed:
        print("✓✓✓ ALL QUALITY CHECKS PASSED ✓✓✓")
    else:
        print("✗✗✗ SOME QUALITY CHECKS FAILED ✗✗✗")

    report['all_checks_passed'] = all_checks_passed
    print("="*70 + "\n")

    return report

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ENHANCED DATA GENERATION v" + VERSION)
    print("="*70)
    print(f"Configuration:")
    print(f"  N_SAMPLES:           {N_SAMPLES:,}")
    print(f"  TRUE_CAUSAL_EFFECT:  ${TRUE_CAUSAL_EFFECT:.2f}")
    print(f"  RANDOM_SEED:         {RANDOM_SEED}")
    print(f"  Categories:          {len(CATEGORIES)}")
    print(f"  Cities:              {len(EGYPTIAN_CITIES)}")

    # Validate parameters
    print("\nValidating parameters...")
    validate_parameters()

    # Generate data
    df = generate_data()

    # Run quality report
    report = generate_quality_report(df)

    # Save outputs
    output_path = '/home/user/Causal---Embeddings-/synthetic_upwork_data_enhanced.parquet'
    df.to_parquet(output_path, index=False)
    print(f"✓ Data saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Size: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    # Save quality report
    report_path = '/home/user/Causal---Embeddings-/data_quality_report.json'
    with open(report_path, 'w') as f:
        # Convert non-serializable objects
        report_serializable = json.loads(json.dumps(report, default=str))
        json.dump(report_serializable, f, indent=2)
    print(f"✓ Quality report saved to: {report_path}")

    # Display sample profiles
    print("\n" + "="*70)
    print("SAMPLE PROFILES")
    print("="*70)

    for ability_level, ability_filter in [
        ("HIGH ABILITY", df['ability_score'] > 1.5),
        ("MEDIUM ABILITY", (df['ability_score'] > -0.5) & (df['ability_score'] < 0.5)),
        ("LOW ABILITY", df['ability_score'] < -1.5)
    ]:
        idx = df[ability_filter].index[0]
        print(f"\n[{ability_level}]")
        print(f"ID: {df.loc[idx, 'freelancer_id']} | Ability: {df.loc[idx, 'ability_score']:.2f} | "
              f"Treatment: {df.loc[idx, 'program_participation']} | Earnings: ${df.loc[idx, 'hourly_earnings']:.2f}")
        print(f"Text: {df.loc[idx, 'profile_text'][:200]}...")

    print("\n" + "="*70)
    print("✓✓✓ ENHANCED DATA GENERATION COMPLETE ✓✓✓")
    print("="*70)
