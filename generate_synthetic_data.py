"""
Synthetic Data Generation for Causal Inference with Text Embeddings
=====================================================================

This script generates a realistic dataset of 5,000 Egyptian freelancers where:
- TRUE CAUSAL EFFECT: $5.00 (the effect of program participation on hourly earnings)
- U (ability/savviness) is a latent confounder observable only via text
- Text profiles are generated using Google Gemini API to reflect ability levels

Author: Principal Causal Investigator
Date: 2025-11-20
"""

import numpy as np
import pandas as pd
import requests
import json
from scipy.special import expit  # sigmoid function
from typing import Dict, List
import time
import os
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Constants
N_SAMPLES = 5000
TRUE_CAUSAL_EFFECT = 5.0  # The TRUE effect of program participation on hourly earnings
GEMINI_API_KEY = "AIzaSyAR9tJzfbD6sjZ7tN1ji5ew7CN_gXavjU8"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"

# Egyptian cities with freelancer concentrations
EGYPTIAN_CITIES = [
    'Cairo', 'Alexandria', 'Giza', 'Shubra El-Kheima', 'Port Said',
    'Suez', 'Luxor', 'Mansoura', 'El-Mahalla El-Kubra', 'Tanta',
    'Asyut', 'Ismailia', 'Fayyum', 'Zagazig', 'Aswan', 'Damietta',
    'Damanhur', 'Minya', 'Beni Suef', 'Qena'
]

# Freelancer categories
CATEGORIES = [
    'Web Development', 'Graphic Design', 'Translation',
    'Administrative Support', 'Mobile App Development', 'Content Writing',
    'Digital Marketing', 'Video Editing', 'Data Entry', 'SEO Specialist',
    'UI/UX Design', 'Social Media Management', 'Accounting', 'Customer Support'
]

# Education levels
EDUCATION_LEVELS = [
    'High School', 'Some College', 'Bachelor\'s Degree',
    'Master\'s Degree', 'Professional Certificate', 'Self-Taught'
]

# Skill demand mapping (for market_demand_score)
CATEGORY_DEMAND = {
    'Web Development': 0.9, 'Mobile App Development': 0.85, 'UI/UX Design': 0.8,
    'Digital Marketing': 0.75, 'SEO Specialist': 0.7, 'Graphic Design': 0.65,
    'Content Writing': 0.6, 'Translation': 0.55, 'Video Editing': 0.6,
    'Data Entry': 0.4, 'Administrative Support': 0.45, 'Customer Support': 0.5,
    'Social Media Management': 0.65, 'Accounting': 0.55
}


def generate_structured_data(n: int) -> pd.DataFrame:
    """
    Generate the structured variables for the synthetic dataset.

    Returns a DataFrame with all covariates, latent confounder, treatment, and outcome.
    """
    print("Generating structured variables...")

    # =============================
    # 1. LATENT CONFOUNDER (U)
    # =============================
    # This represents "Freelancer Savviness/Ability" - UNOBSERVED in practice
    # but OBSERVABLE in the text profile
    ability_score = np.random.normal(0, 1, n)  # U ~ N(0, 1)

    # =============================
    # 2. BASELINE DEMOGRAPHICS
    # =============================
    age = np.random.uniform(18, 50, n)

    # Experience correlated with age (but with noise)
    years_experience = np.maximum(
        0,
        (age - 18) * 0.6 + np.random.normal(0, 2, n)
    )
    years_experience = np.minimum(years_experience, age - 18)  # Can't exceed working years

    # Education (slightly correlated with ability)
    education_numeric = ability_score * 0.3 + np.random.normal(0, 1, n)
    education_level = pd.cut(
        education_numeric,
        bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, 2.0, np.inf],
        labels=EDUCATION_LEVELS
    ).astype(str)

    # Location (random)
    city = np.random.choice(EGYPTIAN_CITIES, n, replace=True)

    # Category (random, but we'll use it for heterogeneity)
    category = np.random.choice(CATEGORIES, n, replace=True)

    # =============================
    # 3. PLATFORM METRICS
    # =============================
    # Profile completeness (correlated with ability)
    profile_completeness = np.clip(
        50 + ability_score * 15 + np.random.normal(0, 10, n),
        0, 100
    )

    # Number of skills listed (correlated with ability and experience)
    num_skills = np.maximum(
        1,
        np.round(5 + ability_score * 2 + years_experience * 0.3 + np.random.normal(0, 2, n))
    ).astype(int)

    # Portfolio items (correlated with ability and experience)
    portfolio_items = np.maximum(
        0,
        np.round(3 + ability_score * 1.5 + years_experience * 0.4 + np.random.normal(0, 2, n))
    ).astype(int)

    # Certifications (correlated with ability)
    certifications = np.maximum(
        0,
        np.round(ability_score * 0.8 + np.random.normal(0, 1, n))
    ).astype(int)

    # Total jobs completed (function of experience and ability)
    total_jobs = np.maximum(
        0,
        np.round(10 + years_experience * 5 + ability_score * 8 + np.random.normal(0, 10, n))
    ).astype(int)

    # Job success rate (strongly correlated with ability)
    success_rate = np.clip(
        60 + ability_score * 10 + years_experience * 1.5 + np.random.normal(0, 8, n),
        0, 100
    )

    # Response rate (correlated with professionalism/ability)
    response_rate = np.clip(
        50 + ability_score * 12 + np.random.normal(0, 15, n),
        0, 100
    )

    # =============================
    # 4. CLIENT-SIDE / MARKET FACTORS
    # =============================
    # Market demand score (category-based + some noise)
    market_demand_score = np.array([
        CATEGORY_DEMAND[cat] + np.random.normal(0, 0.1)
        for cat in category
    ])
    market_demand_score = np.clip(market_demand_score, 0, 1)

    # =============================
    # 5. TREATMENT (D) - Program Participation
    # =============================
    # The government program selects for motivated, savvy, experienced freelancers
    # This creates STRONG SELECTION BIAS
    # P(D=1) = Sigmoid(1.5*U + 0.5*experience + noise)

    treatment_propensity = expit(
        1.5 * ability_score +           # High ability → more likely to join
        0.5 * years_experience +        # Experience helps
        0.3 * (profile_completeness - 75) / 25 +  # Completeness matters
        np.random.normal(0, 0.5, n)     # Random noise
    )

    # Binary treatment assignment
    program_participation = (np.random.uniform(0, 1, n) < treatment_propensity).astype(int)

    # =============================
    # 6. OUTCOME (Y) - Hourly Earnings
    # =============================
    # TRUE CAUSAL EFFECT = $5.00
    # Y = 10 + 5.0*D + 8.0*U + 0.5*experience + category_effect + market_effect + noise

    # Category-based wage effect (some categories pay more)
    category_effect_map = {
        'Web Development': 8, 'Mobile App Development': 10, 'UI/UX Design': 7,
        'Digital Marketing': 5, 'SEO Specialist': 4, 'Graphic Design': 4,
        'Content Writing': 3, 'Translation': 3, 'Video Editing': 5,
        'Data Entry': -2, 'Administrative Support': -1, 'Customer Support': 0,
        'Social Media Management': 4, 'Accounting': 6
    }
    category_effect = np.array([category_effect_map[cat] for cat in category])

    # Hourly earnings equation
    hourly_earnings = (
        10 +                                    # Baseline
        TRUE_CAUSAL_EFFECT * program_participation +  # *** TRUE EFFECT = $5.00 ***
        8.0 * ability_score +                   # STRONG confounding via ability
        0.5 * years_experience +                # Experience matters
        category_effect +                       # Category differentials
        3.0 * market_demand_score +            # Market demand
        0.05 * profile_completeness +          # Profile quality
        np.random.normal(0, 2, n)              # Idiosyncratic noise
    )

    # Ensure non-negative earnings and realistic range
    hourly_earnings = np.clip(hourly_earnings, 3, 80)

    # =============================
    # 7. CREATE DATAFRAME
    # =============================
    df = pd.DataFrame({
        # ID
        'freelancer_id': [f'EGY_{i+1:05d}' for i in range(n)],

        # LATENT CONFOUNDER (unobserved in practice)
        'ability_score': ability_score,

        # Demographics
        'age': age,
        'years_experience': years_experience,
        'education_level': education_level,
        'city': city,
        'category': category,

        # Platform metrics
        'profile_completeness': profile_completeness,
        'num_skills': num_skills,
        'portfolio_items': portfolio_items,
        'certifications': certifications,
        'total_jobs': total_jobs,
        'success_rate': success_rate,
        'response_rate': response_rate,

        # Market factors
        'market_demand_score': market_demand_score,

        # Treatment and outcome
        'treatment_propensity': treatment_propensity,
        'program_participation': program_participation,
        'hourly_earnings': hourly_earnings
    })

    print(f"✓ Generated {n} observations")
    print(f"  Treatment rate: {program_participation.mean():.1%}")
    print(f"  Mean earnings (Treated): ${df[df.program_participation==1]['hourly_earnings'].mean():.2f}")
    print(f"  Mean earnings (Control): ${df[df.program_participation==0]['hourly_earnings'].mean():.2f}")
    print(f"  Naive difference: ${df[df.program_participation==1]['hourly_earnings'].mean() - df[df.program_participation==0]['hourly_earnings'].mean():.2f}")
    print(f"  (True effect should be ${TRUE_CAUSAL_EFFECT:.2f})")

    return df


def call_gemini_api(prompt: str) -> str:
    """
    Call Gemini API via HTTP request.
    """
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.9,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 300,
        }
    }

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        if 'candidates' in result and len(result['candidates']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            return None

    except Exception as e:
        print(f"API Error: {e}")
        return None


def generate_profile_text_gemini(row: pd.Series) -> str:
    """
    Generate a realistic Upwork profile summary using Gemini API.
    The text quality MUST reflect the ability_score (U).

    Quality tiers:
    - High ability (U > 1): Professional, articulate, persuasive
    - Low ability (U < -1): Generic, unprofessional, minor errors
    - Medium ability: Standard average profile
    """
    ability = row['ability_score']
    category = row['category']
    experience = row['years_experience']
    city = row['city']
    num_skills = row['num_skills']
    market_demand = row['market_demand_score']
    education = row['education_level']

    # Determine prompt based on ability tier
    if ability > 1.0:
        # HIGH ABILITY: Exceptional profile
        prompt = f"""Write a highly professional, articulate, and persuasive Upwork profile summary for an expert Egyptian freelancer.

Category: {category}
Location: {city}, Egypt
Experience: {experience:.1f} years
Education: {education}
Number of skills: {num_skills}

Requirements:
- Use perfect grammar and sophisticated vocabulary
- Highlight expertise and unique value proposition
- Mention specific technical skills and tools
- Include client-focused language (ROI, results, quality)
- {"Emphasize high-demand skills and market trends" if market_demand > 0.7 else "Focus on reliability and expertise"}
- Professional tone, confident but not arrogant
- 100-150 words
- Sound like a top 1% freelancer

Do not use placeholder names or specific company names. Write in first person."""

    elif ability < -1.0:
        # LOW ABILITY: Basic, generic profile with minor issues
        prompt = f"""Write a short, basic Upwork profile for an entry-level Egyptian freelancer.

Category: {category}
Location: {city}, Egypt
Experience: {experience:.1f} years

Requirements:
- Simple language, generic phrasing
- Include one or two minor grammatical imperfections or casual language
- Brief and to the point (60-80 words)
- Less polished, more straightforward
- May use some lowercase where uppercase is expected
- Mention "hard working" or "dedicated" (common clichés)
- Sound like a beginner trying their best

Do not use placeholder names. Write in first person."""

    else:
        # MEDIUM ABILITY: Standard, competent profile
        prompt = f"""Write a standard, competent Upwork profile summary for an Egyptian freelancer.

Category: {category}
Location: {city}, Egypt
Experience: {experience:.1f} years
Education: {education}

Requirements:
- Clear and professional language
- Mention relevant skills and experience
- Standard grammar (no major errors)
- 80-120 words
- Sound like an average, reliable freelancer
- Professional but not exceptional

Do not use placeholder names. Write in first person."""

    # Call API
    api_response = call_gemini_api(prompt)

    if api_response:
        return api_response
    else:
        # Fallback if API fails
        if ability > 1.0:
            return f"I am a highly experienced {category} professional based in {city} with {experience:.0f} years of expertise. I specialize in delivering exceptional results for my clients through cutting-edge solutions and meticulous attention to detail."
        elif ability < -1.0:
            return f"i am a {category} freelancer from {city}. i have {experience:.0f} years experience. i am hard working and dedicated. looking for good projects."
        else:
            return f"I am a {category} freelancer based in {city}, Egypt with {experience:.0f} years of experience. I provide quality services to clients and am committed to meeting deadlines and delivering good work."


def generate_text_profiles(df: pd.DataFrame, batch_size: int = 50) -> pd.DataFrame:
    """
    Generate text profiles for all freelancers using Gemini API.
    Processes in batches to manage API rate limits.
    """
    print("\nGenerating text profiles using Gemini 2.0 Flash...")

    profiles = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating profiles"):
        profile_text = generate_profile_text_gemini(row)
        profiles.append(profile_text)

        # Rate limiting: pause every batch_size requests
        if (idx + 1) % batch_size == 0:
            time.sleep(2)  # 2 second pause every batch

    df['profile_text'] = profiles

    print(f"\n✓ Generated {len(profiles)} profile texts")
    print("\nSample profiles by ability tier:")
    print("\n[HIGH ABILITY - U > 1]")
    high_ability = df[df.ability_score > 1.5].iloc[0]
    print(f"Ability score: {high_ability.ability_score:.2f}")
    print(f"{high_ability.profile_text}\n")

    print("\n[MEDIUM ABILITY - -1 < U < 1]")
    med_ability = df[(df.ability_score > -0.5) & (df.ability_score < 0.5)].iloc[0]
    print(f"Ability score: {med_ability.ability_score:.2f}")
    print(f"{med_ability.profile_text}\n")

    print("\n[LOW ABILITY - U < -1]")
    low_ability = df[df.ability_score < -1.5].iloc[0]
    print(f"Ability score: {low_ability.ability_score:.2f}")
    print(f"{low_ability.profile_text}\n")

    return df


def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("SYNTHETIC DATA GENERATION: Causal Inference with Text Embeddings")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Sample size: {N_SAMPLES}")
    print(f"  TRUE CAUSAL EFFECT: ${TRUE_CAUSAL_EFFECT:.2f}")
    print(f"  Latent confounder: ability_score (U)")
    print(f"  Treatment: program_participation (D)")
    print(f"  Outcome: hourly_earnings (Y)")
    print("\n" + "=" * 70 + "\n")

    # Step 1: Generate structured data
    df = generate_structured_data(N_SAMPLES)

    # Step 2: Generate text profiles
    df = generate_text_profiles(df)

    # Step 3: Save to parquet
    output_path = '/home/user/Causal---Embeddings-/synthetic_upwork_data.parquet'
    df.to_parquet(output_path, index=False)
    print(f"\n✓ Data saved to: {output_path}")

    # Step 4: Summary statistics
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nNumerical Summary:")
    print(df[['age', 'years_experience', 'ability_score', 'program_participation',
              'hourly_earnings', 'profile_completeness', 'success_rate']].describe())

    print(f"\n✓ Phase 2 Complete!")
    print(f"  Next step: Generate embeddings from profile_text (Phase 3)")

    return df


if __name__ == "__main__":
    df = main()
