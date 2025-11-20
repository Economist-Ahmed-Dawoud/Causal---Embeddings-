"""
Synthetic Data Generation for Causal Inference with Text Embeddings - OPTIMIZED
==================================================================================

This version includes:
- Batch processing with checkpoints
- Resume capability
- Configurable sample size
- Robust error handling

Author: Principal Causal Investigator
Date: 2025-11-20
"""

import numpy as np
import pandas as pd
import requests
import json
from scipy.special import expit
import time
import os
from tqdm import tqdm
import argparse

# Constants
TRUE_CAUSAL_EFFECT = 5.0
GEMINI_API_KEY = "AIzaSyAR9tJzfbD6sjZ7tN1ji5ew7CN_gXavjU8"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"

# Egyptian cities
EGYPTIAN_CITIES = [
    'Cairo', 'Alexandria', 'Giza', 'Shubra El-Kheima', 'Port Said',
    'Suez', 'Luxor', 'Mansoura', 'El-Mahalla El-Kubra', 'Tanta',
    'Asyut', 'Ismailia', 'Fayyum', 'Zagazig', 'Aswan', 'Damietta'
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

# Skill demand mapping
CATEGORY_DEMAND = {
    'Web Development': 0.9, 'Mobile App Development': 0.85, 'UI/UX Design': 0.8,
    'Digital Marketing': 0.75, 'SEO Specialist': 0.7, 'Graphic Design': 0.65,
    'Content Writing': 0.6, 'Translation': 0.55, 'Video Editing': 0.6,
    'Data Entry': 0.4, 'Administrative Support': 0.45, 'Customer Support': 0.5,
    'Social Media Management': 0.65, 'Accounting': 0.55
}


def generate_structured_data(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate the structured variables for the synthetic dataset."""
    np.random.seed(seed)
    print(f"Generating structured variables for {n} freelancers...")

    # 1. LATENT CONFOUNDER (U)
    ability_score = np.random.normal(0, 1, n)

    # 2. DEMOGRAPHICS
    age = np.random.uniform(18, 50, n)
    years_experience = np.maximum(0, (age - 18) * 0.6 + np.random.normal(0, 2, n))
    years_experience = np.minimum(years_experience, age - 18)

    education_numeric = ability_score * 0.3 + np.random.normal(0, 1, n)
    education_level = pd.cut(
        education_numeric,
        bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, 2.0, np.inf],
        labels=EDUCATION_LEVELS
    ).astype(str)

    city = np.random.choice(EGYPTIAN_CITIES, n, replace=True)
    category = np.random.choice(CATEGORIES, n, replace=True)

    # 3. PLATFORM METRICS
    profile_completeness = np.clip(50 + ability_score * 15 + np.random.normal(0, 10, n), 0, 100)
    num_skills = np.maximum(1, np.round(5 + ability_score * 2 + years_experience * 0.3 + np.random.normal(0, 2, n))).astype(int)
    portfolio_items = np.maximum(0, np.round(3 + ability_score * 1.5 + years_experience * 0.4 + np.random.normal(0, 2, n))).astype(int)
    certifications = np.maximum(0, np.round(ability_score * 0.8 + np.random.normal(0, 1, n))).astype(int)
    total_jobs = np.maximum(0, np.round(10 + years_experience * 5 + ability_score * 8 + np.random.normal(0, 10, n))).astype(int)
    success_rate = np.clip(60 + ability_score * 10 + years_experience * 1.5 + np.random.normal(0, 8, n), 0, 100)
    response_rate = np.clip(50 + ability_score * 12 + np.random.normal(0, 15, n), 0, 100)

    # 4. MARKET FACTORS
    market_demand_score = np.array([CATEGORY_DEMAND[cat] + np.random.normal(0, 0.1) for cat in category])
    market_demand_score = np.clip(market_demand_score, 0, 1)

    # 5. TREATMENT (D) - Strong selection bias
    treatment_propensity = expit(
        1.5 * ability_score +
        0.5 * years_experience +
        0.3 * (profile_completeness - 75) / 25 +
        np.random.normal(0, 0.5, n)
    )
    program_participation = (np.random.uniform(0, 1, n) < treatment_propensity).astype(int)

    # 6. OUTCOME (Y) - TRUE CAUSAL EFFECT = $5.00
    category_effect_map = {
        'Web Development': 8, 'Mobile App Development': 10, 'UI/UX Design': 7,
        'Digital Marketing': 5, 'SEO Specialist': 4, 'Graphic Design': 4,
        'Content Writing': 3, 'Translation': 3, 'Video Editing': 5,
        'Data Entry': -2, 'Administrative Support': -1, 'Customer Support': 0,
        'Social Media Management': 4, 'Accounting': 6
    }
    category_effect = np.array([category_effect_map[cat] for cat in category])

    hourly_earnings = (
        10 +
        TRUE_CAUSAL_EFFECT * program_participation +  # *** TRUE EFFECT ***
        8.0 * ability_score +                          # Strong confounding
        0.5 * years_experience +
        category_effect +
        3.0 * market_demand_score +
        0.05 * profile_completeness +
        np.random.normal(0, 2, n)
    )
    hourly_earnings = np.clip(hourly_earnings, 3, 80)

    # 7. CREATE DATAFRAME
    df = pd.DataFrame({
        'freelancer_id': [f'EGY_{i+1:05d}' for i in range(n)],
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
        'hourly_earnings': hourly_earnings
    })

    print(f"✓ Generated {n} observations")
    print(f"  Treatment rate: {program_participation.mean():.1%}")
    print(f"  Naive effect: ${df[df.program_participation==1]['hourly_earnings'].mean() - df[df.program_participation==0]['hourly_earnings'].mean():.2f}")
    print(f"  True effect: ${TRUE_CAUSAL_EFFECT:.2f}")

    return df


def call_gemini_api(prompt: str, max_retries: int = 3) -> str:
    """Call Gemini API with retry logic."""
    for attempt in range(max_retries):
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.9,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 300,
                }
            }

            response = requests.post(GEMINI_API_URL, headers={'Content-Type': 'application/json'},
                                    json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text'].strip()

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  API failed after {max_retries} attempts: {e}")
                return None
            time.sleep(1 * (attempt + 1))  # Exponential backoff

    return None


def generate_profile_text(row: pd.Series) -> str:
    """Generate profile text based on ability score."""
    ability = row['ability_score']
    category = row['category']
    experience = row['years_experience']
    city = row['city']
    num_skills = row['num_skills']
    market_demand = row['market_demand_score']
    education = row['education_level']

    if ability > 1.0:
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

    api_response = call_gemini_api(prompt)

    if api_response:
        return api_response
    else:
        # Fallback
        if ability > 1.0:
            return f"I am a highly experienced {category} professional based in {city} with {experience:.0f} years of expertise. I specialize in delivering exceptional results for my clients through cutting-edge solutions and meticulous attention to detail."
        elif ability < -1.0:
            return f"i am a {category} freelancer from {city}. i have {experience:.0f} years experience. i am hard working and dedicated. looking for good projects."
        else:
            return f"I am a {category} freelancer based in {city}, Egypt with {experience:.0f} years of experience. I provide quality services to clients and am committed to meeting deadlines and delivering good work."


def add_text_profiles(df: pd.DataFrame, checkpoint_file: str = 'checkpoint.parquet') -> pd.DataFrame:
    """Add text profiles with checkpoint support."""
    print("\nGenerating text profiles using Gemini 2.0 Flash...")

    # Check if checkpoint exists
    start_idx = 0
    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint file. Loading...")
        df_checkpoint = pd.read_parquet(checkpoint_file)
        if 'profile_text' in df_checkpoint.columns:
            # Find where we left off
            completed = df_checkpoint['profile_text'].notna().sum()
            print(f"Resuming from row {completed}")
            start_idx = completed
            df = df_checkpoint.copy()

    if 'profile_text' not in df.columns:
        df['profile_text'] = None

    # Generate profiles
    for idx in tqdm(range(start_idx, len(df)), desc="Generating profiles", initial=start_idx, total=len(df)):
        row = df.iloc[idx]
        profile_text = generate_profile_text(row)
        df.at[idx, 'profile_text'] = profile_text

        # Save checkpoint every 100 rows
        if (idx + 1) % 100 == 0:
            df.to_parquet(checkpoint_file, index=False)

        # Rate limiting
        if (idx + 1) % 50 == 0:
            time.sleep(2)

    # Final save
    df.to_parquet(checkpoint_file, index=False)

    print(f"\n✓ Generated {len(df)} profile texts")

    # Show samples
    print("\nSample profiles by ability tier:")
    print("\n[HIGH ABILITY - U > 1]")
    high = df[df.ability_score > 1.5].iloc[0]
    print(f"Ability: {high.ability_score:.2f}")
    print(f"{high.profile_text[:200]}...\n")

    print("\n[MEDIUM ABILITY]")
    med = df[(df.ability_score > -0.5) & (df.ability_score < 0.5)].iloc[0]
    print(f"Ability: {med.ability_score:.2f}")
    print(f"{med.profile_text[:200]}...\n")

    print("\n[LOW ABILITY - U < -1]")
    low = df[df.ability_score < -1.5].iloc[0]
    print(f"Ability: {low.ability_score:.2f}")
    print(f"{low.profile_text[:200]}...\n")

    return df


def main(n_samples: int = 5000):
    """Main execution pipeline."""
    print("=" * 70)
    print("SYNTHETIC DATA GENERATION: Causal Inference with Text Embeddings")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Sample size: {n_samples}")
    print(f"  TRUE CAUSAL EFFECT: ${TRUE_CAUSAL_EFFECT:.2f}")
    print("\n" + "=" * 70 + "\n")

    # Generate structured data
    df = generate_structured_data(n_samples)

    # Add text profiles
    checkpoint_file = f'/home/user/Causal---Embeddings-/checkpoint_{n_samples}.parquet'
    df = add_text_profiles(df, checkpoint_file)

    # Save final dataset
    output_path = '/home/user/Causal---Embeddings-/synthetic_upwork_data.parquet'
    df.to_parquet(output_path, index=False)
    print(f"\n✓ Data saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"\nKey statistics:")
    print(df[['age', 'years_experience', 'ability_score', 'program_participation',
              'hourly_earnings']].describe())

    print(f"\n✓ Phase 2 Complete!")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples to generate')
    args = parser.parse_args()

    df = main(args.n_samples)
