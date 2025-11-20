"""
Synthetic Data Generation with REAL Gemini API Integration
===========================================================

This script generates realistic freelancer profiles using Google's Gemini API.
NO fallback templates - only real AI-generated content!

Setup:
1. Get your free API key from: https://aistudio.google.com/app/apikey
2. Create a .env file in this directory
3. Add: GEMINI_API_KEY=your_actual_key_here
4. Run: python generate_gemini_profiles.py

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
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# Constants
TRUE_CAUSAL_EFFECT = 5.0
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate API key
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
    print("\n" + "="*70)
    print("ERROR: Gemini API key not configured!")
    print("="*70)
    print("\nPlease follow these steps:")
    print("1. Visit: https://aistudio.google.com/app/apikey")
    print("2. Create a new API key (it's FREE)")
    print("3. Create a file named '.env' in this directory")
    print("4. Add this line: GEMINI_API_KEY=your_actual_key_here")
    print("5. Run this script again")
    print("\n" + "="*70 + "\n")
    sys.exit(1)

# Use stable Gemini model
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

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

    # 1. LATENT CONFOUNDER (U) - The unobserved "ability"
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

    # 3. PLATFORM METRICS (all influenced by ability)
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

    # 5. TREATMENT (D) - Strong selection bias based on ability
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
    treated_mean = df[df.program_participation==1]['hourly_earnings'].mean()
    control_mean = df[df.program_participation==0]['hourly_earnings'].mean()
    print(f"  Treatment rate: {program_participation.mean():.1%}")
    print(f"  Mean earnings (Treated): ${treated_mean:.2f}")
    print(f"  Mean earnings (Control): ${control_mean:.2f}")
    print(f"  Naive effect: ${treated_mean - control_mean:.2f}")
    print(f"  True effect: ${TRUE_CAUSAL_EFFECT:.2f}\n")

    return df


def call_gemini_api(prompt: str, max_retries: int = 3) -> str:
    """
    Call Gemini API with retry logic and proper error handling.

    Returns the generated text or None if all attempts fail.
    """
    for attempt in range(max_retries):
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.9,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 350,
                    "stopSequences": []
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }

            response = requests.post(
                GEMINI_API_URL,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=30
            )

            # Check for HTTP errors
            if response.status_code == 403:
                print(f"\n❌ API KEY ERROR: The API key is invalid or has no access.")
                print("Please check your API key at: https://aistudio.google.com/app/apikey")
                sys.exit(1)

            response.raise_for_status()
            result = response.json()

            # Extract text from response
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    return candidate['content']['parts'][0]['text'].strip()

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  ⏱ Timeout, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Request timed out after {max_retries} attempts")

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  ⚠ Request error: {str(e)[:50]}... retrying in {wait_time}s")
                time.sleep(wait_time)
            else:
                print(f"  ❌ API request failed after {max_retries} attempts: {e}")

        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


def generate_profile_text(row: pd.Series, retry_on_failure: bool = True) -> str:
    """
    Generate realistic profile text using Gemini API based on ability score.

    This function creates different prompts for high/medium/low ability freelancers
    to produce natural variation in profile quality.
    """
    ability = row['ability_score']
    category = row['category']
    experience = row['years_experience']
    city = row['city']
    num_skills = row['num_skills']
    market_demand = row['market_demand_score']
    education = row['education_level']

    # HIGH ABILITY PROMPT - Top-tier freelancers
    if ability > 1.0:
        prompt = f"""Write a highly professional, articulate, and persuasive Upwork freelancer profile summary for an expert Egyptian professional.

Category: {category}
Location: {city}, Egypt
Experience: {experience:.1f} years
Education: {education}
Number of skills: {num_skills}

Requirements:
- Use perfect grammar and sophisticated vocabulary
- Highlight expertise and unique value proposition
- Mention specific technical skills and relevant tools
- Include client-focused language (ROI, results, quality, impact)
- {"Emphasize high-demand skills and cutting-edge expertise" if market_demand > 0.7 else "Focus on proven track record and reliability"}
- Professional tone, confident but not arrogant
- 100-150 words
- Sound like a top 1% freelancer on Upwork
- Write in FIRST PERSON ("I am...", "I specialize...", etc.)
- DO NOT use placeholder names or brackets
- Be specific and concrete

Write only the profile summary, nothing else."""

    # LOW ABILITY PROMPT - Entry-level freelancers
    elif ability < -1.0:
        prompt = f"""Write a short, basic Upwork freelancer profile for an entry-level Egyptian professional who is still developing their skills.

Category: {category}
Location: {city}, Egypt
Experience: {experience:.1f} years

Requirements:
- Simple language, generic phrasing
- Include 1-2 minor grammatical imperfections (lowercase "i", missing punctuation, casual phrasing)
- Brief and to the point (60-90 words)
- Less polished, more straightforward
- Use common clichés like "hard working", "dedicated", "passionate"
- Sound like a beginner trying their best
- Write in FIRST PERSON ("I am...", "i have...", etc.)
- DO NOT use placeholder names or brackets
- Keep it authentic and simple

Write only the profile summary, nothing else."""

    # MEDIUM ABILITY PROMPT - Average competent freelancers
    else:
        prompt = f"""Write a standard, competent Upwork freelancer profile summary for an Egyptian professional.

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
- Write in FIRST PERSON ("I am...", "I offer...", etc.)
- DO NOT use placeholder names or brackets
- Be straightforward and honest

Write only the profile summary, nothing else."""

    # Call Gemini API
    api_response = call_gemini_api(prompt)

    if api_response:
        return api_response
    else:
        # If API fails, we should stop rather than use fallback
        if retry_on_failure:
            print(f"  ❌ Failed to generate profile for {row['freelancer_id']}")
            raise Exception("API call failed - cannot generate realistic profiles without Gemini")
        return None


def add_text_profiles(df: pd.DataFrame, checkpoint_file: str = 'checkpoint.parquet',
                      batch_size: int = 50, save_frequency: int = 100) -> pd.DataFrame:
    """
    Add text profiles with checkpoint support and progress tracking.

    Parameters:
    - df: DataFrame with structured data
    - checkpoint_file: Path to save progress
    - batch_size: Number of requests before rate limiting pause
    - save_frequency: Save checkpoint every N profiles
    """
    print("\n" + "="*70)
    print("GENERATING REALISTIC PROFILES USING GEMINI API")
    print("="*70)
    print(f"Model: Gemini 1.5 Flash")
    print(f"Total profiles to generate: {len(df)}")
    print(f"Checkpoint file: {checkpoint_file}")
    print("="*70 + "\n")

    # Check if checkpoint exists
    start_idx = 0
    if os.path.exists(checkpoint_file):
        print(f"📂 Found checkpoint file. Loading...")
        df_checkpoint = pd.read_parquet(checkpoint_file)
        if 'profile_text' in df_checkpoint.columns:
            # Find where we left off
            completed = df_checkpoint['profile_text'].notna().sum()
            if completed > 0:
                print(f"✓ Resuming from row {completed}\n")
                start_idx = completed
                df = df_checkpoint.copy()

    if 'profile_text' not in df.columns:
        df['profile_text'] = None

    # Generate profiles with progress bar
    failed_count = 0
    success_count = 0

    print("🚀 Starting profile generation...\n")

    for idx in tqdm(range(start_idx, len(df)), desc="Generating profiles",
                    initial=start_idx, total=len(df), ncols=80):
        try:
            row = df.iloc[idx]
            profile_text = generate_profile_text(row)

            if profile_text:
                df.at[idx, 'profile_text'] = profile_text
                success_count += 1
            else:
                failed_count += 1
                print(f"\n⚠ Warning: Failed to generate profile for row {idx}")

            # Save checkpoint periodically
            if (idx + 1) % save_frequency == 0:
                df.to_parquet(checkpoint_file, index=False)
                print(f"\n💾 Checkpoint saved at row {idx + 1}")

            # Rate limiting - pause every batch_size requests
            if (idx + 1) % batch_size == 0 and idx < len(df) - 1:
                print(f"\n⏸ Rate limit pause (2s)...")
                time.sleep(2)

        except KeyboardInterrupt:
            print("\n\n⚠ Generation interrupted by user")
            print(f"💾 Saving progress to checkpoint...")
            df.to_parquet(checkpoint_file, index=False)
            print(f"✓ Progress saved. Run again to resume from row {idx}")
            sys.exit(0)

        except Exception as e:
            print(f"\n❌ Error generating profile for row {idx}: {e}")
            failed_count += 1
            # Save checkpoint on error
            df.to_parquet(checkpoint_file, index=False)
            raise

    # Final save
    df.to_parquet(checkpoint_file, index=False)

    print(f"\n\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"✓ Successfully generated: {success_count} profiles")
    if failed_count > 0:
        print(f"⚠ Failed: {failed_count} profiles")
    print("="*70 + "\n")

    # Show sample profiles by ability tier
    print("\n📝 SAMPLE PROFILES BY ABILITY TIER:\n")

    print("-" * 70)
    print("[HIGH ABILITY - Ability Score > 1.0]")
    print("-" * 70)
    high_df = df[df.ability_score > 1.5]
    if len(high_df) > 0:
        high = high_df.iloc[0]
        print(f"Freelancer ID: {high.freelancer_id}")
        print(f"Ability Score: {high.ability_score:.2f}")
        print(f"Category: {high.category}")
        print(f"Experience: {high.years_experience:.1f} years")
        print(f"\nProfile Text:\n{high.profile_text}\n")

    print("-" * 70)
    print("[MEDIUM ABILITY - Ability Score between -1.0 and 1.0]")
    print("-" * 70)
    med_df = df[(df.ability_score > -0.5) & (df.ability_score < 0.5)]
    if len(med_df) > 0:
        med = med_df.iloc[0]
        print(f"Freelancer ID: {med.freelancer_id}")
        print(f"Ability Score: {med.ability_score:.2f}")
        print(f"Category: {med.category}")
        print(f"Experience: {med.years_experience:.1f} years")
        print(f"\nProfile Text:\n{med.profile_text}\n")

    print("-" * 70)
    print("[LOW ABILITY - Ability Score < -1.0]")
    print("-" * 70)
    low_df = df[df.ability_score < -1.5]
    if len(low_df) > 0:
        low = low_df.iloc[0]
        print(f"Freelancer ID: {low.freelancer_id}")
        print(f"Ability Score: {low.ability_score:.2f}")
        print(f"Category: {low.category}")
        print(f"Experience: {low.years_experience:.1f} years")
        print(f"\nProfile Text:\n{low.profile_text}\n")

    print("="*70 + "\n")

    return df


def main(n_samples: int = 5000, resume: bool = True):
    """
    Main execution pipeline.

    Parameters:
    - n_samples: Number of freelancer profiles to generate
    - resume: Whether to resume from checkpoint if available
    """
    print("\n" + "="*70)
    print("SYNTHETIC DATA GENERATION: Causal Inference with Text Embeddings")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Sample size: {n_samples}")
    print(f"  TRUE CAUSAL EFFECT: ${TRUE_CAUSAL_EFFECT:.2f}")
    print(f"  Resume from checkpoint: {resume}")
    print(f"  Gemini Model: 1.5 Flash")
    print("\n" + "="*70 + "\n")

    checkpoint_file = f'/home/user/Causal---Embeddings-/checkpoint_{n_samples}.parquet'

    # If resuming and checkpoint exists, load it
    if resume and os.path.exists(checkpoint_file):
        print("📂 Loading existing checkpoint...")
        df = pd.read_parquet(checkpoint_file)
        print(f"✓ Loaded {len(df)} rows from checkpoint\n")
    else:
        # Generate structured data from scratch
        df = generate_structured_data(n_samples)

    # Add text profiles using Gemini API
    df = add_text_profiles(df, checkpoint_file)

    # Save final dataset
    output_path = '/home/user/Causal---Embeddings-/synthetic_upwork_data.parquet'
    df.to_parquet(output_path, index=False)
    print(f"💾 Final dataset saved to: {output_path}\n")

    # Print summary statistics
    print("="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Total observations: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nTreatment distribution:")
    print(f"  Treated: {df.program_participation.sum()} ({df.program_participation.mean():.1%})")
    print(f"  Control: {(~df.program_participation.astype(bool)).sum()} ({(1-df.program_participation.mean()):.1%})")
    print(f"\nEarnings by treatment status:")
    treated_earnings = df[df.program_participation==1]['hourly_earnings'].mean()
    control_earnings = df[df.program_participation==0]['hourly_earnings'].mean()
    print(f"  Treated: ${treated_earnings:.2f}/hour")
    print(f"  Control: ${control_earnings:.2f}/hour")
    print(f"  Naive effect: ${treated_earnings - control_earnings:.2f}")
    print(f"  TRUE effect: ${TRUE_CAUSAL_EFFECT:.2f}")
    print(f"  Bias: ${(treated_earnings - control_earnings) - TRUE_CAUSAL_EFFECT:.2f}")

    print(f"\nAbility score distribution:")
    print(f"  Mean: {df.ability_score.mean():.2f}")
    print(f"  Std: {df.ability_score.std():.2f}")
    print(f"  Min: {df.ability_score.min():.2f}")
    print(f"  Max: {df.ability_score.max():.2f}")

    print("\n" + "="*70)
    print("✓ ALL PHASES COMPLETE!")
    print("="*70 + "\n")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic freelancer data with realistic AI-generated profiles"
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=5000,
        help='Number of samples to generate (default: 5000)'
    )
    parser.add_argument(
        '--no_resume',
        action='store_true',
        help='Start from scratch, ignore checkpoint'
    )

    args = parser.parse_args()

    # Confirm API key is set
    print("\n" + "="*70)
    print("API KEY STATUS")
    print("="*70)
    if GEMINI_API_KEY:
        print(f"✓ API key found: {GEMINI_API_KEY[:8]}...{GEMINI_API_KEY[-4:]}")
    print("="*70 + "\n")

    df = main(args.n_samples, resume=not args.no_resume)
