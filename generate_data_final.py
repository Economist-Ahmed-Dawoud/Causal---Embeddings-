"""
Final Data Generation: 5,000 Egyptian Freelancers with Ability-Aware Text
===========================================================================

Since the Gemini API is unavailable, we use sophisticated rule-based text generation
that still captures ability differences in vocabulary, grammar, and professionalism.

This is sufficient for demonstrating the methodology.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
import random

# Set random seed
np.random.seed(42)
random.seed(42)

# Constants
N_SAMPLES = 5000
TRUE_CAUSAL_EFFECT = 5.0

# Data from Phase 2 script
EGYPTIAN_CITIES = [
    'Cairo', 'Alexandria', 'Giza', 'Shubra El-Kheima', 'Port Said',
    'Suez', 'Luxor', 'Mansoura', 'El-Mahalla El-Kubra', 'Tanta',
    'Asyut', 'Ismailia', 'Fayyum', 'Zagazig', 'Aswan', 'Damietta'
]

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

CATEGORY_DEMAND = {
    'Web Development': 0.9, 'Mobile App Development': 0.85, 'UI/UX Design': 0.8,
    'Digital Marketing': 0.75, 'SEO Specialist': 0.7, 'Graphic Design': 0.65,
    'Content Writing': 0.6, 'Translation': 0.55, 'Video Editing': 0.6,
    'Data Entry': 0.4, 'Administrative Support': 0.45, 'Customer Support': 0.5,
    'Social Media Management': 0.65, 'Accounting': 0.55
}

# Enhanced text templates with STRONG ability differentiation
# These templates create clear linguistic signals that embeddings can capture

HIGH_ABILITY_TEMPLATES = [
    "As a distinguished {category} specialist with {exp:.0f} years of expertise, I architect sophisticated solutions delivering quantifiable ROI exceeding client expectations. Based in {city}, Egypt, I synthesize cutting-edge methodologies with strategic foresight, leveraging advanced analytical frameworks and industry best practices. My portfolio exemplifies consistent excellence in {skill_area}, characterized by meticulous attention to detail, innovative problem-solving, and transformative client partnerships.",
    "Accomplished {category} strategist from {city} with {exp:.0f} years orchestrating high-impact initiatives across diverse sectors. I excel at synthesizing complex requirements into elegant, scalable architectures that drive sustainable growth. My consultative approach integrates technical mastery with executive-level communication, ensuring stakeholder alignment and exceptional deliverables. Specializing in {skill_area}, I consistently exceed benchmarks through rigorous methodology and creative innovation.",
    "As a premier {category} consultant with {exp:.0f} years of progressive achievement, I deliver transformative solutions that catalyze organizational success. Operating from {city}, my methodology emphasizes empirical analysis, strategic optimization, and measurable outcomes. My expertise in {skill_area} has empowered Fortune-caliber clients to realize their most ambitious objectives through systematic excellence and unwavering commitment to quality.",
    "Distinguished {category} professional commanding {exp:.0f} years of expertise in delivering enterprise-grade solutions. From {city}, I leverage comprehensive domain knowledge and advanced technical proficiencies to architect innovative systems. My track record demonstrates consistent success in {skill_area}, marked by strategic thinking, exceptional communication, and a relentless pursuit of excellence that transforms challenges into competitive advantages.",
    "Elite {category} expert with {exp:.0f} years mastering the intersection of technical excellence and business acumen. Based in {city}, Egypt, I specialize in engineering sophisticated solutions that align with strategic imperatives. My proficiency in {skill_area} enables me to deliver comprehensive, scalable outcomes that exceed expectations while maintaining the highest standards of professional integrity and client partnership.",
]

MED_ABILITY_TEMPLATES = [
    "I am a {category} freelancer based in {city}, Egypt with {exp:.0f} years of experience. I provide professional services and work with clients to meet their project requirements. My skills include {skill_area} and I deliver quality work on time.",
    "Professional {category} specialist from {city} with {exp:.0f} years of hands-on experience. I focus on reliable service delivery and good client communication throughout the project. I have worked on various projects involving {skill_area}.",
    "Experienced {category} freelancer in {city}, Egypt. With {exp:.0f} years in this field, I offer dependable services in {skill_area}. I value client satisfaction and complete projects efficiently.",
    "{category} professional based in {city} with {exp:.0f} years of practical experience. I handle projects in {skill_area} and maintain consistent quality standards. I communicate regularly with clients and meet deadlines.",
    "Dedicated {category} practitioner from {city}, Egypt. {exp:.0f} years of experience in the industry. I work on {skill_area} projects and ensure timely delivery with professional results.",
]

LOW_ABILITY_TEMPLATES = [
    "hello i am {category} from {city} with {exp:.0f} years. i do good work and deliver on time. looking for work opportunities. i know {skill_area}. contact me.",
    "i am {category} in {city} egypt. {exp:.0f} years experiance. i am hardworking and dedicated. i do {skill_area} stuff. avalable for projects. thanks.",
    "hi im a {category} based {city}. {exp:.0f} yr experience. can do {skill_area} work. i try hard and finish fast. message me for jobs.",
    "{category} from {city}. got {exp:.0f} years doing this. i work hard everyday. {skill_area} is what i do. need projects plz contact.",
    "helo i am {category} freelancer {city} egypt {exp:.0f} years exp. i do {skill_area}. i am reliable worker. looking for good clients to work with. ty.",
]

# Skill areas by category
SKILL_AREAS = {
    'Web Development': ['full-stack development', 'responsive design', 'API integration', 'database optimization'],
    'Graphic Design': ['brand identity', 'visual storytelling', 'Adobe Creative Suite', 'print and digital media'],
    'Translation': ['technical translation', 'localization', 'cultural adaptation', 'proofreading'],
    'Administrative Support': ['calendar management', 'data organization', 'correspondence handling', 'scheduling'],
    'Mobile App Development': ['iOS and Android development', 'cross-platform solutions', 'app optimization', 'UX implementation'],
    'Content Writing': ['SEO copywriting', 'blog content', 'technical writing', 'storytelling'],
    'Digital Marketing': ['campaign strategy', 'social media marketing', 'analytics', 'conversion optimization'],
    'Video Editing': ['post-production', 'color grading', 'motion graphics', 'sound design'],
    'Data Entry': ['data processing', 'spreadsheet management', 'accuracy verification', 'database updates'],
    'SEO Specialist': ['keyword research', 'on-page optimization', 'link building', 'technical SEO'],
    'UI/UX Design': ['user research', 'wireframing', 'prototyping', 'usability testing'],
    'Social Media Management': ['content scheduling', 'community engagement', 'analytics reporting', 'brand voice'],
    'Accounting': ['bookkeeping', 'financial reporting', 'tax preparation', 'reconciliation'],
    'Customer Support': ['client communication', 'issue resolution', 'CRM management', 'satisfaction tracking'],
}


def generate_profile_text_fast(ability, category, experience, city):
    """Generate ability-aware profile text without API."""
    skill_area = random.choice(SKILL_AREAS.get(category, ['professional services']))

    if ability > 1.0:
        template = random.choice(HIGH_ABILITY_TEMPLATES)
    elif ability < -1.0:
        template = random.choice(LOW_ABILITY_TEMPLATES)
    else:
        template = random.choice(MED_ABILITY_TEMPLATES)

    return template.format(
        category=category,
        exp=experience,
        city=city,
        skill_area=skill_area
    )


print("="*70)
print("GENERATING SYNTHETIC UPWORK DATASET: 5,000 FREELANCERS")
print("="*70)
print(f"\nTrue Causal Effect: ${TRUE_CAUSAL_EFFECT:.2f}")
print("Generating data...")

# 1. LATENT CONFOUNDER
ability_score = np.random.normal(0, 1, N_SAMPLES)

# 2. DEMOGRAPHICS
age = np.random.uniform(18, 50, N_SAMPLES)
years_experience = np.maximum(0, (age - 18) * 0.6 + np.random.normal(0, 2, N_SAMPLES))
years_experience = np.minimum(years_experience, age - 18)

education_numeric = ability_score * 0.3 + np.random.normal(0, 1, N_SAMPLES)
education_level = pd.cut(
    education_numeric,
    bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, 2.0, np.inf],
    labels=EDUCATION_LEVELS
).astype(str)

city = np.random.choice(EGYPTIAN_CITIES, N_SAMPLES, replace=True)
category = np.random.choice(CATEGORIES, N_SAMPLES, replace=True)

# 3. PLATFORM METRICS
profile_completeness = np.clip(50 + ability_score * 15 + np.random.normal(0, 10, N_SAMPLES), 0, 100)
num_skills = np.maximum(1, np.round(5 + ability_score * 2 + years_experience * 0.3 + np.random.normal(0, 2, N_SAMPLES))).astype(int)
portfolio_items = np.maximum(0, np.round(3 + ability_score * 1.5 + years_experience * 0.4 + np.random.normal(0, 2, N_SAMPLES))).astype(int)
certifications = np.maximum(0, np.round(ability_score * 0.8 + np.random.normal(0, 1, N_SAMPLES))).astype(int)
total_jobs = np.maximum(0, np.round(10 + years_experience * 5 + ability_score * 8 + np.random.normal(0, 10, N_SAMPLES))).astype(int)
success_rate = np.clip(60 + ability_score * 10 + years_experience * 1.5 + np.random.normal(0, 8, N_SAMPLES), 0, 100)
response_rate = np.clip(50 + ability_score * 12 + np.random.normal(0, 15, N_SAMPLES), 0, 100)

# 4. MARKET FACTORS
market_demand_score = np.array([CATEGORY_DEMAND[cat] + np.random.normal(0, 0.1) for cat in category])
market_demand_score = np.clip(market_demand_score, 0, 1)

# 5. TREATMENT - Strong selection bias with PURE U → D confounding
# Key design choices:
#   - Ability coefficient (2.0) dominates the propensity score
#   - Experience is scaled by /10 to prevent it from dominating
#   - Removed profile_completeness term (it's a collider, not a confounder)
#   - Reduced noise variance for cleaner confounding signal
treatment_propensity = expit(
    2.0 * ability_score +                    # STRONG: Latent ability drives treatment selection
    0.3 * (years_experience / 10) +          # WEAK: Scaled experience effect (prevents domination)
    np.random.normal(0, 0.3, N_SAMPLES)      # REDUCED noise for cleaner signal
)
# Ensure positivity assumption holds (no perfect prediction)
treatment_propensity = np.clip(treatment_propensity, 0.05, 0.95)
program_participation = (np.random.uniform(0, 1, N_SAMPLES) < treatment_propensity).astype(int)

# 6. OUTCOME - TRUE EFFECT = $5.00
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
    TRUE_CAUSAL_EFFECT * program_participation +
    8.0 * ability_score +
    0.5 * years_experience +
    category_effect +
    3.0 * market_demand_score +
    0.05 * profile_completeness +
    np.random.normal(0, 2, N_SAMPLES)
)
hourly_earnings = np.clip(hourly_earnings, 3, 80)

# 7. GENERATE TEXT PROFILES
print("Generating text profiles...")
profile_texts = [
    generate_profile_text_fast(ability_score[i], category[i], years_experience[i], city[i])
    for i in range(N_SAMPLES)
]

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

# 9. SAVE
output_path = '/home/user/Causal---Embeddings-/synthetic_upwork_data.parquet'
df.to_parquet(output_path, index=False)

# 10. SUMMARY
print(f"\n✓ Generated {len(df)} observations")
print(f"✓ Data saved to: {output_path}")
print(f"\nKey Statistics:")
print(f"  Treatment rate: {program_participation.mean():.1%}")
print(f"  Mean earnings (Treated): ${df[df.program_participation==1]['hourly_earnings'].mean():.2f}")
print(f"  Mean earnings (Control): ${df[df.program_participation==0]['hourly_earnings'].mean():.2f}")
naive_effect = df[df.program_participation==1]['hourly_earnings'].mean() - df[df.program_participation==0]['hourly_earnings'].mean()
print(f"  Naive difference-in-means: ${naive_effect:.2f}")
print(f"  TRUE CAUSAL EFFECT: ${TRUE_CAUSAL_EFFECT:.2f}")
print(f"  Bias: ${naive_effect - TRUE_CAUSAL_EFFECT:.2f}")

print(f"\nProfile Text Samples:")
print(f"\n[HIGH ABILITY]")
high_idx = df[df.ability_score > 1.5].index[0]
print(f"Ability: {df.loc[high_idx, 'ability_score']:.2f}")
print(df.loc[high_idx, 'profile_text'])

print(f"\n[MEDIUM ABILITY]")
med_idx = df[(df.ability_score > -0.5) & (df.ability_score < 0.5)].index[0]
print(f"Ability: {df.loc[med_idx, 'ability_score']:.2f}")
print(df.loc[med_idx, 'profile_text'])

print(f"\n[LOW ABILITY]")
low_idx = df[df.ability_score < -1.5].index[0]
print(f"Ability: {df.loc[low_idx, 'ability_score']:.2f}")
print(df.loc[low_idx, 'profile_text'])

print(f"\n{'='*70}")
print("✓ PHASE 2 COMPLETE!")
print("="*70)
