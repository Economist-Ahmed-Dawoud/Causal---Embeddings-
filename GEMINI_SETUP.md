# Gemini API Setup Guide

## Quick Start (5 minutes)

### Step 1: Get Your FREE Gemini API Key

1. Visit: **https://aistudio.google.com/app/apikey**
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy your API key (starts with `AIza...`)

### Step 2: Configure Your Environment

Create a `.env` file in the project directory:

```bash
echo "GEMINI_API_KEY=AIzaSy..." > .env
```

Replace `AIzaSy...` with your actual API key from Step 1.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `python-dotenv` - For environment variable management
- `requests` - For API calls
- `numpy`, `pandas`, `scipy` - Data processing
- `tqdm` - Progress bars
- Other required packages

### Step 4: Generate Realistic Profiles!

```bash
python generate_gemini_profiles.py --n_samples 5000
```

## Features

### Checkpoint Support
The script automatically saves progress every 100 profiles. If interrupted, just run again:

```bash
python generate_gemini_profiles.py --n_samples 5000
# Automatically resumes from last checkpoint
```

To start fresh (ignore checkpoint):
```bash
python generate_gemini_profiles.py --n_samples 5000 --no_resume
```

### Sample Sizes

Generate different sample sizes:
```bash
# Small test run (fast!)
python generate_gemini_profiles.py --n_samples 100

# Medium dataset
python generate_gemini_profiles.py --n_samples 1000

# Full dataset
python generate_gemini_profiles.py --n_samples 5000
```

## What Makes This Different?

### ❌ Old Template-Based Approach
```python
# generate_data_final.py uses templates like:
"I am a {category} freelancer based in {city}..."
```
- **Pros**: Fast, no API needed
- **Cons**: Repetitive, unrealistic patterns

### ✅ New Gemini AI Approach
```python
# generate_gemini_profiles.py uses Gemini AI to generate:
"As an accomplished Web Development specialist with 8 years
of expertise, I deliver transformative solutions that drive
measurable ROI for my clients. Based in Cairo, Egypt, I
leverage cutting-edge methodologies..."
```
- **Pros**: Realistic, natural variation, authentic writing styles
- **Cons**: Slower (~2-3 minutes for 100 profiles), requires API key

## How It Works

The script generates different profile styles based on **ability score**:

### High Ability (ability_score > 1.0)
```
Gemini prompt: "Write a highly professional, articulate,
and persuasive profile..."

Result: Professional vocabulary, sophisticated language,
client-focused messaging
```

### Medium Ability (-1.0 < ability_score < 1.0)
```
Gemini prompt: "Write a standard, competent profile..."

Result: Clear language, straightforward, no errors
```

### Low Ability (ability_score < -1.0)
```
Gemini prompt: "Write a short, basic profile with minor
grammatical imperfections..."

Result: Simple language, casual phrasing, occasional errors
```

This creates **realistic variation** that the text embeddings can capture!

## Expected Runtime

- **100 profiles**: ~2-3 minutes
- **1,000 profiles**: ~20-30 minutes
- **5,000 profiles**: ~2 hours

The script includes:
- ✅ Progress bars
- ✅ Automatic checkpoints every 100 profiles
- ✅ Rate limiting (pause every 50 requests)
- ✅ Retry logic for API failures
- ✅ Resume capability

## Troubleshooting

### Error: "API KEY NOT CONFIGURED"
**Solution**: Make sure you created the `.env` file with your API key

```bash
# Check if .env exists
cat .env

# Should show:
GEMINI_API_KEY=AIzaSy...
```

### Error: "403 Forbidden" or "Invalid API Key"
**Solution**: Your API key is invalid or expired

1. Visit: https://aistudio.google.com/app/apikey
2. Delete old key
3. Create new key
4. Update `.env` file with new key

### Error: "Timeout" or "Connection Error"
**Solution**: Network issues - the script will automatically retry

- The script retries 3 times with exponential backoff
- Checkpoints are saved, so you won't lose progress

### Slow Generation Speed
**Solution**: This is normal! Gemini API has rate limits

- Expect ~30-40 profiles per minute
- Use `--n_samples 100` for testing first
- The script pauses every 50 requests to respect rate limits

## Cost

**Gemini API is FREE** for most use cases!

- Free tier: 15 requests per minute
- Our script respects rate limits
- 5,000 profiles = ~5,000 API calls (well within free tier daily limits)

Check current limits: https://ai.google.dev/pricing

## Comparison: Template vs Gemini

| Feature | Template (`generate_data_final.py`) | Gemini (`generate_gemini_profiles.py`) |
|---------|-------------------------------------|----------------------------------------|
| **Speed** | Instant (5,000 in 10 seconds) | ~2 hours for 5,000 |
| **Realism** | Repetitive templates | Natural variation |
| **API Key** | Not needed | Required (FREE) |
| **Quality** | Sufficient for demo | Realistic for publication |
| **Checkpoint** | No (not needed) | Yes (resume capability) |
| **Cost** | Free | Free (with rate limits) |

## Best Practices

1. **Test first**: Start with `--n_samples 100` to verify your API key works
2. **Use checkpoints**: The script auto-saves progress - just re-run if interrupted
3. **Check samples**: Review the 3 sample profiles printed at the end
4. **Verify output**: Check that `synthetic_upwork_data.parquet` was created

## Example Session

```bash
$ python generate_gemini_profiles.py --n_samples 100

======================================================================
API KEY STATUS
======================================================================
✓ API key found: AIzaSyAR...jU8
======================================================================

======================================================================
SYNTHETIC DATA GENERATION: Causal Inference with Text Embeddings
======================================================================

Parameters:
  Sample size: 100
  TRUE CAUSAL EFFECT: $5.00
  Resume from checkpoint: True
  Gemini Model: 1.5 Flash

======================================================================

Generating structured variables for 100 freelancers...
✓ Generated 100 observations
  Treatment rate: 88.0%
  Mean earnings (Treated): $24.56
  Mean earnings (Control): $8.23
  Naive effect: $16.33
  True effect: $5.00

======================================================================
GENERATING REALISTIC PROFILES USING GEMINI API
======================================================================
Model: Gemini 1.5 Flash
Total profiles to generate: 100
Checkpoint file: checkpoint_100.parquet
======================================================================

🚀 Starting profile generation...

Generating profiles: 100%|████████████| 100/100 [02:34<00:00,  1.54s/it]

💾 Checkpoint saved at row 100

======================================================================
GENERATION COMPLETE
======================================================================
✓ Successfully generated: 100 profiles
======================================================================

📝 SAMPLE PROFILES BY ABILITY TIER:

----------------------------------------------------------------------
[HIGH ABILITY - Ability Score > 1.0]
----------------------------------------------------------------------
Freelancer ID: EGY_00023
Ability Score: 2.18
Category: Web Development
Experience: 8.3 years

Profile Text:
As an accomplished Web Development specialist with 8 years of
expertise, I deliver transformative solutions that drive measurable
ROI for my clients...

[Additional samples...]

✓ ALL PHASES COMPLETE!
```

## Next Steps

After generating profiles:

1. **Verify output**: Check `synthetic_upwork_data.parquet` exists
2. **View data**:
   ```python
   import pandas as pd
   df = pd.read_parquet('synthetic_upwork_data.parquet')
   print(df.head())
   print(df['profile_text'].iloc[0])  # View a profile
   ```
3. **Continue to Phase 3**: Run `python phase3_embeddings.py` to generate embeddings
4. **Run causal estimation**: `python phase4_causal_estimation.py`

## Support

If you encounter issues:

1. Check this guide first
2. Verify your API key at: https://aistudio.google.com/app/apikey
3. Review the error message carefully
4. Try with a smaller sample size first (`--n_samples 10`)

## Summary

✅ **Free API key** from Google AI Studio
✅ **5-minute setup**
✅ **Realistic profiles** with natural variation
✅ **Checkpoint support** for long runs
✅ **Automatic retry** for network issues
✅ **Progress tracking** with tqdm

**You're ready to generate realistic freelancer profiles!** 🚀
