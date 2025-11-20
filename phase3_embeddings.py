"""
Phase 3: Text Embeddings and Dimensionality Reduction
======================================================

Transform profile_text into numerical embeddings and apply PCA for dimensionality reduction.

Steps:
1. Load synthetic data
2. Generate embeddings using sentence-transformers (all-MiniLM-L6-v2)
3. Perform PCA to extract top K components explaining 90% variance
4. Save enhanced dataset
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("="*70)
print("PHASE 3: TEXT EMBEDDINGS & DIMENSIONALITY REDUCTION")
print("="*70)

# Step 1: Load data
print("\n[1/4] Loading synthetic data...")
df = pd.read_parquet('/home/user/Causal---Embeddings-/synthetic_upwork_data.parquet')
print(f"✓ Loaded {len(df)} observations")
print(f"  Columns: {df.shape[1]}")

# Step 2: Generate embeddings
print("\n[2/4] Generating text embeddings...")
print("  Model: all-MiniLM-L6-v2 (384 dimensions)")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['profile_text'].tolist(), show_progress_bar=True, batch_size=64)

print(f"✓ Generated embeddings: {embeddings.shape}")
print(f"  Shape: (n_samples={embeddings.shape[0]}, n_features={embeddings.shape[1]})")

# Verify embeddings correlate with ability
correlation = np.corrcoef(df['ability_score'], embeddings.T)[0, 1:]
print(f"  Max correlation with ability_score: {np.max(np.abs(correlation)):.4f}")
print(f"  Mean absolute correlation: {np.mean(np.abs(correlation)):.4f}")

# Step 3: PCA for dimensionality reduction
print("\n[3/4] Performing PCA...")

# Standardize embeddings
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Fit PCA
pca_full = PCA()
pca_full.fit(embeddings_scaled)

# Find number of components for 90% variance
cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = np.argmax(cumsum_variance >= 0.90) + 1

print(f"  Total components: {embeddings.shape[1]}")
print(f"  Components for 90% variance: {n_components_90}")
print(f"  Components for 95% variance: {np.argmax(cumsum_variance >= 0.95) + 1}")
print(f"  Components for 99% variance: {np.argmax(cumsum_variance >= 0.99) + 1}")

# Use top components
n_components_final = min(n_components_90, 50)  # Cap at 50 for Bayesian model
print(f"  Using {n_components_final} components for analysis")

# Transform to PCA space
pca = PCA(n_components=n_components_final)
pca_components = pca.fit_transform(embeddings_scaled)

print(f"✓ PCA components shape: {pca_components.shape}")
print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.1%}")

# Add to dataframe
for i in range(n_components_final):
    df[f'pca_{i+1}'] = pca_components[:, i]

# Also add first 10 for easy Bayesian model
print(f"  Added pca_1 through pca_{n_components_final} to dataframe")

# Check correlation of PC1 with ability
print(f"\n  Correlation(PC1, ability_score): {df[['pca_1', 'ability_score']].corr().iloc[0, 1]:.4f}")
print(f"  Correlation(PC2, ability_score): {df[['pca_2', 'ability_score']].corr().iloc[0, 1]:.4f}")
print(f"  Correlation(PC3, ability_score): {df[['pca_3', 'ability_score']].corr().iloc[0, 1]:.4f}")

# Step 4: Save enhanced dataset
print("\n[4/4] Saving enhanced dataset...")

# Save embeddings separately (large array)
np.save('/home/user/Causal---Embeddings-/embeddings_raw.npy', embeddings)
np.save('/home/user/Causal---Embeddings-/embeddings_scaled.npy', embeddings_scaled)

# Save PCA object
import joblib
joblib.dump(pca, '/home/user/Causal---Embeddings-/pca_model.pkl')
joblib.dump(scaler, '/home/user/Causal---Embeddings-/scaler.pkl')

# Save enhanced dataframe
output_path = '/home/user/Causal---Embeddings-/data_with_embeddings.parquet'
df.to_parquet(output_path, index=False)
print(f"✓ Saved enhanced data to: {output_path}")
print(f"  New shape: {df.shape}")
print(f"  New columns: {df.shape[1]} (added {n_components_final} PCA components)")

# Generate variance plot
print("\n[Visualization] Creating scree plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
ax1.plot(range(1, min(51, len(pca_full.explained_variance_ratio_))+1),
         pca_full.explained_variance_ratio_[:50],
         marker='o', linewidth=2)
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Variance Explained', fontsize=12)
ax1.set_title('Scree Plot: Variance per Component', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axvline(n_components_final, color='red', linestyle='--', label=f'Selected: {n_components_final} components')
ax1.legend()

# Cumulative variance
ax2.plot(range(1, min(101, len(cumsum_variance))+1),
         cumsum_variance[:100],
         marker='o', linewidth=2, color='green')
ax2.axhline(0.90, color='red', linestyle='--', label='90% threshold')
ax2.axhline(0.95, color='orange', linestyle='--', label='95% threshold')
ax2.axvline(n_components_final, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('Number of Components', fontsize=12)
ax2.set_ylabel('Cumulative Variance Explained', fontsize=12)
ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('/home/user/Causal---Embeddings-/pca_variance_plot.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved plot to: pca_variance_plot.png")

# Summary
print("\n" + "="*70)
print("PHASE 3 SUMMARY")
print("="*70)
print(f"\n✓ Embeddings generated: {embeddings.shape}")
print(f"✓ PCA components: {n_components_final} ({pca.explained_variance_ratio_.sum():.1%} variance)")
print(f"✓ Files saved:")
print(f"  - data_with_embeddings.parquet (main dataset)")
print(f"  - embeddings_raw.npy (raw 384-dim embeddings)")
print(f"  - embeddings_scaled.npy (standardized)")
print(f"  - pca_model.pkl (fitted PCA)")
print(f"  - scaler.pkl (fitted scaler)")
print(f"  - pca_variance_plot.png (visualization)")

print(f"\n✓ Ready for Phase 4: Causal Estimation")
print("="*70)
