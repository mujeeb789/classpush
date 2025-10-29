# =============================================================================
# Comprehensive LUCAS Dataset Exploration Script
# Purpose: Analyze structure, labels, images, and metadata of LUCAS EU dataset
# Author: You + AI Assistant
# Dependencies: pandas, numpy, matplotlib, pillow, geopandas (optional), os, sys
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Optional: for geospatial plotting (install with: pip install geopandas)
try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    print("[INFO] geopandas not installed. Skipping map visualization.")

# =============================================================================
# 1. CONFIGURATION & PATH SETUP
# =============================================================================
# Config parameters (could be moved to a separate config.py)
from config import CSV_PATH, DOWNLOAD_DIR, SUFFIXES
# Define base directories
PROJECT_ROOT = os.getcwd()

csv_file = os.path.join(PROJECT_ROOT, CSV_PATH)
image_folder = os.path.join(PROJECT_ROOT, DOWNLOAD_DIR)
# PROJECT_ROOT = os.getcwd()
# DATA_DIR = os.path.join(PROJECT_ROOT, "CSV_PATH")
# PHOTOS_DIR = os.path.join(DATA_DIR, "DOWNLOAD_DIR")
# METADATA_FILE = os.path.join(DATA_DIR, "LUCAS2018_Metadata.csv")

# Validate paths
# if not os.path.exists(DATA_DIR):
#     raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
# if not os.path.exists(PHOTOS_DIR):
#     raise FileNotFoundError(f"Photos directory not found: {PHOTOS_DIR}")
# if not os.path.exists(METADATA_FILE):
#     raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

# print(f"✅ Project root: {PROJECT_ROOT}")
# print(f"✅ Photos directory: {PHOTOS_DIR}")
# print(f"✅ Metadata file: {METADATA_FILE}")

# =============================================================================
# 2. LAND COVER CLASS DEFINITIONS (LC0 - Level 1)
# Source: Eurostat LUCAS nomenclature
# =============================================================================

LC0_CODE_TO_NAME = {
    'A': 'Artificial surfaces',
    'B': 'Agricultural areas',
    'C': 'Forest and semi-natural areas',
    'D': 'Wetlands',
    'E': 'Water bodies',
    'F': 'Permanent crops',
    'G': 'Open spaces with little or no vegetation',
    'H': 'Inland wetlands (supplementary)'
}

LC0_COLORS = {
    'A': '#FF5733',  # Red-orange
    'B': '#C7D36D',  # Olive green
    'C': '#2E8B57',  # Sea green
    'D': '#4682B4',  # Steel blue
    'E': '#1E90FF',  # Dodger blue
    'F': '#FFD700',  # Gold
    'G': '#D2B48C',  # Tan
    'H': '#8A2BE2'   # Blue violet
}

print("\n📘 LC0 Land Cover Classes:")
for code, name in LC0_CODE_TO_NAME.items():
    print(f"  {code} → {name}")

# =============================================================================
# 3. LOAD METADATA WITH ROBUST ERROR HANDLING
# =============================================================================

print("\n📥 Loading metadata...")
try:
    # Use low_memory=False to avoid dtype warnings on large files
    df_meta = pd.read_csv(csv_file, low_memory=False)
    print(f"✅ Metadata loaded: {df_meta.shape[0]:,} rows, {df_meta.shape[1]} columns")
except Exception as e:
    print(f"❌ Failed to load metadata: {e}")
    sys.exit(1)

# Check required columns
required_cols = ['OC', 'pH_H2O', 'P', 'K', 'N', 'EC', 'CaCO3', 
                    'OC (20-30 cm)', 'CaCO3 (20-30 cm)', 'Ox_Al', 'Ox_Fe']
# ['POINT_ID', 'LC0', 'X', 'Y']
missing_cols = [col for col in required_cols if col not in df_meta.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in metadata: {missing_cols}")

# =============================================================================
# 4. BUILD IMAGE PATHS AND VERIFY EXISTENCE
# =============================================================================

print("\n🔍 Building image paths and verifying existence...")

def build_photo_path(point_id):
    """Construct panoramic photo path from POINT_ID"""
    # POINT_ID in metadata is integer (e.g., 1), but filename is P000001_1.jpg
    return os.path.join(image_folder, f"P{point_id:06d}_1.jpg")

# Add photo path column
df_meta['photo_path'] = df_meta['POINT_ID'].apply(build_photo_path)

# Check if file exists
df_meta['photo_exists'] = df_meta['photo_path'].apply(os.path.exists)

# Filter to only existing photos
df_valid = df_meta[df_meta['photo_exists']].copy().reset_index(drop=True)
df_missing = df_meta[~df_meta['photo_exists']]

print(f"✅ Valid images found: {len(df_valid):,}")
print(f"⚠️  Missing images: {len(df_missing):,}")

if len(df_missing) > 0:
    print("   Example missing POINT_IDs:", df_missing['POINT_ID'].head().tolist())

# =============================================================================
# 5. FILTER AND CLEAN LABELS (LC0)
# =============================================================================

print("\n🧹 Cleaning LC0 labels...")

# Keep only valid LC0 codes
valid_lc0_mask = df_valid['LC0'].isin(LC0_CODE_TO_NAME.keys())
df_clean = df_valid[valid_lc0_mask].copy().reset_index(drop=True)
df_invalid_labels = df_valid[~valid_lc0_mask]

print(f"✅ After LC0 filtering: {len(df_clean):,} samples")
if len(df_invalid_labels) > 0:
    print(f"⚠️  Invalid LC0 labels: {len(df_invalid_labels):,}")
    print("   Example invalid labels:", df_invalid_labels['LC0'].unique())

# Add human-readable label
df_clean['lc0_name'] = df_clean['LC0'].map(LC0_CODE_TO_NAME)

# =============================================================================
# 6. DETAILED STATISTICAL SUMMARY
# =============================================================================

print("\n📊 Dataset Summary:")
print("-" * 50)
print(f"Total survey points in metadata: {len(df_meta):,}")
print(f"Points with photos available:    {len(df_valid):,} ({len(df_valid)/len(df_meta)*100:.1f}%)")
print(f"Points with valid LC0 labels:    {len(df_clean):,} ({len(df_clean)/len(df_meta)*100:.1f}%)")
print(f"Number of land cover classes:    {len(LC0_CODE_TO_NAME)}")

# Class distribution
class_counts = df_clean['LC0'].value_counts().reindex(LC0_CODE_TO_NAME.keys(), fill_value=0)
class_percent = (class_counts / len(df_clean)) * 100

print("\n📈 Class Distribution (LC0):")
print("-" * 50)
for code in LC0_CODE_TO_NAME.keys():
    count = class_counts[code]
    pct = class_percent[code]
    name = LC0_CODE_TO_NAME[code]
    print(f"{code} | {name:<35} | {count:>6,} ({pct:>5.2f}%)")

# =============================================================================
# 7. VISUALIZE CLASS DISTRIBUTION
# =============================================================================

print("\n🖼️  Generating class distribution plots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot
colors = [LC0_COLORS[code] for code in class_counts.index]
bars = axes[0].bar(class_counts.index, class_counts.values, color=colors)
axes[0].set_title('LUCAS Land Cover Distribution (LC0) - Bar Plot', fontsize=14)
axes[0].set_ylabel('Number of Samples')
axes[0].set_xlabel('Land Cover Class (LC0)')
axes[0].tick_params(axis='x', rotation=0)

# Annotate bars
for bar, count in zip(bars, class_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_counts)*0.01,
                 f'{count:,}', ha='center', va='bottom', fontsize=9)

# Pie chart
wedges, texts, autotexts = axes[1].pie(
    class_counts.values,
    labels=[LC0_CODE_TO_NAME[code] for code in class_counts.index],
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 9}
)
axes[1].set_title('LUCAS Land Cover Distribution (LC0) - Pie Chart', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "lucas_class_distribution.png"), dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 8. VISUALIZE SAMPLE IMAGES PER CLASS
# =============================================================================

print("\n🖼️  Generating sample image grid...")

n_classes = len(LC0_CODE_TO_NAME)
cols = 4
rows = (n_classes + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
axes = axes.flatten() if n_classes > 1 else [axes]

for idx, (code, name) in enumerate(LC0_CODE_TO_NAME.items()):
    ax = axes[idx]
    # Get one random sample from this class
    class_samples = df_clean[df_clean['LC0'] == code]
    if not class_samples.empty:
        sample_row = class_samples.sample(1).iloc[0]
        img_path = sample_row['photo_path']
        try:
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_title(f"{code}: {name}\n(ID: {sample_row['POINT_ID']})", fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f"Load error:\n{str(e)[:30]}...", 
                    ha='center', va='center', transform=ax.transAxes, color='red')
    else:
        ax.text(0.5, 0.5, f"{name}\n(NO SAMPLES)", 
                ha='center', va='center', transform=ax.transAxes, color='gray')
    ax.axis('off')

# Hide unused subplots
for j in range(idx + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Sample Panoramic Images by Land Cover Class (LUCAS)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(PROJECT_ROOT, "lucas_sample_images.png"), dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 9. IMAGE QUALITY & DIMENSIONS ANALYSIS
# =============================================================================

print("\n🔍 Analyzing image dimensions...")

def get_image_size(path):
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except:
        return (None, None)

# Sample 1000 images to avoid long wait (adjust as needed)
sample_df = df_clean.sample(n=min(1000, len(df_clean)), random_state=42)
sample_df['img_width'], sample_df['img_height'] = zip(*sample_df['photo_path'].apply(get_image_size))

# Remove failed reads
sample_df = sample_df.dropna(subset=['img_width', 'img_height'])
sample_df['img_width'] = sample_df['img_width'].astype(int)
sample_df['img_height'] = sample_df['img_height'].astype(int)
sample_df['aspect_ratio'] = sample_df['img_width'] / sample_df['img_height']

print(f"✅ Analyzed {len(sample_df)} images")
print(f"📏 Avg. width: {sample_df['img_width'].mean():.0f} px")
print(f"📏 Avg. height: {sample_df['img_height'].mean():.0f} px")
print(f"📐 Avg. aspect ratio: {sample_df['aspect_ratio'].mean():.2f}")

# Plot dimensions
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].hist(sample_df['img_width'], bins=30, color='skyblue', edgecolor='black')
ax[0].set_title('Image Width Distribution')
ax[0].set_xlabel('Width (pixels)')

ax[1].hist(sample_df['img_height'], bins=30, color='lightgreen', edgecolor='black')
ax[1].set_title('Image Height Distribution')
ax[1].set_xlabel('Height (pixels)')

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "lucas_image_dimensions.png"), dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 10. (OPTIONAL) GEOSPATIAL OVERVIEW
# =============================================================================

if GEOSPATIAL_AVAILABLE:
    print("\n🌍 Generating geospatial overview...")
    try:
        # Create geometry
        geometry = [Point(xy) for xy in zip(df_clean['X'], df_clean['Y'])]
        gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:3035")  # ETRS89 LAEA

        # Get EU boundary (simplified)
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        eu_countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark',
                        'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland',
                        'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands',
                        'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden']
        eu = world[world['name'].isin(eu_countries)]

        # Reproject to same CRS
        eu = eu.to_crs("EPSG:3035")

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        eu.plot(ax=ax, color='lightgrey', edgecolor='white')
        gdf.plot(ax=ax, column='LC0', cmap='tab10', legend=True, markersize=1)
        ax.set_title('LUCAS Survey Points Across Europe (2018)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_ROOT, "lucas_geospatial.png"), dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"[WARN] Geospatial plot failed: {e}")
else:
    print("\n🌍 Skipping geospatial plot (geopandas not available)")

# =============================================================================
# 11. SAVE CLEANED DATASET FOR TRAINING
# =============================================================================

output_path = os.path.join(PROJECT_ROOT, "lucas_clean_metadata.csv")
df_clean.to_csv(output_path, index=False)
print(f"\n💾 Cleaned metadata saved to: {output_path}")
print(f"   Columns: {list(df_clean.columns)}")

# Final summary
print("\n" + "="*70)
print("✅ EXPLORATION COMPLETE!")
print("="*70)
print(f"Final dataset size: {len(df_clean):,} images")
print(f"Classes: {len(class_counts[class_counts > 0])} / {len(LC0_CODE_TO_NAME)} present")
print(f"Saved cleaned CSV for training: {output_path}")
print("="*70)
