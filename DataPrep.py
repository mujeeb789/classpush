# data_prep.py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from config import CSV_PATH, DOWNLOAD_DIR, SUFFIXES, SCORE_WEIGHTS, IMAGE_SHAPE

def load_and_clean_data(csv_path=CSV_PATH):
    """Load and clean the soil data with robust numeric conversion"""
    # Load data with proper numeric parsing
    df = pd.read_csv(csv_path, 
                    converters={
                        'OC': lambda x: pd.to_numeric(x, errors='coerce'),
                        'pH_H2O': lambda x: pd.to_numeric(x, errors='coerce'),
                        'P': lambda x: pd.to_numeric(x, errors='coerce'),
                        'K': lambda x: pd.to_numeric(x, errors='coerce'),
                        'N': lambda x: pd.to_numeric(x, errors='coerce'),
                        'EC': lambda x: pd.to_numeric(x, errors='coerce'),
                        'CaCO3': lambda x: pd.to_numeric(x, errors='coerce')
                    })
    
    # Handle special cases like '< LOD' (Limit of Detection)
    numeric_cols = ['OC', 'pH_H2O', 'P', 'K', 'N', 'EC', 'CaCO3']
    for col in numeric_cols:
        if col in df.columns:
            # Replace '< LOD' with NaN (will be filled with median later)
            df[col] = df[col].replace('< LOD', np.nan)
            # Convert to numeric in case any strings remain
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Select relevant columns
    relevant_cols = [
        'POINTID', 'NUTS_0', 'pH_H2O', 'pH_CaCl2', 'OC', 'P', 'K', 'N', 
        'EC', 'CaCO3', 'LC0_Desc', 'LC1_Desc', 'LU1_Desc', 'TH_LAT', 'TH_LONG'
    ]
    
    # Keep only relevant columns that exist in the dataframe
    df = df[[col for col in relevant_cols if col in df.columns]]
    
    # Handle missing values
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Convert land use to categorical codes if needed
    if 'LU1_Desc' in df.columns:
        df['land_use_code'] = pd.Categorical(df['LU1_Desc']).codes
    
    return df

def classify_soil_fertility(df):
    """
    Enhanced fertility classification with categories and scores
    """
    # 1. Individual Parameter Classifications
    conditions_oc = [
        (df['OC'] < 10),
        (df['OC'] >= 10) & (df['OC'] <= 20),
        (df['OC'] > 20)
    ]
    choices_oc = ['Low', 'Moderate', 'High']
    df['OC_Class'] = np.select(conditions_oc, choices_oc, default='Unknown')
    
    conditions_ph = [
        (df['pH_H2O'] < 5.5),
        (df['pH_H2O'] >= 5.5) & (df['pH_H2O'] <= 6.8),
        (df['pH_H2O'] > 6.8)
    ]
    choices_ph = ['Acidic', 'Neutral', 'Alkaline']
    df['pH_Class'] = np.select(conditions_ph, choices_ph, default='Unknown')
    
    # 2. Combined Fertility Classification
    conditions_fertility = [
        (df['OC'] < 10) | (df['pH_H2O'] < 5.5),
        ((df['OC'] >= 10) & (df['OC'] <= 20)) & 
        ((df['pH_H2O'] >= 5.5) & (df['pH_H2O'] <= 6.8)),
        (df['OC'] > 20) | (df['pH_H2O'] > 6.8)
    ]
    choices_fertility = ['Low', 'Moderate', 'High']
    df['Fertility_Class'] = np.select(conditions_fertility, choices_fertility, default='Unknown')
    
    # 3. Enhanced Fertility Score (0-100 scale)
    df['Fertility_Score_Simple'] = (
        np.where(df['OC'] < 10, 20,
        np.where(df['OC'] <= 20, 40, 60)) +
        np.where(df['pH_H2O'] < 5.5, 10,
        np.where(df['pH_H2O'] <= 6.8, 30, 40))
    )
    
    # 4. Detailed Fertility Description
    df['Fertility_Description'] = (
        "Organic Carbon: " + df['OC_Class'] + " (" + df['OC'].round(1).astype(str) + " g/kg), " +
        "pH: " + df['pH_Class'] + " (" + df['pH_H2O'].round(1).astype(str) + ")"
    )
    
    return df

def calculate_fertility_score(df, weights=None):
    """Advanced fertility scoring incorporating multiple factors"""
    # Default weights if not provided
    if weights is None:
        weights = {
            'ph': {'weight': 0.3, 'min': 4, 'max': 9},
            'oc': {'weight': 0.4, 'max': 300},
            'nutrients': {'weight': 0.2},
            'land_use': {'weight': 0.1}
        }
    
    # Ensure required columns exist
    required_cols = ['pH_H2O', 'OC', 'N', 'P', 'K', 'land_use_code']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    try:
        # Normalize each parameter
        df['ph_norm'] = np.clip(
            (df['pH_H2O'] - weights['ph']['min']) / 
            (weights['ph']['max'] - weights['ph']['min']), 
            0, 1
        )
        
        df['oc_norm'] = np.clip(
            df['OC'] / weights['oc']['max'], 
            0, 1
        )
        
        # Calculate nutrient balance index (N:P:K ratio)
        df['nutrient_balance'] = (
            np.log1p(df['N'].fillna(0)) + 
            np.log1p(df['P'].fillna(0)) + 
            np.log1p(df['K'].fillna(0))
        )
        
        # Normalize nutrient balance
        if df['nutrient_balance'].nunique() > 1:
            df['nutrient_balance_norm'] = (
                (df['nutrient_balance'] - df['nutrient_balance'].min()) / 
                (df['nutrient_balance'].max() - df['nutrient_balance'].min())
            )
        else:
            df['nutrient_balance_norm'] = 0.5  # Default if all values are same
            
        # Normalize land use codes
        if df['land_use_code'].nunique() > 1:
            land_use_norm = df['land_use_code'] / df['land_use_code'].max()
        else:
            land_use_norm = 0.5  # Default if all values are same
            
        # Calculate composite score (0-1 range)
        df['fertility_score_advanced'] = (
            weights['ph']['weight'] * df['ph_norm'] +
            weights['oc']['weight'] * df['oc_norm'] +
            weights['nutrients']['weight'] * df['nutrient_balance_norm'] +
            weights['land_use']['weight'] * land_use_norm
        )
        
        # Scale advanced score to 0-100 for comparison
        df['fertility_score_advanced'] = df['fertility_score_advanced'] * 100
        
        return df
        
    except KeyError as e:
        raise ValueError(f"Invalid weights structure. Missing key: {e}")
    except Exception as e:
        raise ValueError(f"Error calculating fertility score: {str(e)}")

def match_images(df, download_dir=DOWNLOAD_DIR, suffixes=SUFFIXES):
    """Match satellite images to soil samples"""
    image_rows = []
    missing = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Finding images"):
        pointid = str(row['POINTID'])
        nut0 = row['NUTS_0'].upper() if pd.notna(row['NUTS_0']) else 'UNKNOWN'
        
        # Try multiple directory structures
        search_paths = [
            os.path.join(download_dir, nut0, pointid[:3], pointid[3:6]),
            os.path.join(download_dir, nut0, pointid[:4], pointid[4:6]),
            os.path.join(download_dir, nut0, pointid)
        ]
        
        found = False
        for suffix in suffixes:
            for base_path in search_paths:
                filename = f"{pointid}{suffix}.jpg"
                path = os.path.join(base_path, filename)
                if os.path.exists(path):
                    image_rows.append({**row.to_dict(), 'image_path': path})
                    found = True
                    break
            if found:
                break
                
        if not found:
            missing.append(pointid)
            
    return pd.DataFrame(image_rows), missing

def preprocess_image(img_path, target_size=IMAGE_SHAPE[:2]):
    """Preprocess satellite images for analysis"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        
        if img_array.mean() < 0.1 or img_array.mean() > 0.9:
            return None  # likely invalid image
            
        return img_array
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None



def full_pipeline(csv_path=CSV_PATH):
    """Complete data preparation pipeline with enhanced fertility analysis"""
    try:
        # Load and clean data
        df = load_and_clean_data(csv_path)
        
        # Apply both classification systems
        df = classify_soil_fertility(df)
        
        # Calculate fertility score with proper weights
        # Either use from config or the defaults we defined
        try:
            from config import SCORE_WEIGHTS
            weights = SCORE_WEIGHTS
        except (ImportError, AttributeError):
            weights = None  # Will use defaults
            
        df = calculate_fertility_score(df, weights)
        
        # Match images
        df_with_images, missing = match_images(df)
        
        print(f"\nFertility Classification Summary:")
        print(df['Fertility_Class'].value_counts())
        print(f"\nFound images for {len(df_with_images)}/{len(df)} samples")
        if missing:
            print(f"Missing images for {len(missing)} samples")
        
        return df_with_images
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise
    
    



if __name__ == "__main__":
    df = full_pipeline()
    print("\nSample data with fertility classifications:")
    print(df[['POINTID', 'OC', 'pH_H2O', 'Fertility_Class', 
             'Fertility_Score_Simple', 'fertility_score_advanced']].head())