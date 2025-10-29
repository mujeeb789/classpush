# data_prep.py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from config import CSV_PATH, DOWNLOAD_DIR, SUFFIXES, SCORE_WEIGHTS, IMAGE_SHAPE

def load_and_score(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    # Fertility score calculation
    weights = SCORE_WEIGHTS
    df['fertility_score'] = (
        weights['ph'][0] * (df['pH_H2O'].fillna(7)/weights['ph'][1]) +
        weights['oc'][0] * (df['OC'].fillna(0)/weights['oc'][1]) +
        weights['erosion'][0] * (1 - df['erosion'].fillna(0)/weights['erosion'][1]) +
        0.3 * df['land_use'].map(weights['land_use']).fillna(0.5)
    ).clip(0, 1)
    return df

def match_images(df):
    image_rows = []
    missing = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Finding images"):
        pointid, nut0 = str(row['POINTID']), row['NUTS_0'].upper()
        sub1, sub2 = pointid[:3], pointid[3:6]
        found = False
        for suffix in SUFFIXES:
            filename = f"{pointid}{suffix}.jpg"
            path = os.path.join(DOWNLOAD_DIR, nut0, sub1, sub2, filename)
            if os.path.exists(path):
                image_rows.append({**row, 'image_path': path})
                found = True
                break
        if not found:
            missing.append(pointid)
    return pd.DataFrame(image_rows), missing

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMAGE_SHAPE[:2])
    return np.array(img)/255.0
