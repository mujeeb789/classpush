# Enhanced Soil Fertility and Land Use Classification Model
# Combines image data with tabular soil data for improved predictions.
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.efficientnet import preprocess_input as ef_preprocess
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tqdm import tqdm
from skimage import exposure
import joblib

# Try to import user config, else define defaults

from config import CSV_PATH, DOWNLOAD_DIR, SUFFIXES, SCORE_WEIGHTS, IMAGE_SHAPE
# except Exception:
#     CSV_PATH = "soil_data.csv"
#     DOWNLOAD_DIR = "./images"
#     SUFFIXES = ["", "_sat"]
#     SCORE_WEIGHTS = {
#         'ph': {'min': 3.5, 'max': 9.0, 'weight': 0.25},
#         'oc': {'max': 50.0, 'weight': 0.35},
#         'nutrients': {'weight': 0.25},
#         'land_use': {'weight': 0.15}
#     }
#     IMAGE_SHAPE = (224, 224, 3)

# Hyperparams
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
PATIENCE = 8
DROPOUT_RATE = 0.3

# ----------------- Data loading & preprocessing -----------------
def load_and_clean_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path, low_memory=False)
    # Robust numeric conversion for expected numeric columns
    numeric_cols = ['OC', 'pH_H2O', 'P', 'K', 'N', 'EC', 'CaCO3', 
                    'OC (20-30 cm)', 'CaCO3 (20-30 cm)', 'Ox_Al', 'Ox_Fe']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace('< LOD', '', regex=False)
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].fillna(df[c].median())

    if 'Depth' in df.columns:
        df['Depth_code'] = pd.Categorical(df['Depth']).codes

    if 'LU1_Desc' in df.columns:
        df['land_use_code'] = pd.Categorical(df['LU1_Desc']).codes

    return df

def classify_soil_fertility(df):
    # Added safe guards for missing columns
    if 'OC' not in df.columns:
        df['OC'] = 0.0
    if 'pH_H2O' not in df.columns:
        df['pH_H2O'] = 6.0

    conditions_oc = [
        (df['OC'] < 10),
        (df['OC'] >= 10) & (df['OC'] <= 20),
        (df['OC'] > 20)
    ]
    df['OC_Class'] = np.select(conditions_oc, ['Low', 'Moderate', 'High'], default='Unknown')

    conditions_ph = [
        (df['pH_H2O'] < 5.5),
        (df['pH_H2O'] >= 5.5) & (df['pH_H2O'] <= 6.8),
        (df['pH_H2O'] > 6.8)
    ]
    df['pH_Class'] = np.select(conditions_ph, ['Acidic', 'Neutral', 'Alkaline'], default='Unknown')

    conditions_fertility = [
        (df['OC'] < 10) | (df['pH_H2O'] < 5.5),
        ((df['OC'] >= 10) & (df['OC'] <= 20)) & ((df['pH_H2O'] >= 5.5) & (df['pH_H2O'] <= 6.8)),
        (df['OC'] > 20) | (df['pH_H2O'] > 6.8)
    ]
    df['Fertility_Class'] = np.select(conditions_fertility, ['Low', 'Moderate', 'High'], default='Unknown')

    df['Fertility_Score_Simple'] = (
        np.where(df['OC'] < 10, 20,
                 np.where(df['OC'] <= 20, 40, 60)) +
        np.where(df['pH_H2O'] < 5.5, 10,
                 np.where(df['pH_H2O'] <= 6.8, 30, 40))
    )
    return df

def calculate_fertility_score(df, weights=None):
    if weights is None:
        weights = SCORE_WEIGHTS

    required_cols = ['pH_H2O', 'OC', 'N', 'P', 'K', 'land_use_code']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    df['ph_norm'] = np.clip((df['pH_H2O'] - weights['ph']['min']) / (weights['ph']['max'] - weights['ph']['min']), 0, 1)
    df['oc_norm'] = np.clip(df['OC'] / max(1.0, weights['oc']['max']), 0, 1)

    df['nutrient_balance'] = np.log1p(df['N'].fillna(0)) + np.log1p(df['P'].fillna(0)) + np.log1p(df['K'].fillna(0))
    if df['nutrient_balance'].nunique() > 1:
        df['nutrient_balance_norm'] = (df['nutrient_balance'] - df['nutrient_balance'].min()) / (df['nutrient_balance'].max() - df['nutrient_balance'].min())
    else:
        df['nutrient_balance_norm'] = 0.5

    if df['land_use_code'].nunique() > 1:
        land_use_norm = df['land_use_code'] / df['land_use_code'].max()
    else:
        land_use_norm = 0.5
    df['fertility_score_advanced'] = (
        weights['ph']['weight'] * df['ph_norm'] +
        weights['oc']['weight'] * df['oc_norm'] +
        weights['nutrients']['weight'] * df['nutrient_balance_norm'] +
        weights['land_use']['weight'] * land_use_norm
    ) * 100.0
    return df

def match_images(df, download_dir=DOWNLOAD_DIR, suffixes=SUFFIXES):
    image_rows = []
    missing = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Finding images"):
        pointid = str(row.get('POINTID', ''))
        nut0 = str(row.get('NUTS_0', '')).upper() if pd.notna(row.get('NUTS_0', None)) else 'UNKNOWN'
        search_paths = [
            os.path.join(download_dir, nut0, pointid[:3], pointid[3:6]) if pointid else None,
            os.path.join(download_dir, nut0, pointid[:4], pointid[4:6]) if pointid else None,
            os.path.join(download_dir, nut0, pointid) if pointid else None
        ]
        found = False
        for suffix in suffixes:
            for base_path in [p for p in search_paths if p]:
                filename = f"{pointid}{suffix}.jpg"
                path = os.path.join(base_path, filename)
                if os.path.exists(path):
                    row2 = row.copy()
                    row2['image_path'] = path
                    image_rows.append(row2)
                    found = True
                    break
            if found:
                break
        if not found:
            missing.append(pointid)
    if not image_rows:
        return pd.DataFrame(columns=list(df.columns) + ['image_path']), missing
    return pd.DataFrame(image_rows), missing

# ---------------- Image loading ----------------


def load_image(image_path, target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1])):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            if min(img.size) < 10:
                return None
            img = img.resize(target_size, Image.LANCZOS)
            arr = np.array(img).astype(np.float32)
            # Use model-specific preprocessing (EfficientNet)
            arr = ef_preprocess(arr)
            return arr
    except Exception as e:
        # return a zero image on failure
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)

# ---------------- Model ----------------
def create_augmentation_layer():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
    ], name="augmentation")

def build_enhanced_model(num_fertility_classes, num_land_use_classes, num_csv_features):
    img_input = layers.Input(shape=IMAGE_SHAPE, name='image_input')
    csv_input = layers.Input(shape=(num_csv_features,), name='csv_input')

    augmented = create_augmentation_layer()(img_input)

    base_model = tf.keras.applications.EfficientNetB3(include_top=False, weights='imagenet', input_tensor=augmented, pooling='avg')
    # Initially freeze the backbone to train CSV + heads faster
    base_model.trainable = False

    img_features = base_model.output
    img_features = layers.Dropout(DROPOUT_RATE)(img_features)

    csv_features = layers.Dense(128, activation='relu')(csv_input)
    csv_features = layers.BatchNormalization()(csv_features)
    csv_features = layers.Dropout(DROPOUT_RATE)(csv_features)

    combined = layers.Concatenate()([img_features, csv_features])
    combined = layers.Dense(256, activation='relu')(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dropout(DROPOUT_RATE)(combined)

    # Regression head
    fertility_reg = layers.Dense(128, activation='relu')(combined)
    fertility_reg_output = layers.Dense(1, activation='linear', name='fertility_reg')(fertility_reg)

    # Classification head
    fertility_cls = layers.Dense(128, activation='relu')(combined)
    fertility_cls_output = layers.Dense(num_fertility_classes, activation='softmax', name='fertility_cls')(fertility_cls)

    # Land use head
    land_use = layers.Dense(128, activation='relu')(combined)
    land_use_output = layers.Dense(num_land_use_classes, activation='softmax', name='land_use')(land_use)

    model = models.Model(inputs=[img_input, csv_input], outputs=[fertility_reg_output, fertility_cls_output, land_use_output])
    return model

# ---------------- Generator ----------------
class EnhancedSoilGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, csv_features, batch_size=32, target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]), shuffle=True):
        self.df = df.reset_index(drop=True).copy()
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.csv_features = [f for f in csv_features if f in df.columns]
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.df))
        batch_df = self.df.iloc[start:end]

        X_img = []
        X_csv = []
        y_reg = []
        y_fert = []
        y_land = []

        for _, row in batch_df.iterrows():
            img = load_image(row.get('image_path', ''), target_size=self.target_size)
            if img is None:
                img = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
            X_img.append(img)

            csv_vals = []
            for f in self.csv_features:
                val = row.get(f, 0.0)
                try:
                    val = float(val) if (pd.notna(val) and val != '') else 0.0
                except Exception:
                    val = 0.0
                csv_vals.append(val)
            X_csv.append(csv_vals)

            # Targets
            y_reg.append(float(row.get('fertility_score_advanced', 0.0)))
            y_fert.append(int(row.get('fertility_class_encoded', 0)))
            y_land.append(int(row.get('land_use_encoded', 0)))

        X_img = np.stack(X_img).astype(np.float32)
        X_csv = np.array(X_csv, dtype=np.float32)
        y_reg = np.array(y_reg, dtype=np.float32).reshape(-1, 1)
        y_fert = np.array(y_fert, dtype=np.int32)
    
        y_land = np.array(y_land, dtype=np.int32)


        sample_weights = {
            "fertility_reg": np.ones(len(y_reg), dtype=np.float32),  # neutral
            "fertility_cls": np.array([class_weights["fertility_cls"].get(c, 1.0) for c in y_fert], dtype=np.float32),
            "land_use": np.array([class_weights["land_use"].get(c, 1.0) for c in y_land], dtype=np.float32),
        }

        return [X_img, X_csv], {
            "fertility_reg": y_reg,
            "fertility_cls": y_fert,
            "land_use": y_land,
        }, sample_weights
        
    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)
            self.df = self.df.iloc[self.indices].reset_index(drop=True)

# ---------------- Utilities ----------------
def prepare_datasets(df, csv_features, test_size=0.2):
    if 'fertility_class_encoded' not in df.columns:
        raise ValueError("df must contain 'fertility_class_encoded' before splitting.")
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=test_size, random_state=42, stratify=df['fertility_class_encoded'])
    train_gen = EnhancedSoilGenerator(df.iloc[train_idx], csv_features, batch_size=BATCH_SIZE, shuffle=True)
    test_gen = EnhancedSoilGenerator(df.iloc[test_idx], csv_features, batch_size=BATCH_SIZE, shuffle=False)
    return train_gen, test_gen

def compute_class_weights_from_df(df):
    # Fertility classes
    classes, counts = np.unique(df['fertility_class_encoded'], return_counts=True)
    fertility_weights = dict(zip(classes, (len(df) / (len(classes) * counts)).tolist()))
    # Land use classes
    lu_classes, lu_counts = np.unique(df['land_use_encoded'], return_counts=True)
    land_weights = dict(zip(lu_classes, (len(df) / (len(lu_classes) * lu_counts)).tolist()))
    return {'fertility_cls': fertility_weights, 'land_use': land_weights}

# ---------------- Train / Eval ----------------
def train_enhanced_model(model, train_gen, test_gen, class_weights):
    def huber_loss(y_true, y_pred):
        return tf.keras.losses.huber(y_true, y_pred, delta=1.0)

    losses = {
        'fertility_reg': huber_loss,
        'fertility_cls': 'sparse_categorical_crossentropy',
        'land_use': 'sparse_categorical_crossentropy'
    }
    loss_weights = {'fertility_reg': 0.4, 'fertility_cls': 0.3, 'land_use': 0.3}

    metrics = {
        'fertility_reg': ['mae', tf.keras.metrics.RootMeanSquaredError()],
        'fertility_cls': ['accuracy'],
        'land_use': ['accuracy']
    }

    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=max(2, PATIENCE//2), verbose=1, min_lr=1e-7),
        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', verbose=1)
    ]

    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    return history

def evaluate_model(model, test_gen, fertility_encoder, land_use_encoder):
    y_true_reg, y_true_fert, y_true_land = [], [], []
    y_pred_reg, y_pred_fert, y_pred_land = [], [], []
    for i in range(len(test_gen)):
        X, y, _ = test_gen[i]
        preds = model.predict(X, verbose=0)
        y_true_reg.append(y['fertility_reg'])
        y_true_fert.append(y['fertility_cls'])
        y_true_land.append(y['land_use'])
        y_pred_reg.append(preds[0])
        y_pred_fert.append(preds[1])
        y_pred_land.append(preds[2])

    y_true_reg = np.vstack(y_true_reg).ravel()
    y_true_fert = np.concatenate(y_true_fert)
    y_true_land = np.concatenate(y_true_land)
    y_pred_reg = np.vstack(y_pred_reg).ravel()
    y_pred_fert_cls = np.argmax(np.vstack(y_pred_fert), axis=1)
    y_pred_land_cls = np.argmax(np.vstack(y_pred_land), axis=1)

    print("Regression MAE:", mean_absolute_error(y_true_reg, y_pred_reg))
    print("Regression RMSE:", mean_squared_error(y_true_reg, y_pred_reg, squared=False))
    print("Regression R2:", r2_score(y_true_reg, y_pred_reg))

    print("\nFertility classification report:")
    print(classification_report(y_true_fert, y_pred_fert_cls, target_names=fertility_encoder.classes_))

    print("\nLand use classification report:")
    print(classification_report(y_true_land, y_pred_land_cls, target_names=land_use_encoder.classes_))

# ---------------- Main ----------------
if __name__ == "__main__":
    print("Starting pipeline...")
    df = load_and_clean_data(CSV_PATH)
    df = classify_soil_fertility(df)
    df = calculate_fertility_score(df, weights=SCORE_WEIGHTS)

    df, missing = match_images(df)
    print(f"Found images for {len(df)} samples, missing: {len(missing)}")

    # Encode targets
    fertility_encoder = LabelEncoder()
    df['fertility_class_encoded'] = fertility_encoder.fit_transform(df['Fertility_Class'].astype(str))
    if 'LU1_Desc' in df.columns:
        land_use_encoder = LabelEncoder()
        df['land_use_encoded'] = land_use_encoder.fit_transform(df['LU1_Desc'].astype(str))
    else:
        land_use_encoder = LabelEncoder()
        df['land_use_encoded'] = 0

    csv_features = ['Depth_code', 'pH_CaCl2', 'pH_H2O', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K',
                    'OC (20-30 cm)', 'CaCO3 (20-30 cm)', 'Ox_Al', 'Ox_Fe', 'TH_LAT', 'TH_LONG', 'Elev']
    csv_features = [f for f in csv_features if f in df.columns]

    train_gen, test_gen = prepare_datasets(df, csv_features, test_size=0.2)

    class_weights = compute_class_weights_from_df(df)
    print("Class weights:", class_weights)

    model = build_enhanced_model(num_fertility_classes=len(fertility_encoder.classes_), num_land_use_classes=max(1, len(land_use_encoder.classes_)), num_csv_features=len(csv_features))
    model.summary()

    # Train
    history = train_enhanced_model(model, train_gen, test_gen, class_weights)

    # Evaluate
    evaluate_model(model, test_gen, fertility_encoder, land_use_encoder)

    # Save
    model.save("soil_fertility_land_use_model.h5")
    joblib.dump(fertility_encoder, 'fertility_encoder.pkl')
    joblib.dump(land_use_encoder, 'land_use_encoder.pkl')
    joblib.dump(csv_features, 'csv_features.pkl')

    print("Done.")