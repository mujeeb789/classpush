# Enhanced Soil Fertility and Land Use Classification Model
# Combines image data with tabular soil data for improved predictions.
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.efficientnet import preprocess_input as ef_preprocess
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tqdm import tqdm
import joblib

# Configuration with safe defaults
try:
    from config import CSV_PATH, DOWNLOAD_DIR, SUFFIXES, SCORE_WEIGHTS, IMAGE_SHAPE
except ImportError:
    CSV_PATH = "soil_data.csv"
    DOWNLOAD_DIR = "./images"
    SUFFIXES = ["", "_sat"]
    SCORE_WEIGHTS = {
        'ph': {'min': 3.5, 'max': 9.0, 'weight': 0.25},
        'oc': {'max': 50.0, 'weight': 0.35},
        'nutrients': {'weight': 0.25},
        'land_use': {'weight': 0.15}
    }
    IMAGE_SHAPE = (224, 224, 3)

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
PATIENCE = 8
DROPOUT_RATE = 0.3

class SoilDataProcessor:
    """Handles all data loading, cleaning, and feature engineering"""
    
    def __init__(self, csv_path=CSV_PATH):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def load_and_clean_data(self):
        """Load and clean soil data with robust error handling"""
        try:
            df = pd.read_csv(self.csv_path, low_memory=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")
        
        # Define expected numeric columns
        numeric_cols = ['OC', 'pH_H2O', 'P', 'K', 'N', 'EC', 'CaCO3', 
                       'OC (20-30 cm)', 'CaCO3 (20-30 cm)', 'Ox_Al', 'Ox_Fe']
        
        # Clean and convert numeric columns
        for col in numeric_cols:
            if col in df.columns:
                df[col] = self._clean_numeric_column(df[col])
        
        # Handle categorical encoding
        if 'Depth' in df.columns:
            df['Depth_code'] = pd.Categorical(df['Depth']).codes
            
        if 'LU1_Desc' in df.columns:
            df['land_use_code'] = pd.Categorical(df['LU1_Desc']).codes
        else:
            df['LU1_Desc'] = 'Unknown'
            df['land_use_code'] = 0
            
        return df
    
    def _clean_numeric_column(self, series):
        """Clean a single numeric column"""
        # Convert to string and clean
        cleaned = series.astype(str).str.replace('< LOD', '', regex=False)
        cleaned = cleaned.str.replace('<', '', regex=False)
        cleaned = cleaned.str.replace('>', '', regex=False)
        
        # Convert to numeric, coercing errors to NaN
        cleaned = pd.to_numeric(cleaned, errors='coerce')
        
        # Only fill with median if we have some valid values
        if cleaned.notna().sum() > 0:
            median_val = cleaned.median()
            cleaned = cleaned.fillna(median_val)
        else:
            cleaned = cleaned.fillna(0.0)
            
        return cleaned
    
    def classify_soil_fertility(self, df):
        """Classify soil fertility based on OC and pH levels"""
        # Safeguard for missing columns
        if 'OC' not in df.columns:
            df['OC'] = 10.0  # Reasonable default
        if 'pH_H2O' not in df.columns:
            df['pH_H2O'] = 6.5  # Reasonable default
            
        # Organic Carbon classification
        conditions_oc = [
            (df['OC'] < 10),
            (df['OC'] >= 10) & (df['OC'] <= 20),
            (df['OC'] > 20)
        ]
        choices_oc = ['Low', 'Moderate', 'High']
        df['OC_Class'] = np.select(conditions_oc, choices_oc, default='Moderate')
        
        # pH classification
        conditions_ph = [
            (df['pH_H2O'] < 5.5),
            (df['pH_H2O'] >= 5.5) & (df['pH_H2O'] <= 6.8),
            (df['pH_H2O'] > 6.8)
        ]
        choices_ph = ['Acidic', 'Neutral', 'Alkaline']
        df['pH_Class'] = np.select(conditions_ph, choices_ph, default='Neutral')
        
        # Combined fertility classification
        conditions_fertility = [
            (df['OC'] < 10) | (df['pH_H2O'] < 5.5),
            ((df['OC'] >= 10) & (df['OC'] <= 20)) & (df['pH_H2O'] >= 5.5) & (df['pH_H2O'] <= 6.8),
            (df['OC'] > 20) | (df['pH_H2O'] > 6.8)
        ]
        choices_fertility = ['Low', 'Moderate', 'High']
        df['Fertility_Class'] = np.select(conditions_fertility, choices_fertility, default='Moderate')
        
        # Simple fertility score
        df['Fertility_Score_Simple'] = (
            np.where(df['OC'] < 10, 20,
                    np.where(df['OC'] <= 20, 40, 60)) +
            np.where(df['pH_H2O'] < 5.5, 10,
                    np.where(df['pH_H2O'] <= 6.8, 30, 40))
        )
        
        return df
    
    def calculate_fertility_score(self, df, weights=None):
        """Calculate advanced fertility score using multiple factors"""
        if weights is None:
            weights = SCORE_WEIGHTS
            
        # Ensure required columns exist
        required_cols = ['pH_H2O', 'OC', 'N', 'P', 'K', 'land_use_code']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0 if col != 'land_use_code' else 0
                
        # Normalize pH
        ph_range = weights['ph']['max'] - weights['ph']['min']
        df['ph_norm'] = np.clip(
            (df['pH_H2O'] - weights['ph']['min']) / max(ph_range, 0.1), 0, 1
        )
        
        # Normalize Organic Carbon
        df['oc_norm'] = np.clip(df['OC'] / max(weights['oc']['max'], 1.0), 0, 1)
        
        # Nutrient balance (log scale to handle large ranges)
        df['nutrient_balance'] = (
            np.log1p(df['N'].fillna(0)) + 
            np.log1p(df['P'].fillna(0)) + 
            np.log1p(df['K'].fillna(0))
        )
        
        # Normalize nutrient balance
        if df['nutrient_balance'].nunique() > 1:
            nut_min = df['nutrient_balance'].min()
            nut_max = df['nutrient_balance'].max()
            df['nutrient_balance_norm'] = (df['nutrient_balance'] - nut_min) / max((nut_max - nut_min), 0.1)
        else:
            df['nutrient_balance_norm'] = 0.5
            
        # Normalize land use
        if df['land_use_code'].nunique() > 1:
            land_use_max = df['land_use_code'].max()
            land_use_norm = df['land_use_code'] / max(land_use_max, 1)
        else:
            land_use_norm = 0.5
            
        # Combined score
        df['fertility_score_advanced'] = (
            weights['ph']['weight'] * df['ph_norm'] +
            weights['oc']['weight'] * df['oc_norm'] +
            weights['nutrients']['weight'] * df['nutrient_balance_norm'] +
            weights['land_use']['weight'] * land_use_norm
        ) * 100.0
        
        return df
    
    def prepare_features(self, df, csv_features):
        """Prepare and scale features for training"""
        # Select and fill missing features
        selected_features = []
        for feature in csv_features:
            if feature in df.columns:
                selected_features.append(feature)
            else:
                # Add missing feature with default value
                df[feature] = 0.0
                selected_features.append(feature)
                
        # Scale features
        feature_array = df[selected_features].values
        if not self.is_fitted:
            feature_array_scaled = self.scaler.fit_transform(feature_array)
            self.is_fitted = True
        else:
            feature_array_scaled = self.scaler.transform(feature_array)
            
        return selected_features, feature_array_scaled

class ImageManager:
    """Handles image loading and matching"""
    def __init__(self, download_dir=DOWNLOAD_DIR, suffixes=SUFFIXES):
        self.download_dir = download_dir
        self.suffixes = suffixes

        
    def match_images(self, df):
        """Match soil samples with their corresponding images"""
        image_rows = []
        missing_images = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Matching images"):
            pointid = str(row.get('POINTID', ''))
            nut0 = str(row.get('NUTS_0', 'UNKNOWN')).upper()
            
            image_path = self._find_image_path(pointid, nut0)
            
            if image_path and os.path.exists(image_path):
                row_copy = row.copy()
                row_copy['image_path'] = image_path
                image_rows.append(row_copy)
            else:
                missing_images.append(pointid)
                
        if not image_rows:
            print("Warning: No images found for any samples!")
            return pd.DataFrame(), missing_images
            
        return pd.DataFrame(image_rows), missing_images
    
    def _find_image_path(self, pointid, nut0):
        """Find image path using multiple directory structures"""
        if not pointid or pointid == 'nan':
            return None

        search_patterns = []
        for suffix in self.suffixes:
            search_patterns.extend([
                os.path.join(self.download_dir, nut0, pointid[:3], pointid[3:6], f"{pointid}{suffix}.jpg"),
                os.path.join(self.download_dir, nut0, pointid[:4], pointid[4:6], f"{pointid}{suffix}.jpg"),
                os.path.join(self.download_dir, nut0, pointid, f"{pointid}{suffix}.jpg"),
                os.path.join(self.download_dir, nut0, f"{pointid}{suffix}.jpg"),
                os.path.join(self.download_dir, f"{pointid}{suffix}.jpg") ])    
        # search_patterns = [
        #     os.path.join(self.download_dir, nut0, pointid[:3], pointid[3:6], f"{pointid}{suffix}.jpg"),
        #     os.path.join(self.download_dir, nut0, pointid[:4], pointid[4:6], f"{pointid}{suffix}.jpg"),
        #     os.path.join(self.download_dir, nut0, pointid, f"{pointid}{suffix}.jpg"),
        #     os.path.join(self.download_dir, nut0, f"{pointid}{suffix}.jpg"),
        #     os.path.join(self.download_dir, f"{pointid}{suffix}.jpg")
        #     for suffix in self.suffixes
        # ]
        
        # Flatten the list of patterns
        search_patterns = [item for sublist in search_patterns for item in sublist]
        
        for pattern in search_patterns:
            if os.path.exists(pattern):
                return pattern
                
        return None
    
    def load_image(self, image_path, target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1])):
        """Load and preprocess image with proper error handling"""
        if not image_path or not os.path.exists(image_path):
            return self._create_blank_image(target_size)
            
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                
                # Check if image is valid
                if min(img.size) < 10:
                    print(f"Warning: Image too small: {image_path}")
                    return self._create_blank_image(target_size)
                    
                img = img.resize(target_size, Image.LANCZOS)
                arr = np.array(img).astype(np.float32)
                
                # Apply EfficientNet preprocessing
                arr = ef_preprocess(arr)
                return arr
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self._create_blank_image(target_size)
    
    def _create_blank_image(self, target_size):
        """Create a blank image as fallback"""
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)

class EnhancedSoilGenerator(keras.utils.Sequence):
    """Data generator for soil images and tabular data"""
    
    def __init__(self, df, csv_features, batch_size=32, target_size=(224, 224), shuffle=True):
        self.df = df.reset_index(drop=True).copy()
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.csv_features = csv_features
        self.image_manager = ImageManager()
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.df))
        batch_df = self.df.iloc[start:end]
        
        X_img, X_csv, y_reg, y_fert, y_land = [], [], [], [], []
        
        for _, row in batch_df.iterrows():
            # Load image
            img = self.image_manager.load_image(
                row.get('image_path', ''), 
                target_size=self.target_size
            )
            X_img.append(img)
            
            # Load CSV features
            csv_vals = []
            for feature in self.csv_features:
                val = row.get(feature, 0.0)
                try:
                    val = float(val) if pd.notna(val) and str(val).strip() != '' else 0.0
                except (ValueError, TypeError):
                    val = 0.0
                csv_vals.append(val)
            X_csv.append(csv_vals)
            
            # Load targets
            y_reg.append(float(row.get('fertility_score_advanced', 0.0)))
            y_fert.append(int(row.get('fertility_class_encoded', 0)))
            y_land.append(int(row.get('land_use_encoded', 0)))
        
        # Convert to numpy arrays
        X_img = np.array(X_img, dtype=np.float32)
        X_csv = np.array(X_csv, dtype=np.float32)
        y_reg = np.array(y_reg, dtype=np.float32).reshape(-1, 1)
        y_fert = np.array(y_fert, dtype=np.int32)
        y_land = np.array(y_land, dtype=np.int32)
        
        return [X_img, X_csv], {
            "fertility_reg": y_reg,
            "fertility_cls": y_fert,
            "land_use": y_land,
        }
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)
            self.df = self.df.iloc[self.indices].reset_index(drop=True)

class SoilFertilityModel:
    """Main model class for soil fertility prediction"""
    
    def __init__(self):
        self.model = None
        self.processor = SoilDataProcessor()
        self.image_manager = ImageManager()
        
    def create_augmentation_layer(self):
        """Create image augmentation layer"""
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.1),
        ], name="augmentation")
    
    def build_model(self, num_fertility_classes, num_land_use_classes, num_csv_features):
        """Build the multi-task model architecture"""
        # Image input branch
        img_input = layers.Input(shape=IMAGE_SHAPE, name='image_input')
        augmented = self.create_augmentation_layer()(img_input)
        
        # CNN backbone (initially frozen)
        base_model = tf.keras.applications.EfficientNetB3(
            include_top=False, 
            weights='imagenet', 
            input_tensor=augmented, 
            pooling='avg'
        )
        base_model.trainable = False
        
        img_features = base_model.output
        img_features = layers.Dropout(DROPOUT_RATE)(img_features)
        
        # CSV input branch
        csv_input = layers.Input(shape=(num_csv_features,), name='csv_input')
        csv_features = layers.Dense(128, activation='relu')(csv_input)
        csv_features = layers.BatchNormalization()(csv_features)
        csv_features = layers.Dropout(DROPOUT_RATE)(csv_features)
        
        # Combined features
        combined = layers.Concatenate()([img_features, csv_features])
        combined = layers.Dense(256, activation='relu')(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(DROPOUT_RATE)(combined)
        
        # Output heads
        # Regression head for fertility score
        fertility_reg = layers.Dense(128, activation='relu')(combined)
        fertility_reg_output = layers.Dense(1, activation='linear', name='fertility_reg')(fertility_reg)
        
        # Classification head for fertility class
        fertility_cls = layers.Dense(128, activation='relu')(combined)
        fertility_cls_output = layers.Dense(
            num_fertility_classes, activation='softmax', name='fertility_cls'
        )(fertility_cls)
        
        # Classification head for land use
        land_use = layers.Dense(128, activation='relu')(combined)
        land_use_output = layers.Dense(
            num_land_use_classes, activation='softmax', name='land_use'
        )(land_use)
        
        # Create model
        model = models.Model(
            inputs=[img_input, csv_input], 
            outputs=[fertility_reg_output, fertility_cls_output, land_use_output]
        )
        
        return model
    
    def compute_class_weights(self, df):
        """Compute class weights for imbalanced datasets"""
        fertility_weights = {}
        land_use_weights = {}
        
        # Fertility class weights
        if 'fertility_class_encoded' in df.columns:
            classes, counts = np.unique(df['fertility_class_encoded'], return_counts=True)
            if len(classes) > 0:
                fertility_weights = dict(
                    zip(classes, len(df) / (len(classes) * counts))
                )
        
        # Land use class weights
        if 'land_use_encoded' in df.columns:
            lu_classes, lu_counts = np.unique(df['land_use_encoded'], return_counts=True)
            if len(lu_classes) > 0:
                land_use_weights = dict(
                    zip(lu_classes, len(df) / (len(lu_classes) * lu_counts))
                )
        
        return {
            'fertility_cls': fertility_weights,
            'land_use': land_use_weights
        }
    
    def prepare_datasets(self, df, csv_features, test_size=0.2):
        """Prepare training and validation datasets"""
        if 'fertility_class_encoded' not in df.columns:
            raise ValueError("DataFrame must contain 'fertility_class_encoded'")
            
        # Stratified split
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), 
            test_size=test_size, 
            random_state=42, 
            stratify=df['fertility_class_encoded']
        )
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        
        train_gen = EnhancedSoilGenerator(
            train_df, csv_features, batch_size=BATCH_SIZE, shuffle=True
        )
        test_gen = EnhancedSoilGenerator(
            test_df, csv_features, batch_size=BATCH_SIZE, shuffle=False
        )
        
        return train_gen, test_gen
    
    def train(self, train_gen, test_gen, class_weights=None):
        """Train the model"""
        # Define losses
        def huber_loss(y_true, y_pred):
            return tf.keras.losses.huber(y_true, y_pred, delta=1.0)
        
        losses = {
            'fertility_reg': huber_loss,
            'fertility_cls': 'sparse_categorical_crossentropy',
            'land_use': 'sparse_categorical_crossentropy'
        }
        
        loss_weights = {
            'fertility_reg': 0.4, 
            'fertility_cls': 0.3, 
            'land_use': 0.3
        }
        
        metrics = {
            'fertility_reg': ['mae', tf.keras.metrics.RootMeanSquaredError()],
            'fertility_cls': ['accuracy'],
            'land_use': ['accuracy']
        }
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
        self.model.compile(
            optimizer=optimizer, 
            loss=losses, 
            loss_weights=loss_weights, 
            metrics=metrics
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=PATIENCE, 
                restore_best_weights=True, 
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.2, 
                patience=max(2, PATIENCE//2), 
                verbose=1, 
                min_lr=1e-7
            ),
            ModelCheckpoint(
                'best_model.h5', 
                save_best_only=True, 
                monitor='val_loss', 
                verbose=1
            )
        ]
        
        # Prepare class weights
        cw = {}
        if class_weights:
            if 'fertility_cls' in class_weights and class_weights['fertility_cls']:
                cw['fertility_cls'] = class_weights['fertility_cls']
            if 'land_use' in class_weights and class_weights['land_use']:
                cw['land_use'] = class_weights['land_use']
        
        # Train model
        history = self.model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
            class_weight=cw if cw else None,
            workers=4,
            use_multiprocessing=False  # Set to True if your system supports it
        )
        
        return history
    
    def evaluate(self, test_gen, fertility_encoder, land_use_encoder):
        """Evaluate model performance"""
        y_true_reg, y_true_fert, y_true_land = [], [], []
        y_pred_reg, y_pred_fert, y_pred_land = [], [], []
        
        for i in range(len(test_gen)):
            X, y = test_gen[i]
            preds = self.model.predict(X, verbose=0)
            
            y_true_reg.append(y['fertility_reg'])
            y_true_fert.append(y['fertility_cls'])
            y_true_land.append(y['land_use'])
            
            y_pred_reg.append(preds[0])
            y_pred_fert.append(preds[1])
            y_pred_land.append(preds[2])
        
        # Concatenate results
        y_true_reg = np.vstack(y_true_reg).ravel()
        y_true_fert = np.concatenate(y_true_fert)
        y_true_land = np.concatenate(y_true_land)
        
        y_pred_reg = np.vstack(y_pred_reg).ravel()
        y_pred_fert_cls = np.argmax(np.vstack(y_pred_fert), axis=1)
        y_pred_land_cls = np.argmax(np.vstack(y_pred_land), axis=1)
        
        # Print results
        print("=== Regression Results ===")
        print(f"MAE: {mean_absolute_error(y_true_reg, y_pred_reg):.4f}")
        print(f"RMSE: {mean_squared_error(y_true_reg, y_pred_reg, squared=False):.4f}")
        print(f"R²: {r2_score(y_true_reg, y_pred_reg):.4f}")
        
        print("\n=== Fertility Classification ===")
        print(classification_report(
            y_true_fert, y_pred_fert_cls, 
            target_names=fertility_encoder.classes_
        ))
        
        print("\n=== Land Use Classification ===")
        print(classification_report(
            y_true_land, y_pred_land_cls,
            target_names=land_use_encoder.classes_
        ))

def main():
    """Main execution function"""
    print("Starting Enhanced Soil Fertility Pipeline...")
    
    # Initialize components
    soil_model = SoilFertilityModel()
    
    # Load and process data
    print("1. Loading and cleaning data...")
    df = soil_model.processor.load_and_clean_data()
    
    print("2. Classifying soil fertility...")
    df = soil_model.processor.classify_soil_fertility(df)
    
    print("3. Calculating fertility scores...")
    df = soil_model.processor.calculate_fertility_score(df)
    
    print("4. Matching images...")
    df_with_images, missing = soil_model.image_manager.match_images(df)
    
    print(f"Found images for {len(df_with_images)} samples")
    if missing:
        print(f"Missing images for {len(missing)} samples")
    
    if len(df_with_images) == 0:
        print("Error: No images found. Cannot proceed with training.")
        return
    
    # Encode targets
    print("5. Encoding targets...")
    fertility_encoder = LabelEncoder()
    land_use_encoder = LabelEncoder()
    
    df_with_images['fertility_class_encoded'] = fertility_encoder.fit_transform(
        df_with_images['Fertility_Class'].astype(str)
    )
    df_with_images['land_use_encoded'] = land_use_encoder.fit_transform(
        df_with_images['LU1_Desc'].astype(str)
    )
    
    # Prepare features
    csv_features = [
        'Depth_code', 'pH_H2O', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K',
        'OC (20-30 cm)', 'CaCO3 (20-30 cm)', 'Ox_Al', 'Ox_Fe', 
        'TH_LAT', 'TH_LONG', 'Elev'
    ]
    
    # Compute class weights
    print("6. Computing class weights...")
    class_weights = soil_model.compute_class_weights(df_with_images)
    print(f"Class weights: {class_weights}")
    
    # Prepare datasets
    print("7. Preparing datasets...")
    train_gen, test_gen = soil_model.prepare_datasets(df_with_images, csv_features)
    
    # Build model
    print("8. Building model...")
    soil_model.model = soil_model.build_model(
        num_fertility_classes=len(fertility_encoder.classes_),
        num_land_use_classes=len(land_use_encoder.classes_),
        num_csv_features=len(csv_features)
    )
    
    soil_model.model.summary()
    
    # Train model
    print("9. Training model...")
    history = soil_model.train(train_gen, test_gen, class_weights)
    
    # Evaluate model
    print("10. Evaluating model...")
    soil_model.evaluate(test_gen, fertility_encoder, land_use_encoder)
    
    # Save model and artifacts
    print("11. Saving model and artifacts...")
    soil_model.model.save("soil_fertility_land_use_model.h5")
    joblib.dump(fertility_encoder, 'fertility_encoder.pkl')
    joblib.dump(land_use_encoder, 'land_use_encoder.pkl')
    joblib.dump(csv_features, 'csv_features.pkl')
    joblib.dump(soil_model.processor.scaler, 'feature_scaler.pkl')
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()