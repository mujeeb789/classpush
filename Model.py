import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, 
                           accuracy_score, f1_score, confusion_matrix, classification_report)
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import CSV_PATH, DOWNLOAD_DIR, SUFFIXES, SCORE_WEIGHTS, IMAGE_SHAPE, EPOCHS, BATCH_SIZE, LEARNING_RATE, PATIENCE, DROPOUT_RATE, L2_REG
from DataPrep import load_and_clean_data, classify_soil_fertility


# --- DATA LOADING ---
def load_data():
    """Load and preprocess data using your existing pipeline"""
    from DataPrep import full_pipeline
    df = full_pipeline(CSV_PATH)
    
    # Ensure we have both fertility and land use data
    if 'Fertility_Class' not in df.columns:
        df = classify_soil_fertility(df)
    if 'LU1_Desc' not in df.columns:
        raise ValueError("Land use data missing")
    
    # Encode categorical targets
    fertility_encoder = LabelEncoder()
    land_use_encoder = LabelEncoder()
    
    df['fertility_class_encoded'] = fertility_encoder.fit_transform(df['Fertility_Class'])
    df['land_use_encoded'] = land_use_encoder.fit_transform(df['LU1_Desc'])
    
    return df, fertility_encoder, land_use_encoder

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow loading of truncated images

def load_image(image_path, target_size=IMAGE_SHAPE[:2]):
    """Robust image loading with error handling"""
    try:
        # First verify the image can be opened
        with Image.open(image_path) as img:
            img.verify()  # Verify it's a valid image
            
        # Now open for real processing
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img) / 255.0
            
            # Validate image content
            if img_array.mean() < 0.1 or img_array.mean() > 0.9:
                print(f"Warning: Suspicious image {image_path} with mean value {img_array.mean()}")
                return None
                
            # Normalize with ImageNet stats
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img_array = (img_array - mean) / std
            
            return img_array
            
    except (IOError, OSError, Image.DecompressionBombError) as e:
        print(f"Error loading {image_path}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error with {image_path}: {str(e)}")
        return None



# --- MODEL ARCHITECTURE ---
def build_multi_task_model(num_fertility_classes, num_land_use_classes):
    """Multi-task CNN model predicting both fertility and land use"""
    # Base CNN for feature extraction
    img_input = layers.Input(shape=IMAGE_SHAPE, name='image_input')
    
    # Initial convolution blocks
    x = layers.Conv2D(64, (7,7), strides=2, activation='relu', 
                      kernel_regularizer=regularizers.l2(L2_REG),
                      padding='same')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3,3), strides=2)(x)
    
    # Residual blocks
    for filters in [64, 128, 256]:
        x = residual_block(x, filters)
    
    # Global features
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Task-specific branches
    # Fertility prediction branch (regression + classification)
    fertility_reg = layers.Dense(64, activation='relu')(x)
    fertility_reg = layers.Dropout(DROPOUT_RATE)(fertility_reg)
    fertility_reg_output = layers.Dense(1, activation='linear', name='fertility_reg')(fertility_reg)
    
    fertility_cls = layers.Dense(64, activation='relu')(x)
    fertility_cls = layers.Dropout(DROPOUT_RATE)(fertility_cls)
    fertility_cls_output = layers.Dense(num_fertility_classes, activation='softmax', 
                                       name='fertility_cls')(fertility_cls)
    
    # Land use prediction branch
    land_use = layers.Dense(128, activation='relu')(x)
    land_use = layers.Dropout(DROPOUT_RATE)(land_use)
    land_use_output = layers.Dense(num_land_use_classes, activation='softmax', 
                                  name='land_use')(land_use)
    
    model = models.Model(
        inputs=img_input,
        outputs=[fertility_reg_output, fertility_cls_output, land_use_output]
    )
    
    return model

def residual_block(x, filters, kernel_size=3):
    """Basic residual block for feature extraction"""
    shortcut = x
    
    x = layers.Conv2D(filters, kernel_size, padding='same', 
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), padding='same')(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x


# --- MODIFIED DATA LOADING ---
class SoilImageGenerator(tf.keras.utils.Sequence):
    """Custom data generator to avoid memory issues"""
    def __init__(self, df, batch_size=32, target_size=(256, 256), shuffle=True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        
        X = np.empty((len(batch_df), *self.target_size, 3), dtype=np.float32)
        y_reg = np.empty((len(batch_df), 1), dtype=np.float32)
        y_fert = np.empty((len(batch_df),), dtype=np.int32)
        y_land = np.empty((len(batch_df),), dtype=np.int32)
        
        for i, (_, row) in enumerate(batch_df.iterrows()):
            img = load_image(row['image_path'], self.target_size)
            if img is None:
                # Fallback to black image if loading fails
                img = np.zeros((*self.target_size, 3), dtype=np.float32)
            
            X[i] = img
            y_reg[i] = row['fertility_score_advanced']
            y_fert[i] = row['fertility_class_encoded']
            y_land[i] = row['land_use_encoded']
            
        return X, {'fertility_reg': y_reg, 'fertility_cls': y_fert, 'land_use': y_land}
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

def prepare_datasets(df, test_size=0.2):
    """Modified to return generators instead of full datasets"""
    # Filter out rows with invalid images first
    valid_df = df.copy()
    
    # Create train-test split indices
    train_idx, test_idx = train_test_split(
        np.arange(len(valid_df)),
        test_size=test_size,
        random_state=42,
        stratify=valid_df['fertility_class_encoded']
    )
    
    # Create generators
    train_gen = SoilImageGenerator(valid_df.iloc[train_idx], batch_size=BATCH_SIZE)
    test_gen = SoilImageGenerator(valid_df.iloc[test_idx], batch_size=BATCH_SIZE, shuffle=False)
    
    return train_gen, test_gen


def train_model(model, train_gen, test_gen):
    """Modified to work with generators"""
    # Loss weights and metrics remain the same
    loss_weights = {
        'fertility_reg': 0.4,
        'fertility_cls': 0.3,
        'land_use': 0.3
    }
    
    loss = {
        'fertility_reg': 'mse',
        'fertility_cls': 'sparse_categorical_crossentropy',
        'land_use': 'sparse_categorical_crossentropy'
    }
    
    metrics = {
        'fertility_reg': ['mae', tf.keras.metrics.RootMeanSquaredError()],
        'fertility_cls': ['accuracy'],
        'land_use': ['accuracy']
    }
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE//2)
    ]
    
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, test_gen, fertility_encoder, land_use_encoder):
    """Evaluate using the generator"""
    # Collect all predictions and true values
    y_true_reg, y_true_fert, y_true_land = [], [], []
    y_pred_reg, y_pred_fert, y_pred_land = [], [], []
    
    for i in range(len(test_gen)):
        X, y_true = test_gen[i]
        y_pred = model.predict(X, verbose=0)
        
        y_true_reg.append(y_true['fertility_reg'])
        y_true_fert.append(y_true['fertility_cls'])
        y_true_land.append(y_true['land_use'])
        
        y_pred_reg.append(y_pred[0])
        y_pred_fert.append(y_pred[1])
        y_pred_land.append(y_pred[2])
    
    # Concatenate batches
    y_true_reg = np.concatenate(y_true_reg)
    y_true_fert = np.concatenate(y_true_fert)
    y_true_land = np.concatenate(y_true_land)
    
    y_pred_reg = np.concatenate(y_pred_reg)
    y_pred_fert = np.concatenate(y_pred_fert)
    y_pred_land = np.concatenate(y_pred_land)
    
    # Convert class predictions
    y_pred_fert_cls = np.argmax(y_pred_fert, axis=1)
    y_pred_land_cls = np.argmax(y_pred_land, axis=1)
    
    # Rest of your evaluation code remains the same...
    print("\nFertility Score Regression Evaluation:")
    print(f"MAE: {mean_absolute_error(y_true_reg, y_pred_reg):.4f}")
    print(f"RMSE: {mean_squared_error(y_true_reg, y_pred_reg, squared=False):.4f}")
    print(f"R²: {r2_score(y_true_reg, y_pred_reg):.4f}")
    
    print("\nFertility Classification Evaluation:")
    print(classification_report(y_true_fert, y_pred_fert_cls, 
                               target_names=fertility_encoder.classes_))
    
    print("\nLand Use Classification Evaluation:")
    print(classification_report(y_true_land, y_pred_land_cls,
                               target_names=land_use_encoder.classes_))
    
    plot_confusion_matrix(y_true_fert, y_pred_fert_cls, fertility_encoder.classes_, 
                         "Fertility Class Confusion Matrix")
    plot_confusion_matrix(y_true_land, y_pred_land_cls, land_use_encoder.classes_,
                         "Land Use Confusion Matrix")

def plot_confusion_matrix(y_true, y_pred, classes, title):
    """Plot a confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load and prepare data
    df, fertility_encoder, land_use_encoder = load_data()
    
    # Prepare generators
    train_gen, test_gen = prepare_datasets(df)
    
    # Build model
    model = build_multi_task_model(
        num_fertility_classes=len(fertility_encoder.classes_),
        num_land_use_classes=len(land_use_encoder.classes_)
    )
    
    # Train
    history = train_model(model, train_gen, test_gen)
    
    # Evaluate
    evaluate_model(model, test_gen, fertility_encoder, land_use_encoder)
    
    # Save model
    model.save("soil_fertility_land_use_model.h5")
    print("Model saved successfully.")