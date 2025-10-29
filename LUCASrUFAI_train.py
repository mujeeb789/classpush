# =============================================================================
# LUCAS Land Cover Classifier: Train & Test Multiple Models
# Models: EfficientNetB0, ResNet50V2, Vision Transformer (ViT)
# Author: You + AI Assistant
# Dependencies: tensorflow>=2.10, pandas, numpy, matplotlib, pillow
# =============================================================================

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
from config import CSV_PATH
PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PHOTOS_DIR = os.path.join(DATA_DIR, "Photo")
CLEAN_META_PATH = os.path.join(PROJECT_ROOT, CSV_PATH)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Create models directory
os.makedirs(MODELS_DIR, exist_ok=True)

# Model constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 8

# LC0 class names (must match exploration script)
CLASS_NAMES = [
    'Artificial surfaces',
    'Agricultural areas',
    'Forest and semi-natural areas',
    'Wetlands',
    'Water bodies',
    'Permanent crops',
    'Open spaces with little or no vegetation',
    'Inland wetlands (supplementary)'
]

# Map LC0 codes to indices (A=0, B=1, ..., H=7)
LC0_TO_IDX = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_and_preprocess_image(path, label=None):
    """Load and preprocess a single image."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return (img, label) if label is not None else img

def build_dataset(df, shuffle=True, augment=False):
    """Build tf.data.Dataset from DataFrame."""
    paths = df['photo_path'].values
    labels = df['LC0'].map(LC0_TO_IDX).values

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        lambda x, y: load_and_preprocess_image(x, y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment:
        dataset = dataset.map(
            lambda x, y: (tf.image.random_flip_left_right(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

def get_model(model_name):
    """Return compiled model based on name."""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    if model_name == "efficientnet":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights='imagenet', input_tensor=inputs
        )
    elif model_name == "resnet":
        base = tf.keras.applications.ResNet50V2(
            include_top=False, weights='imagenet', input_tensor=inputs
        )
    elif model_name == "vit":
        # Simple ViT implementation (small)
        base = build_vit_model()
        return base  # ViT is built differently
    else:
        raise ValueError("Unknown model")

    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_vit_model():
    """Build a small Vision Transformer."""
    # Patch settings
    patch_size = 16
    num_patches = (IMG_SIZE // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [projection_dim * 2, projection_dim]
    transformer_layers = 4
    mlp_head_units = [2048, 1024]

    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # Augment & patch
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Transformer blocks
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    # Final MLP
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.2)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.2)
    logits = layers.Dense(NUM_CLASSES, activation='softmax')(features)

    model = keras.Model(inputs=inputs, outputs=logits)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ViT Helper Layers
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(model_name):
    print(f"\n🚀 Training {model_name.upper()}...")

    # Load data
    if not os.path.exists(CLEAN_META_PATH):
        print(f"❌ Clean metadata not found at {CLEAN_META_PATH}")
        print("👉 Please run the exploration script first!")
        return

    df = pd.read_csv(CLEAN_META_PATH)
    df['photo_path'] = df['photo_path'].apply(lambda x: x if os.path.exists(x) else None)
    df = df.dropna(subset=['photo_path'])

    # Split
    train_size = int(0.8 * len(df))
    df_train = df.sample(frac=1, random_state=42).iloc[:train_size]
    df_val = df.sample(frac=1, random_state=42).iloc[train_size:]

    train_ds = build_dataset(df_train, augment=True)
    val_ds = build_dataset(df_val, shuffle=False)

    # Model
    model = get_model(model_name)
    model_path = os.path.join(MODELS_DIR, f"model_{model_name}")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
    ]

    # Initial training (frozen base)
    print("Phase 1: Training head (frozen base)...")
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks, verbose=1)

    # Fine-tuning
    if model_name != "vit":  # ViT is trained end-to-end
        print("Phase 2: Fine-tuning...")
        model.layers[1].trainable = True  # Unfreeze base
        model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        history2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS-5, callbacks=callbacks, verbose=1)
        # Combine history
        history = {k: history1.history[k] + history2.history[k] for k in history1.history}
    else:
        history = history1.history

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(PROJECT_ROOT, f"training_{model_name}.png"), dpi=150)
    plt.show()

    print(f"✅ Model saved to: {model_path}")

# =============================================================================
# TESTING FUNCTION
# =============================================================================

def test_model(model_name):
    print(f"\n🧪 Testing {model_name.upper()}...")

    model_path = os.path.join(MODELS_DIR, f"model_{model_name}")
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("👉 Train the model first!")
        return

    # Load model
    if model_name == "vit":
        model = build_vit_model()
        model.load_weights(os.path.join(model_path, "variables", "variables"))
    else:
        model = keras.models.load_model(model_path)

    # Load validation data
    df = pd.read_csv(CLEAN_META_PATH)
    df = df.sample(frac=1, random_state=42).iloc[:1000]  # Sample for speed
    df['photo_path'] = df['photo_path'].apply(lambda x: x if os.path.exists(x) else None)
    df = df.dropna(subset=['photo_path'])
    test_ds = build_dataset(df, shuffle=False)

    # Evaluate
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"📊 Validation Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Predict a single image
    sample = df.sample(1).iloc[0]
    img_path = sample['photo_path']
    true_label = CLASS_NAMES[LC0_TO_IDX[sample['LC0']]]

    img = load_and_preprocess_image(img_path)
    pred = model.predict(tf.expand_dims(img, 0), verbose=0)
    pred_class = CLASS_NAMES[np.argmax(pred)]
    confidence = np.max(pred)

    print(f"\n🔍 Single Image Prediction:")
    print(f"   Image: {img_path}")
    print(f"   True:  {true_label}")
    print(f"   Pred:  {pred_class} (confidence: {confidence:.2%})")

    # Show image
    img_np = tf.keras.utils.array_to_img(img)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    plt.title(f"True: {true_label}\nPredicted: {pred_class} ({confidence:.1%})")
    plt.axis('off')
    plt.show()

# =============================================================================
# MAIN INTERACTIVE LOOP
# =============================================================================

def main():
    print("="*60)
    print("🌍 LUCAS Land Cover Classification - Train & Test")
    print("="*60)

    # Mode selection
    print("\nChoose mode:")
    print("1. Train a model")
    print("2. Test a model")
    mode = input("Enter choice (1/2): ").strip()

    if mode not in ['1', '2']:
        print("❌ Invalid choice.")
        return

    # Model selection
    print("\nChoose model:")
    print("1. EfficientNetB0")
    print("2. ResNet50V2")
    print("3. Vision Transformer (ViT)")
    model_choice = input("Enter choice (1/2/3): ").strip()

    model_map = {'1': 'efficientnet', '2': 'resnet', '3': 'vit'}
    if model_choice not in model_map:
        print("❌ Invalid model choice.")
        return

    model_name = model_map[model_choice]

    if mode == '1':
        train_model(model_name)
    else:
        test_model(model_name)

if __name__ == "__main__":
    main()
