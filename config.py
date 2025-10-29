# config.py
BASE_URL = "https://gisco-services.ec.europa.eu/lucas/photos/2018/"
DOWNLOAD_DIR = "newlucas_images"
CSV_PATH = "LUCASSOIL2018.csv"


# --- CONFIG ---

IMAGE_SHAPE = (256, 256, 3)  # Increased resolution
SUFFIXES = ['E', 'N', 'S', 'W', 'P']
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
PATIENCE = 10
DROPOUT_RATE = 0.5
L2_REG = 1e-4

# Fertility scoring weights
SCORE_WEIGHTS = {
    'ph': {'weight': 0.3, 'min': 4, 'max': 9},
    'oc': {'weight': 0.4, 'max': 300},
    'nutrients': {'weight': 0.2},
    'land_use': {'weight': 0.1}
}

