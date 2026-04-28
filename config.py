"""
Configuration file for DCGAN project.
Defines constants and hyperparameters for training and inference.
"""

import torch

# ============ Dataset Configuration ============
DATA_DIR = './data/PlantVillage'
IMAGE_SIZE = 128
NUM_CHANNELS = 3

DISEASE_CLASSES = [
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Septoria_leaf_spot'
]

# ============ Model Configuration ============
# Generator
LATENT_DIM = 100
GENERATOR_CHANNELS = 32

# Discriminator
DISCRIMINATOR_CHANNELS = 32

# Classifier
CLASSIFIER_INPUT_SIZE = 224
NUM_DISEASE_CLASSES = len(DISEASE_CLASSES)

# ============ Training Configuration ============
# DCGAN
GAN_EPOCHS = 100
GAN_BATCH_SIZE = 64
GAN_LR = 0.0002
GAN_BETA1 = 0.5
GAN_BETA2 = 0.999
GAN_NUM_WORKERS = 4

# Classifier
CLASSIFIER_EPOCHS = 50
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LR = 0.001
CLASSIFIER_NUM_WORKERS = 4
CLASSIFIER_TRAIN_SPLIT = 0.8
CLASSIFIER_VAL_SPLIT = 0.1
CLASSIFIER_TEST_SPLIT = 0.1

# ============ Optimization Configuration ============
OPTIMIZER = 'adam'
SCHEDULER = 'reduce_on_plateau'

# ============ Directories ============
OUTPUT_DIR = './output'
CHECKPOINT_DIR_GAN = './checkpoints'
CHECKPOINT_DIR_CLASSIFIER = './classifier_checkpoints'

# ============ Logging ============
LOG_INTERVAL = 50
SAVE_INTERVAL = 10
SEED = 42

# ============ Device Configuration ============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ Normalization ============
# ImageNet normalization for classifier
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# GAN normalization (to [-1, 1])
GAN_MEAN = [0.5, 0.5, 0.5]
GAN_STD = [0.5, 0.5, 0.5]

# ============ Data Augmentation ============
# Classifier augmentation
CLASSIFIER_AUGMENTATION = {
    'horizontal_flip': True,
    'rotation': 20,
    'brightness': 0.1,
    'contrast': 0.1
}

# GAN data augmentation (minimal for training)
GAN_AUGMENTATION = {
    'horizontal_flip': False,
    'rotation': 0
}

# ============ Loss Configuration ============
GAN_LOSS_FUNCTION = 'bce'  # Binary Cross Entropy
CLASSIFIER_LOSS_FUNCTION = 'cross_entropy'  # Cross Entropy

# ============ Inference Configuration ============
INFERENCE_BATCH_SIZE = 4
CONFIDENCE_THRESHOLD = 0.7

# ============ Visualization Configuration ============
GRID_SIZE = (4, 4)  # Grid size for saving generated images
DPI = 100
