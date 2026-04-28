# PyTorch DCGAN for Tomato Leaf Disease Images

A complete implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) and disease classifier for tomato leaf disease images using the PlantVillage dataset.

## Project Structure

```
tomato-leaf-disease/
├── generator.py           # DCGAN Generator model
├── discriminator.py       # DCGAN Discriminator model
├── train_gan.py          # GAN training script
├── classifier.py         # Disease classifier based on ResNet50
├── utils.py              # Utility functions for inference
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Features

- **DCGAN Implementation**: State-of-the-art generative model for 128x128 images
- **Disease Classification**: ResNet50-based classifier for 3 tomato disease types
- **PlantVillage Dataset Support**: Direct integration with PlantVillage dataset
- **Training & Inference**: Complete training pipelines and inference utilities
- **Visualization**: Generated image samples and training curves

## Supported Tomato Diseases

1. **Early Blight** - Caused by Alternaria
2. **Late Blight** - Caused by Phytophthora
3. **Septoria Leaf Spot** - Caused by Septoria lycopersici

## Installation

### 1. Clone or download the project
```bash
cd tomato-leaf-disease
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Dataset Setup

### Download PlantVillage Dataset

1. Download the PlantVillage dataset from: https://www.kaggle.com/datasets/emmargerison/plantvillage-dataset

2. Extract the dataset to `./data/PlantVillage/`

3. Directory structure should be:
```
data/
└── PlantVillage/
    ├── Tomato___Early_blight/
    ├── Tomato___Late_blight/
    └── Tomato___Septoria_leaf_spot/
```

## Usage

### Training the DCGAN

```bash
python train_gan.py \
    --data_dir ./data/PlantVillage \
    --output_dir ./output \
    --checkpoint_dir ./checkpoints \
    --num_epochs 100 \
    --batch_size 64 \
    --lr 0.0002 \
    --image_size 128
```

**Arguments:**
- `--data_dir`: Path to PlantVillage dataset root
- `--output_dir`: Directory to save generated images
- `--checkpoint_dir`: Directory to save model checkpoints
- `--num_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 64)
- `--lr`: Learning rate (default: 0.0002)
- `--latent_dim`: Dimension of noise vector (default: 100)
- `--image_size`: Image size (default: 128)
- `--save_interval`: Interval for saving generated images (default: 10)
- `--log_interval`: Interval for logging progress (default: 50)

### Training the Classifier

```bash
python classifier.py \
    --data_dir ./data/PlantVillage \
    --checkpoint_dir ./classifier_checkpoints \
    --num_epochs 50 \
    --batch_size 32 \
    --lr 0.001
```

**Arguments:**
- `--data_dir`: Path to PlantVillage dataset root
- `--checkpoint_dir`: Directory to save model checkpoints
- `--num_epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--log_interval`: Interval for logging progress (default: 50)

### Using Trained Models for Inference

```python
from utils import ImageInference

# Initialize with trained models
inference = ImageInference(
    classifier_path='./classifier_checkpoints/best_classifier.pt',
    generator_path='./checkpoints/generator_final.pt'
)

# Classify an image
result = inference.classify_image('./tomato_leaf.jpg')
print(f"Disease: {result['class']}")
print(f"Confidence: {result['confidence']:.4f}")

# Generate synthetic images
synthetic_images = inference.generate_images(num_images=4)
```

## Model Architecture

### Generator
- Input: 100-dimensional noise vector
- Architecture: Linear → 4 Transposed Conv layers → Output (3×128×128)
- Activation: ReLU + BatchNorm (except input and output)
- Output activation: Tanh (normalized to [-1, 1])

### Discriminator
- Input: 3×128×128 image
- Architecture: 5 Conv layers → Linear layers → Binary classification
- Activation: LeakyReLU (0.2) + BatchNorm (except input)
- Output activation: Sigmoid

### Classifier
- Base: Pre-trained ResNet50 (ImageNet weights)
- Input size: 224×224
- Output: 3 disease classes
- Fine-tuned on PlantVillage tomato disease images

## Output Files

### Training Outputs
- `output/generated_epoch_*.png`: Generated image samples (4×4 grid)
- `output/training_losses.png`: Loss curves plot
- `checkpoints/generator_final.pt`: Trained generator weights
- `checkpoints/discriminator_final.pt`: Trained discriminator weights
- `classifier_checkpoints/best_classifier.pt`: Best classifier weights
- `classifier_checkpoints/confusion_matrix.png`: Test confusion matrix
- `classifier_checkpoints/training_curves.png`: Training/validation loss curves
- `classifier_checkpoints/metrics.json`: Performance metrics

## Performance Metrics

The classifier reports:
- **Accuracy**: Overall correctness
- **Precision**: False positive rate
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall

## Tips for Training

1. **GPU Acceleration**: Ensure CUDA is available for faster training
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

2. **Batch Size**: Use larger batches (64) for GAN training, smaller (32) for classifier

3. **Learning Rate**: DCGAN training is stable with lr=0.0002. Use AdamOptimizer with β₁=0.5, β₂=0.999

4. **Data Augmentation**: Classifier uses random horizontal flips and rotations

5. **Early Stopping**: Monitor validation loss to prevent overfitting

## Troubleshooting

### Out of Memory Error
- Reduce batch size: `--batch_size 32`
- Reduce image size (not recommended, breaks architecture)

### No Images Found
- Ensure dataset directory structure matches expected format
- Check file names don't have special characters
- Ensure images are valid (.jpg, .png, etc.)

### Poor Generation Quality
- Train for more epochs (100-200)
- Adjust learning rate
- Ensure sufficient training data (1000+ images per class)

## References

- DCGAN Paper: https://arxiv.org/abs/1511.06434
- PlantVillage Dataset: https://www.kaggle.com/datasets/emmargerison/plantvillage-dataset
- PyTorch Documentation: https://pytorch.org/docs/

## License

This project is provided for educational and research purposes.

## Future Improvements

- [ ] Add data augmentation techniques (mixup, cutmix)
- [ ] Implement Wasserstein GAN (WGAN)
- [ ] Add attention mechanisms
- [ ] Support for multi-GPU training
- [ ] Web interface for inference
- [ ] Real-time disease detection from camera feed
