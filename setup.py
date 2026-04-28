"""
Setup script to prepare the environment and dataset.
Run: python setup.py
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"Error: Python 3.8+ required, but found {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("✗ CUDA not available. Will use CPU (slower)")
    except ImportError:
        print("✗ PyTorch not installed yet")


def create_directories():
    """Create necessary project directories."""
    dirs = [
        './data',
        './data/PlantVillage',
        './output',
        './checkpoints',
        './classifier_checkpoints',
        './inference_output'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified directory: {dir_path}")


def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'seaborn': 'Seaborn'
    }
    
    missing_packages = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def download_dataset():
    """Provide instructions for downloading dataset."""
    print("\n" + "="*60)
    print("DATASET SETUP INSTRUCTIONS")
    print("="*60)
    
    print("""
The project requires the PlantVillage dataset. Follow these steps:

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/emmargerison/plantvillage-dataset

2. Extract the downloaded file to the './data/PlantVillage/' directory

3. Expected directory structure:
   data/
   └── PlantVillage/
       ├── Tomato___Early_blight/
       │   ├── image1.jpg
       │   ├── image2.jpg
       │   └── ...
       ├── Tomato___Late_blight/
       │   └── ...
       └── Tomato___Septoria_leaf_spot/
           └── ...

Note: You can also use a subset of the dataset for testing/development.
      The script will work with any available tomato disease classes.
    """)


def verify_dataset():
    """Check if dataset is present and valid."""
    base_dir = './data/PlantVillage'
    
    if not os.path.exists(base_dir):
        print(f"✗ Dataset directory not found: {base_dir}")
        return False
    
    disease_classes = [
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Septoria_leaf_spot'
    ]
    
    found_classes = []
    for disease_class in disease_classes:
        class_dir = os.path.join(base_dir, disease_class)
        if os.path.exists(class_dir):
            num_images = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))])
            print(f"✓ Found {disease_class}: {num_images} images")
            found_classes.append(disease_class)
        else:
            print(f"✗ Not found: {disease_class}")
    
    if not found_classes:
        print("\n✗ No disease classes found in dataset directory")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Setup script for DCGAN project')
    parser.add_argument('--skip_dataset', action='store_true', help='Skip dataset verification')
    args = parser.parse_args()
    
    print("="*60)
    print("DCGAN Setup - Tomato Leaf Disease Detection")
    print("="*60 + "\n")
    
    # Check Python version
    print("1. Checking Python version...")
    check_python_version()
    
    # Check CUDA
    print("\n2. Checking CUDA availability...")
    check_cuda()
    
    # Create directories
    print("\n3. Creating project directories...")
    create_directories()
    
    # Check dependencies
    print("\n4. Checking dependencies...")
    if check_dependencies():
        print("\n✓ All dependencies are installed!")
    else:
        print("\n⚠ Some dependencies are missing. Install with:")
        print("pip install -r requirements.txt")
    
    # Verify/download dataset
    if not args.skip_dataset:
        print("\n5. Verifying dataset...")
        if verify_dataset():
            print("\n✓ Dataset verification complete!")
        else:
            download_dataset()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Ensure dataset is in ./data/PlantVillage/")
    print("2. Train GAN:       python train_gan.py")
    print("3. Train Classifier: python classifier.py")
    print("4. Run Inference:   python inference.py --mode classify --image_path <image>")
    print("\nFor more information, see README.md")


if __name__ == '__main__':
    main()
