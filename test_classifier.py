"""
Test classifier on random validation images from different disease classes.
"""

import os
import random
from utils import ImageInference

# Initialize classifier
print("Loading trained classifier...")
inference = ImageInference(
    classifier_path='./classifier_checkpoints/best_classifier.pt'
)

if inference.classifier is None:
    print("Error: Classifier not found!")
    exit()

# Test on validation data
val_dir = './data/PlantVillage/PlantVillage/val'
disease_classes = [
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Septoria_leaf_spot'
]

print("\nTesting classifier on random validation images:\n")

for disease_class in disease_classes:
    class_path = os.path.join(val_dir, disease_class)
    
    if not os.path.exists(class_path):
        print(f"✗ {disease_class}: Directory not found")
        continue
    
    # Get random image from class
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))]
    
    if not images:
        print(f"✗ {disease_class}: No images found")
        continue
    
    test_image = random.choice(images)
    image_path = os.path.join(class_path, test_image)
    
    print(f"Testing: {disease_class}")
    print(f"  Image: {test_image}")
    
    try:
        result = inference.classify_image(image_path)
        print(f"  Predicted: {result['class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print("  All probabilities:")
        for cls_name, prob in result['all_probabilities'].items():
            print(f"    - {cls_name}: {prob:.4f}")
        print()
    except Exception as e:
        print(f"  Error: {e}\n")

print("Testing complete!")
