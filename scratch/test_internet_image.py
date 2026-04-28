import os
import urllib.request
from PIL import Image, ImageFilter
import numpy as np
import sys

# Add parent directory to path so we can import utils
sys.path.append(os.path.abspath('.'))
from utils import ImageInference

def add_noise(image_path, out_path):
    img = Image.open(image_path).convert('RGB')
    # Add blur
    img_blur = img.filter(ImageFilter.GaussianBlur(radius=3.0))
    # Add noise
    img_array = np.array(img_blur, dtype=np.float32)
    noise = np.random.normal(0, 30, img_array.shape)
    img_noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    Image.fromarray(img_noisy).save(out_path)

if __name__ == "__main__":
    print("Loading model...")
    inference = ImageInference(classifier_path='./classifier_checkpoints/best_classifier.pt')
    
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/e/ec/Tomato_early_blight_leaf.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Tomato_leaf.jpg/800px-Tomato_leaf.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Tomato_Leaves.jpg/800px-Tomato_Leaves.jpg",
        "https://images.unsplash.com/photo-1591857177580-dc82b9ac4e1e"
    ]
    img_path = "scratch/internet_image.jpg"
    noisy_path = "scratch/internet_noisy.jpg"
    
    downloaded = False
    for url in urls:
        try:
            print(f"Trying to download {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(img_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            downloaded = True
            break
        except Exception as e:
            print(f"Failed: {e}")
            
    if not downloaded:
        print("Failed to download any images. Creating a fake random noise image.")
        # Create dummy image
        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(dummy).save(img_path)
    
    print(f"Testing {img_path}")
    
    res1 = inference.classify_image(img_path)
    print(f"\n--- ORIGINAL IMAGE ---")
    print(f"Prediction: {res1['class']}")
    print(f"Confidence: {res1['confidence']:.4f}")
    
    add_noise(img_path, noisy_path)
    res2 = inference.classify_image(noisy_path)
    print(f"\n--- BLURRED / NOISY IMAGE ---")
    print(f"Prediction: {res2['class']}")
    print(f"Confidence: {res2['confidence']:.4f}")

