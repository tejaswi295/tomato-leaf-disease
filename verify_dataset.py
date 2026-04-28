from train_gan import TomatoDiseaseDataset
import os

print("Verifying dataset...")
ds = TomatoDiseaseDataset('./data')
print(f"✓ Found {len(ds)} images")

if len(ds.images) > 0:
    print(f"✓ Sample image: {ds.images[0]}")
    print(f"Files in dataset:")
    for i, label in enumerate(set(ds.labels)):
        count = ds.labels.count(label)
        print(f"  {label}: {count} images")
else:
    print("✗ No images found!")
    print(f"Expected path: ./data/PlantVillage/train/")
    print(f"Actual structure:")
    for root, dirs, files in os.walk('./data'):
        level = root.replace('./data', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level < 3:  # Limit depth
            for d in dirs[:5]:  # Show first 5 dirs
                print(f'{indent}  {d}/')
