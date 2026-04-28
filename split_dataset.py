import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def set_seed(seed=42):
    random.seed(seed)

def group_by_prefix(filenames):
    """
    Groups filenames by prefix to ensure near-duplicates stay in the same split.
    Assuming typical PlantVillage convention uuid___RS_... OR augmented names like image_1.jpg, image_2.jpg.
    If there is a UUID, we group by it. If it's something else, group by first part before '_'.
    """
    groups = defaultdict(list)
    for f in filenames:
        # Many datasets use 'uuid___class' or 'base_name_aug1'
        # We try splitting by '___' or '_' to find a common stem.
        stem = Path(f).stem
        if '___' in stem:
            prefix = stem.split('___')[0]
        elif '_' in stem:
            # If it's like leaf_1, leaf_2, group by 'leaf'
            prefix = stem.split('_')[0]
        else:
            prefix = stem
        
        # If the prefix is too short, use the whole stem
        if len(prefix) < 3:
            prefix = stem
            
        groups[prefix].append(f)
    return groups

def main():
    set_seed(42)
    
    source_dir = Path('data/PlantVillage')
    dest_dir = Path('dataset')
    
    # Target tomato classes
    target_classes = [
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    
    # Ratios
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Create structure
    for split in ['train', 'val', 'test']:
        for cls in target_classes:
            (dest_dir / split / cls).mkdir(parents=True, exist_ok=True)
            
    # Find all source images
    print("Scanning source directory...")
    # Gather all images for each class from anywhere inside the source_dir
    class_images = defaultdict(list)
    for root, _, files in os.walk(source_dir):
        root_path = Path(root)
        if root_path.name in target_classes:
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Store absolute or relative path
                    class_images[root_path.name].append(str(root_path / f))
                    
    total_copied = 0
    
    for cls in target_classes:
        images = class_images.get(cls, [])
        if not images:
            print(f"Warning: No images found for {cls}")
            continue
            
        # Group to prevent leakage
        groups = group_by_prefix(images)
        group_keys = list(groups.keys())
        
        # Shuffle group keys safely
        random.shuffle(group_keys)
        
        train_count = 0
        val_count = 0
        test_count = 0
        
        target_train = int(len(images) * train_ratio)
        target_val = int(len(images) * val_ratio)
        # test gets the rest
        
        for key in group_keys:
            group_imgs = groups[key]
            
            # Decide which split this group goes to
            if train_count + len(group_imgs) <= target_train or train_count == 0:
                split = 'train'
                train_count += len(group_imgs)
            elif val_count + len(group_imgs) <= target_val or val_count == 0:
                split = 'val'
                val_count += len(group_imgs)
            else:
                split = 'test'
                test_count += len(group_imgs)
                
            # Copy files
            for img_path in group_imgs:
                img_name = Path(img_path).name
                dest_path = dest_dir / split / cls / img_name
                if not dest_path.exists():
                    shutil.copy2(img_path, dest_path)
                    total_copied += 1
                    
        print(f"{cls}: Train({train_count}) Val({val_count}) Test({test_count})")
        
    print(f"Successfully split and created dataset into {dest_dir} (Total {total_copied} images copied)")

if __name__ == '__main__':
    main()
