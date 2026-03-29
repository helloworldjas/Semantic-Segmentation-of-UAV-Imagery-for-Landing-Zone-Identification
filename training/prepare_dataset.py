import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import yaml

# Configuration
DATA_ROOT = r'd:\AAE4203\project\aeroscapes'
IMG_DIR = os.path.join(DATA_ROOT, 'images', 'train')
MASK_DIR = os.path.join(DATA_ROOT, 'SegmentationClass')
OUTPUT_DIR = r'd:\AAE4203\project\yolo_dataset'
LABEL_DIR = os.path.join(DATA_ROOT, 'labels')

# Class Mapping
# 0: Safe Landing Zone (Road=10, Background=0)
# 1: Obstacle (1-9)
SAFE_IDS = [0, 10]
OBSTACLE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

def mask_to_polygons(mask, ids):
    """Convert mask with multiple IDs to normalized polygons."""
    binary_mask = np.isin(mask, ids).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    h, w = mask.shape
    for cnt in contours:
        if len(cnt) < 3: continue
        poly = cnt.reshape(-1, 2).astype(np.float32)
        poly[:, 0] /= w
        poly[:, 1] /= h
        polygons.append(poly.flatten().tolist())
    return polygons

def prepare_data():
    if os.path.exists(LABEL_DIR):
        import shutil
        shutil.rmtree(LABEL_DIR)
    os.makedirs(LABEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    data = []
    
    print("Converting masks to YOLO polygons...")
    for img_name in tqdm(image_files):
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(MASK_DIR, mask_name)
        if not os.path.exists(mask_path): continue
        mask = cv2.imread(mask_path, 0)
        if mask is None: continue
        
        labels = []
        # Safe -> 0
        for poly in mask_to_polygons(mask, SAFE_IDS):
            labels.append(f"0 {' '.join(map(str, poly))}")
        
        # Obstacle -> 1
        has_obstacle = False
        polys_obs = mask_to_polygons(mask, OBSTACLE_IDS)
        if polys_obs:
            has_obstacle = True
            for poly in polys_obs:
                labels.append(f"1 {' '.join(map(str, poly))}")
        
        if not labels: continue
        
        label_path = os.path.join(LABEL_DIR, img_name.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
            
        has_safe = any(l.startswith('0') for l in labels)
        strat_label = (1 if has_obstacle else 0) + (2 if has_safe else 0)
        data.append({'img_path': os.path.abspath(os.path.join(IMG_DIR, img_name)), 'strat_label': strat_label})

    paths = np.array([d['img_path'] for d in data])
    strat_labels = np.array([d['strat_label'] for d in data])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, strat_labels)):
        fold_dir = os.path.join(OUTPUT_DIR, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        
        with open(os.path.join(fold_dir, 'train.txt'), 'w') as f:
            f.write('\n'.join(paths[train_idx]))
        with open(os.path.join(fold_dir, 'val.txt'), 'w') as f:
            f.write('\n'.join(paths[val_idx]))
            
        dataset_cfg = {
            'path': os.path.abspath(fold_dir),
            'train': 'train.txt',
            'val': 'val.txt',
            'names': {0: 'safe', 1: 'obstacle'}
        }
        with open(os.path.join(fold_dir, 'dataset.yaml'), 'w') as f:
            yaml.dump(dataset_cfg, f)
            
    print(f"Dataset preparation complete. Folds saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_data()
