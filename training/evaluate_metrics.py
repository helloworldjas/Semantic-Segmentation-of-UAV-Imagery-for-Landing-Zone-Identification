import os
import yaml
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import torch

def calculate_per_class_iou(model, dataset_yaml, class_id=1):
    """
    Calculate per-class IoU for semantic segmentation on the validation set.
    """
    # Load dataset.yaml to find validation images
    with open(dataset_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    val_file = os.path.join(data_cfg['path'], data_cfg['val'])
    with open(val_file, 'r') as f:
        val_images = f.read().splitlines()
    
    # Run inference on validation set and calculate IoU
    total_intersection = 0
    total_union = 0
    
    # We'll use the mask output of YOLOv8-seg
    for img_path in tqdm(val_images, desc=f"Calculating IoU for class {class_id}"):
        # Load image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # Load ground truth mask (binary mask for class_id)
        # Assuming label files exist in ../labels/val/img_name.txt
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if int(parts[0]) == class_id:
                        poly = np.array(list(map(float, parts[1:]))).reshape(-1, 2)
                        poly[:, 0] *= w
                        poly[:, 1] *= h
                        cv2.fillPoly(gt_mask, [poly.astype(np.int32)], 1)
        
        # Run inference
        results = model(img, verbose=False)
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        
        if results[0].masks is not None:
            for i, mask in enumerate(results[0].masks.xy):
                cls_id = int(results[0].boxes.cls[i])
                if cls_id == class_id:
                    poly = mask.astype(np.int32)
                    cv2.fillPoly(pred_mask, [poly], 1)
        
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        
        total_intersection += intersection
        total_union += union
        
    iou = total_intersection / (total_union + 1e-6)
    return iou

def evaluate_folds(project_name, dataset_root, class_id=1):
    """Evaluate all folds and report mean IoU on the target class."""
    ious = []
    for fold in range(5):
        best_ckpt = os.path.join(project_name, f"fold_{fold}_final", 'weights', 'best.pt')
        if not os.path.exists(best_ckpt):
            continue
        model = YOLO(best_ckpt)
        dataset_yaml = os.path.join(dataset_root, f'fold_{fold}', 'dataset.yaml')
        iou = calculate_per_class_iou(model, dataset_yaml, class_id=class_id)
        ious.append(iou)
        print(f"Fold {fold} - Obstacle IoU: {iou:.4f}")
    
    if ious:
        print(f"Mean Obstacle IoU across all folds: {np.mean(ious):.4f}")
    return ious
