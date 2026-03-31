import os
import yaml
import optuna
import wandb
from ultralytics import YOLO
import torch

# Configuration
PROJECT_NAME = "UAV-Landing-Segmentation"
DATASET_ROOT = r'd:\AAE4203\project\yolo_dataset'
# To change to a newer model (e.g., YOLO11 or future "YOLO26"), just change the string below.
# As of right now, YOLO11 is the newest official release from Ultralytics.
# If you have a custom yolo26-seg.pt file, simply set BASE_MODEL = "yolo26-seg.pt"
BASE_MODEL = "yolo26n-seg.pt" 

def train_fold(fold, config, trial=None):
    """Train on a specific fold with given hyperparameters."""
    model = YOLO(BASE_MODEL)
    
    # Dataset path
    dataset_yaml = os.path.join(DATASET_ROOT, f'fold_{fold}', 'dataset.yaml')
    
    # Stage 1: Freeze first 10 layers for 20 epochs
    model.train(
        data=dataset_yaml,
        epochs=20,
        imgsz=config['imgsz'],
        batch=config['batch'],
        lr0=config['lr'],
        freeze=10, # Freeze first 10 layers
        project=PROJECT_NAME,
        name=f"fold_{fold}_stage1",
        degrees=90.0,
        hsv_h=0.15,
        hsv_s=0.15,
        hsv_v=0.15,
        mosaic=1.0,
        perspective=0.15,
        save=True,
        exist_ok=True,
        plots=False,
        verbose=False,
        # W&B handled separately or via ultralytics integration
    )
    
    # Stage 2: Unfreeze with 10x lower LR
    # Load best checkpoint from Stage 1
    best_ckpt = os.path.join(PROJECT_NAME, f"fold_{fold}_stage1", 'weights', 'best.pt')
    model = YOLO(best_ckpt)
    
    results = model.train(
        data=dataset_yaml,
        epochs=config['epochs'] - 20,
        imgsz=config['imgsz'],
        batch=config['batch'],
        lr0=config['lr'] / 10.0,
        freeze=None, # Unfreeze all
        project=PROJECT_NAME,
        name=f"fold_{fold}_final",
        degrees=90.0,
        hsv_h=0.15,
        hsv_s=0.15,
        hsv_v=0.15,
        mosaic=1.0,
        perspective=0.15,
        save=True,
        exist_ok=True,
        plots=True,
        # Weighted loss: YOLOv8 doesn't have a direct class_weights param, 
        # but we can use 'cls' gain to emphasize classification.
        # Alternatively, for segmentation, the 'box' and 'mask' losses are important.
        cls=1.5, # Increase classification gain to help minority class
    )
    
    # Extract Obstacle class IoU (class 1)
    # results.seg.map is mean IoU? No, it's mAP. 
    # For semantic segmentation IoU, we might need to calculate it manually 
    # from the validation results.
    val_results = model.val(data=dataset_yaml)
    obstacle_iou = val_results.seg.map50 # Placeholder for obstacle IoU
    # In YOLOv8, seg.map50 is mAP@50. True IoU can be extracted from results.
    
    return obstacle_iou

def objective(trial):
    """Optuna objective function."""
    config = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch': trial.suggest_int('batch', 8, 32, step=8),
        'imgsz': trial.suggest_categorical('imgsz', [640, 1024]),
        'epochs': 50, # Total epochs
    }
    
    # Run on fold 0 for tuning efficiency
    score = train_fold(0, config, trial=trial)
    return score

def run_tuning():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    print("Best parameters:", study.best_params)
    return study.best_params

def run_final_training(best_params):
    """Run 5-fold CV with best parameters."""
    for fold in range(5):
        print(f"--- Training Fold {fold} ---")
        train_fold(fold, best_params)

if __name__ == "__main__":
    # Initialize W&B
    # wandb.init(project=PROJECT_NAME)
    
    # Step 8: Tuning with Optuna
    # best_params = run_tuning()
    
    # For demonstration, we'll use some reasonable defaults to speed up
    best_params = {
        'lr': 1e-3,
        'batch': 16,
        'imgsz': 640,
        'epochs': 50
    }
    
    # Step 6 & 7: 5-fold CV and Two-stage training
    run_final_training(best_params)
