import os
import optuna
import wandb
from ultralytics import YOLO
import yaml

PROJECT_NAME = "UAV-Landing-Segmentation-HPO"
DATASET_ROOT = r'd:\AAE4203\project\yolo_dataset'
# Update to newer architecture string here as well
BASE_MODEL = "yolo11n-seg.pt"

def objective(trial):
    """Optuna objective function targeting maximum obstacle-class IoU."""
    # Hyperparameters to vary
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch = trial.suggest_categorical('batch', [8, 16, 32])
    imgsz = trial.suggest_categorical('imgsz', [640, 1024])
    aug_intensity = trial.suggest_float('aug_intensity', 0.5, 1.5)
    
    # Run on fold 0 for efficiency
    dataset_yaml = os.path.join(DATASET_ROOT, 'fold_0', 'dataset.yaml')
    model = YOLO(BASE_MODEL)
    
    # Stage 1: Freeze for 20 epochs
    model.train(
        data=dataset_yaml,
        epochs=20,
        imgsz=imgsz,
        batch=batch,
        lr0=lr,
        freeze=10,
        project=PROJECT_NAME,
        name=f"trial_{trial.number}_stage1",
        degrees=90.0 * aug_intensity,
        hsv_h=0.15 * aug_intensity,
        hsv_s=0.15 * aug_intensity,
        hsv_v=0.15 * aug_intensity,
        mosaic=1.0,
        perspective=0.15 * aug_intensity,
        save=True,
        plots=False,
        verbose=False,
        exist_ok=True,
    )
    
    # Stage 2: Unfreeze with lower LR
    best_ckpt = os.path.join(PROJECT_NAME, f"trial_{trial.number}_stage1", 'weights', 'best.pt')
    model = YOLO(best_ckpt)
    
    results = model.train(
        data=dataset_yaml,
        epochs=30, # 20 + 30 = 50 total
        imgsz=imgsz,
        batch=batch,
        lr0=lr / 10.0,
        freeze=None,
        project=PROJECT_NAME,
        name=f"trial_{trial.number}_final",
        degrees=90.0 * aug_intensity,
        hsv_h=0.15 * aug_intensity,
        hsv_s=0.15 * aug_intensity,
        hsv_v=0.15 * aug_intensity,
        mosaic=1.0,
        perspective=0.15 * aug_intensity,
        save=True,
        plots=False,
        verbose=False,
        exist_ok=True,
        cls=1.5, # Static weight for now, or vary it too
    )
    
    # Get mask mAP@50 as a proxy for IoU (or use my custom IoU function)
    # results.seg.map50 is the mAP@50 for segment masks
    score = results.seg.map50 
    
    # Log to W&B
    wandb.log({
        "trial_number": trial.number,
        "lr": lr,
        "batch": batch,
        "imgsz": imgsz,
        "aug_intensity": aug_intensity,
        "mask_map50": score
    })
    
    return score

if __name__ == "__main__":
    wandb.init(project=PROJECT_NAME, job_type="HPO")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("Best Trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")
    
    # Save best params
    with open('best_params.yaml', 'w') as f:
        yaml.dump(study.best_trial.params, f)
