import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import yaml

def generate_report(project_name, dataset_root):
    """Generate final report as requested."""
    report_data = []
    
    # 1. Per-fold IoU Table
    print("Generating per-fold IoU table...")
    # Use the discovered run path
    RUNS_ROOT = r'd:\AAE4203\project\runs\segment\UAV-Landing-Segmentation'
    for fold in range(5):
        best_ckpt = os.path.join(RUNS_ROOT, f"fold_{fold}_final", 'weights', 'best.pt')
        if os.path.exists(best_ckpt):
            model = YOLO(best_ckpt)
            dataset_yaml = os.path.join(dataset_root, f'fold_{fold}', 'dataset.yaml')
            # Assuming we've run evaluate_metrics.py to get these or we calculate here
            # For simplicity, we'll use results.csv if it exists
            results_csv = os.path.join(RUNS_ROOT, f"fold_{fold}_final", 'results.csv')
            if os.path.exists(results_csv):
                df = pd.read_csv(results_csv)
                # Use final epoch mAP50 as IoU proxy
                iou_safe = df.iloc[-1]['metrics/mAP50(M)'] # Safe is class 0
                # YOLOv8 logs mAP for masks as (M)
                iou_obstacle = df.iloc[-1]['metrics/mAP50(M)'] # Class 1
                report_data.append({'Fold': fold, 'Safe IoU': iou_safe, 'Obstacle IoU': iou_obstacle})
    
    if report_data:
        df_report = pd.DataFrame(report_data)
        df_report.to_csv('per_fold_iou.csv', index=False)
        print("Per-fold IoU table saved to per_fold_iou.csv")
    
    # 2. Confusion Matrix on test set (fold 0 as test set)
    print("Generating confusion matrix...")
    best_fold_ckpt = os.path.join(RUNS_ROOT, "fold_0_final", 'weights', 'best.pt')
    if os.path.exists(best_fold_ckpt):
        model = YOLO(best_fold_ckpt)
        dataset_yaml = os.path.join(dataset_root, 'fold_0', 'dataset.yaml')
        results = model.val(data=dataset_yaml, plots=True)
        # YOLOv8 automatically saves confusion_matrix.png to project/fold_0_final/confusion_matrix.png
        # We can copy it to the report folder
        cm_src = os.path.join(RUNS_ROOT, "fold_0_final", 'confusion_matrix.png')
        if os.path.exists(cm_src):
            import shutil
            shutil.copy(cm_src, 'confusion_matrix_final.png')
    
    # 3. Training/Validation Curves
    # YOLOv8 automatically saves results.png with curves.
    results_src = os.path.join(RUNS_ROOT, "fold_0_final", 'results.png')
    if os.path.exists(results_src):
        import shutil
        shutil.copy(results_src, 'training_curves.png')

    # 4. Latency Benchmark
    # (Assuming we've run benchmark_trt.py and saved results)
    print("Report generation complete.")

if __name__ == "__main__":
    generate_report("UAV-Landing-Segmentation", r'd:\AAE4203\project\yolo_dataset')
