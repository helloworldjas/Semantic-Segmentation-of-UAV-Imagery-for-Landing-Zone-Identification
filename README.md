# UAV Safe Landing Zone Segmentation Pipeline

## Project Overview

This project implements an end-to-end UAV vision pipeline for **safe landing zone analysis** using segmentation models from Ultralytics YOLO. The workflow converts AeroScapes semantic masks into YOLO segmentation labels, trains a binary segmentation model, evaluates per-fold performance, benchmarks inference speed, and provides both command-line and web UI inference interfaces.

The core detection task is binary:

- **Class 0: Safe Landing Zone** (road/flat background regions)
- **Class 1: Obstacle** (people, vehicles, vegetation, structures, and other non-landing-safe objects)

The repository includes:

- Dataset conversion and stratified 5-fold split generation
- Two-stage YOLO segmentation training (freeze/unfreeze)
- Hyperparameter tuning with Optuna + Weights & Biases
- Fold-level IoU evaluation and report artifact generation
- TensorRT benchmark helper for Jetson-oriented deployment
- Interactive Gradio UI for image upload and mask visualization

---

## Detection Methodology

### 1) Data Representation and Class Mapping

Source masks are read from AeroScapes semantic labels (`SegmentationClass/*.png`) and remapped into two binary groups:

- Safe IDs: `0, 10`
- Obstacle IDs: `1..9`

Each image is converted from raster masks to YOLO segmentation polygons:

1. Build binary mask per target class group
2. Extract contours via OpenCV
3. Normalize polygon coordinates to `[0, 1]`
4. Save per-image label as YOLO-seg `.txt`

This conversion is implemented in `prepare_dataset.py` with `mask_to_polygons()` and `prepare_data()`.

### 2) Stratified 5-Fold Cross Validation

The preprocessing script computes a simple stratification flag based on whether an image contains safe and/or obstacle labels, then applies:

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

For each fold, it generates:

- `train.txt` and `val.txt` (absolute image paths)
- `dataset.yaml` (YOLO data config with class names)

### 3) Model Training Strategy

Training uses Ultralytics segmentation models (currently configured as `yolo26n-seg.pt`) in a two-stage schedule:

- **Stage 1 (stabilization):**
  - Freeze first 10 layers
  - Train 20 epochs
- **Stage 2 (fine-tuning):**
  - Unfreeze all layers
  - Train remaining epochs with `lr0 / 10`

Key augmentation choices tuned for aerial imagery:

- `degrees=90`
- `hsv_h/s/v=0.15`
- `mosaic=1.0`
- `perspective=0.15`

Classification gain (`cls=1.5`) is increased to bias learning toward minority obstacle behavior.

### 4) Hyperparameter Optimization

`hpo_tuning.py` runs Optuna trials over:

- learning rate (`lr`)
- batch size (`batch`)
- image size (`imgsz`)
- augmentation intensity scalar (`aug_intensity`)

Objective is maximizing segmentation `map50` proxy after two-stage training on fold 0, logged to Weights & Biases.

### 5) Evaluation and Reporting

`evaluate_metrics.py` computes per-class IoU by rasterizing both:

- ground-truth polygons from YOLO labels
- predicted polygons from model masks

and accumulating global intersection/union over validation images.

`generate_report.py` consolidates artifacts:

- `per_fold_iou.csv`
- `confusion_matrix_final.png`
- `training_curves.png`

### 6) Inference and Visualization

Two user-facing inference options:

- **CLI single image**: `test_my_picture.py`
- **Web UI**: `app_v2.py` (Gradio upload + segmentation output + detection summary)

UI output uses semantic color meaning:

- Green mask = Safe Landing Zone
- Red mask = Obstacle

---

## Repository Structure

```text
project/
├─ aeroscapes/                         # Raw dataset root (images, masks, original caches)
│  ├─ images/train/                    # Input training images
│  ├─ SegmentationClass/               # Input semantic masks (.png)
│  └─ labels/                          # Generated YOLO polygon labels
├─ yolo_dataset/                       # Generated 5-fold YOLO dataset manifests
│  ├─ fold_0/
│  │  ├─ dataset.yaml                  # Fold-specific YOLO data config
│  │  ├─ train.txt                     # Absolute paths to fold train images
│  │  └─ val.txt                       # Absolute paths to fold val images
│  ├─ fold_1/ ... fold_4/              # Remaining folds
├─ runs/segment/                       # Ultralytics training/validation outputs
│  └─ UAV-Landing-Segmentation/
│     ├─ fold_0_stage1/
│     ├─ fold_0_final/
│     └─ ...                           # Fold-specific checkpoints, curves, metrics
├─ prepare_dataset.py                  # Mask->polygon conversion + stratified 5-fold split
├─ train_yolo.py                       # Two-stage fold training pipeline
├─ hpo_tuning.py                       # Optuna+W&B hyperparameter search
├─ evaluate_metrics.py                 # Per-class IoU computation on validation sets
├─ generate_report.py                  # Report artifacts (CSV + figures) generation
├─ benchmark_trt.py                    # TensorRT export + latency benchmark helper
├─ inference_jetson.py                 # Runtime inference script for .pt/.engine models
├─ app_v2.py                           # Refined Gradio interface with legend/summary
├─ augmentation_config.yaml            # UAV-specific augmentation parameter reference
├─ landing_zones.yaml                  # Legacy YOLO dataset config (12-class mapping)
├─ zip_package.py                      # Deployment bundle creator (deployment_package.zip)
├─ test_my_picture.py                  # Pipeline excute in CLI
└─ deployment_package.zip              # Packaged deployment artifacts

```

---

## Prerequisites

### System Requirements

- OS: Windows 10/11 (current project paths are Windows-style absolute paths)
- Python: 3.10+ (project has been used with Python 3.13 virtual environment)
- GPU: NVIDIA CUDA-capable GPU recommended for training/inference speed
- Optional for deployment benchmark:
  - TensorRT runtime/SDK
  - Jetson Orin target (for final embedded deployment)

### Python Dependencies

Install these packages in a virtual environment:

```powershell
pip install ultralytics opencv-python numpy tqdm scikit-learn pyyaml
pip install optuna wandb gradio pandas matplotlib seaborn
```

If you use GPU training, make sure PyTorch + CUDA build is installed appropriately for your CUDA version.

### Environment Setup

```powershell
cd d:\AAE4203\project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install ultralytics opencv-python numpy tqdm scikit-learn pyyaml optuna wandb gradio pandas matplotlib seaborn
```

---

## Run the Pipeline

### Step 1: Verify Dataset Layout

Expected input structure:

```text
aeroscapes/
├─ images/train/*.jpg
└─ SegmentationClass/*.png
```

### Step 2: Prepare YOLO-Seg Dataset + 5 Folds

```powershell
python prepare_dataset.py
```

Expected outputs:

- `aeroscapes/labels/*.txt`
- `yolo_dataset/fold_0..fold_4/{dataset.yaml,train.txt,val.txt}`

### Step 3: Train the 5-Fold Model

```powershell
python train_yolo.py
```

Expected outputs:

- `runs/segment/UAV-Landing-Segmentation/fold_*_stage1/`
- `runs/segment/UAV-Landing-Segmentation/fold_*_final/weights/best.pt`
- Per-fold training curves and metrics files in each run directory

### Step 4: Optional Hyperparameter Optimization (Optuna)

```powershell
python hpo_tuning.py
```

Expected outputs:

- W&B run logs
- `best_params.yaml`
- Trial directories under `UAV-Landing-Segmentation-HPO/`

### Step 5: Evaluate IoU Across Folds

`evaluate_metrics.py` provides functions (no direct CLI entrypoint). Run this from PowerShell:

```powershell
python -c "from evaluate_metrics import evaluate_folds; evaluate_folds(r'd:\AAE4203\project\runs\segment\UAV-Landing-Segmentation', r'd:\AAE4203\project\yolo_dataset', class_id=1)"
```

Expected output:

- Console per-fold obstacle IoU
- Mean obstacle IoU

### Step 6: Generate Final Report Artifacts

```powershell
python generate_report.py
```

Expected outputs:

- `per_fold_iou.csv`
- `confusion_matrix_final.png`
- `training_curves.png`

### Step 7: Test on a Custom Image (CLI)

```powershell
python test_my_picture.py --image test2.jpg --model d:\AAE4203\project\runs\segment\UAV-Landing-Segmentation\fold_0_final\weights\best.pt --output test2_result.jpg
```

Expected output:

- Annotated image saved to `test2_result.jpg`

### Step 8: Launch Web UI

```powershell
python app_v2.py
```

Expected behavior:

- Starts local Gradio server (auto-selects an available port near `7861`)
- Allows image upload and returns:
  - Segmentation visualization
  - Detection summary by class
  - Color legend (green safe, red obstacle)

### Step 9: Benchmark Latency (TensorRT/FP16 Path)

```powershell
python benchmark_trt.py
```

Expected behavior:

- Attempts `.pt -> .engine` export
- If TensorRT export fails, falls back to `.pt`
- Prints mean latency (ms) and FPS

### Step 10: Package Deployment Assets

```powershell
python zip_package.py
```

Expected output:

- `deployment_package.zip` containing key model/config/report assets

---

## Configuration Notes

- Most scripts currently use absolute paths rooted at `d:\AAE4203\project`.
- To relocate the project, update constants such as:
  - `DATA_ROOT`, `OUTPUT_DIR` in `prepare_dataset.py`
  - `DATASET_ROOT`, `PROJECT_NAME`, `BASE_MODEL` in `train_yolo.py`
  - `model_path` in `app_v2.py`

---

## Troubleshooting

### 1) `Model weights not found` in UI or inference

- Verify trained checkpoint exists:
  - `runs/segment/UAV-Landing-Segmentation/fold_0_final/weights/best.pt`
- Update model path in `app_v2.py` or pass `--model` explicitly in CLI.

### 2) Gradio server port already in use

- `app_v2.py` already scans for an open port (starting at 7861).
- If startup still fails, close old Python/Gradio processes and restart.

### 3) No detections or missed small humans/obstacles

- Use higher resolution inputs if possible.
- Lower confidence threshold for inference (UI uses `conf=0.15`).
- Improve training with more samples containing small objects.

### 4) TensorRT export errors on Windows

- Ensure TensorRT SDK and compatible CUDA/cuDNN are installed.
- If unavailable, use `.pt` inference path and skip engine export.

### 5) Slow training or CUDA not used

- Verify CUDA visibility in Python:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

- Install GPU-enabled PyTorch matching your CUDA environment.

### 6) Dataset conversion issues (missing labels or empty folds)

- Confirm both `images/train/*.jpg` and matching `SegmentationClass/*.png` exist.
- Re-run `prepare_dataset.py` to regenerate `aeroscapes/labels` and `yolo_dataset`.

---

## Sample Result

The sample result in Web app 
<img width="3836" height="1773" alt="image" src="https://github.com/user-attachments/assets/459b52ac-4d64-49d4-90c0-98e5de8194f9" />


## References

- Ultralytics YOLO Documentation: https://docs.ultralytics.com/
- Ultralytics Python API (train/val/predict/export): https://docs.ultralytics.com/usage/python/
- AeroScapes Dataset (UAV semantic segmentation): https://github.com/UAV-Centre-ITC/AeroScapes
- Optuna Documentation: https://optuna.org/
- Weights & Biases Documentation: https://docs.wandb.ai/
- Gradio Documentation: https://www.gradio.app/docs
- OpenCV Documentation: https://docs.opencv.org/
- NVIDIA TensorRT Documentation: https://docs.nvidia.com/deeplearning/tensorrt/

