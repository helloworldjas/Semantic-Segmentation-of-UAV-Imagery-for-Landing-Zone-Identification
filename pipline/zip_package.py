import zipfile
import os

files_to_zip = [
    r'd:\AAE4203\project\runs\segment\UAV-Landing-Segmentation\fold_0_final\weights\best.pt',
    r'd:\AAE4203\project\augmentation_config.yaml',
    r'd:\AAE4203\project\inference_jetson.py',
    r'd:\AAE4203\project\yolo_dataset\fold_0\dataset.yaml',
    r'd:\AAE4203\project\per_fold_iou.csv',
    r'd:\AAE4203\project\training_curves.png',
    r'd:\AAE4203\project\confusion_matrix_final.png'
]

with zipfile.ZipFile(r'd:\AAE4203\project\deployment_package.zip', 'w') as zipf:
    for file in files_to_zip:
        if os.path.exists(file):
            zipf.write(file, os.path.basename(file))
        else:
            print(f"Warning: File not found {file}")

print("Package created at d:\AAE4203\project\deployment_package.zip")
