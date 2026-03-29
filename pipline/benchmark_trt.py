import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

def benchmark_latency(model_path, imgsz=(2160, 3840), num_runs=100):
    """Benchmark inference latency on 4k frames using TensorRT FP16."""
    # Export to TensorRT FP16 if not already
    model = YOLO(model_path)
    
    # Export to TensorRT
    # Note: On Windows, this requires TensorRT SDK installed and configured.
    # We'll use the .engine if it exists, otherwise use .pt and warn.
    engine_path = model_path.replace('.pt', '.engine')
    if not os.path.exists(engine_path):
        print("Exporting model to TensorRT FP16 (this may take several minutes)...")
        # model.export(format='engine', half=True, device=0, imgsz=imgsz)
        # For this sandbox, we'll just use the PT model if export fails.
        # But we'll try it.
        try:
            model.export(format='engine', half=True, device=0, imgsz=imgsz)
        except Exception as e:
            print(f"TensorRT export failed: {e}. Benchmarking on PT FP16 instead.")
            engine_path = model_path
    
    # Load model
    bench_model = YOLO(engine_path)
    
    # Create dummy 4k frame
    dummy_frame = np.random.randint(0, 255, (imgsz[0], imgsz[1], 3), dtype=np.uint8)
    
    print(f"Benchmarking on {imgsz[1]}x{imgsz[0]} frame...")
    
    # Warmup
    for _ in range(10):
        bench_model(dummy_frame, verbose=False)
    
    # Measure latency
    latencies = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        bench_model(dummy_frame, verbose=False)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # in ms
    
    mean_latency = np.mean(latencies)
    fps = 1000 / mean_latency
    print(f"Mean Latency: {mean_latency:.2f} ms")
    print(f"Mean FPS: {fps:.2f}")
    
    return mean_latency, fps

if __name__ == "__main__":
    # Use best checkpoint from fold 0 for benchmarking
    best_ckpt = r'd:\AAE4203\project\runs\segment\UAV-Landing-Segmentation\fold_0_final\weights\best.pt'
    if os.path.exists(best_ckpt):
        benchmark_latency(best_ckpt)
    else:
        print(f"Checkpoint not found at {best_ckpt}. Please run training first.")
