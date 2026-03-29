import cv2
import argparse
from ultralytics import YOLO
import numpy as np

def run_inference(model_path, source, imgsz=640, conf=0.25):
    """
    Run YOLOv8-seg inference for safe landing zones and obstacles.
    Supports Jetson Orin deployment using TensorRT (.engine).
    """
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(source, imgsz=imgsz, conf=conf, stream=True)
    
    for r in results:
        # Visualize results
        annotated_frame = r.plot()
        
        # Display frame
        cv2.imshow("UAV Landing Segmentation", annotated_frame)
        
        # Extract binary masks if needed
        if r.masks is not None:
            # Mask 0: Safe Landing Zone
            # Mask 1: Obstacle
            pass
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best.engine', help='Model path (.pt or .engine)')
    parser.add_argument('--source', type=str, default='0', help='Image path or camera ID')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    args = parser.parse_args()
    
    run_inference(args.model, args.source, args.imgsz)
