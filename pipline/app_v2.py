import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import logging
import socket
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the trained model with error handling
model_path = r'd:\AAE4203\project\runs\segment\UAV-Landing-Segmentation\fold_0_final\weights\best.pt'
try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
    model = YOLO(model_path)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Class names and colors mapping
# Using standard AeroScapes mapping if possible, or defined binary classes
CLASS_INFO = {
    0: {"name": "Safe Landing Zone", "color": "🟢", "hex": "#00FF00"}, # Green
    1: {"name": "Obstacle", "color": "🔴", "hex": "#FF0000"}           # Red
}

def predict_image(img):
    """
    Run YOLO segmentation, return an image with masks only (no boxes),
    and a summary of detections. Includes error handling and color legend.
    """
    if img is None:
        return None, "⚠️ No image uploaded."
    
    if model is None:
        return img, "❌ Model failed to load. Please check server logs."
        
    try:
        # Run inference - reduce confidence threshold slightly if objects like humans are missed
        results = model.predict(source=img, conf=0.15, imgsz=640, verbose=False)
        result = results[0]
        
        # 1. Generate visual output
        # If human is not highlighted, it might be due to confidence threshold or 
        # how .plot() handles overlapping masks. 
        annotated_img = result.plot(boxes=False, labels=True, probs=False)
        
        # Convert BGR to RGB for Gradio display
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # 2. Extract detection counts and types
        summary_text = "### 📊 Detection Summary\n"
        if result.boxes is not None and len(result.boxes) > 0:
            classes_detected = result.boxes.cls.cpu().numpy()
            counts = Counter(classes_detected)
            
            for cls_id, count in counts.items():
                cls_id = int(cls_id)
                info = CLASS_INFO.get(cls_id, {"name": f"Unknown ({cls_id})", "color": "⚪", "hex": "#FFFFFF"})
                summary_text += f"- {info['color']} **{info['name']}**: {count} detected\n"
        else:
            summary_text += "No detections found.\n"
            
        # 3. Add Color Legend to Summary
        summary_text += "\n---\n### 🎨 Color Legend\n"
        summary_text += "The colors in the image represent:\n"
        summary_text += "🟢 **Green Mask**: Safe Landing Zone (Road/Flat Area)\n"
        summary_text += "🔴 **Red Mask**: Obstacle (Human, Vehicle, Object)\n"
        summary_text += "\n*Note: If a human is not highlighted, try uploading a higher resolution image or adjusting lighting conditions.*"
            
        return annotated_img_rgb, summary_text
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return img, f"❌ An error occurred during processing: {str(e)}"

def find_available_port(start_port=7861, max_tries=10):
    """Utility to find an open port starting from start_port."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except socket.error:
                continue
    return None

# Define the Gradio interface
with gr.Blocks(title="UAV Landing Zone v2") as demo:
    gr.Markdown("# 🚁 UAV Landing Zone & Obstacle Analysis (v2)")
    gr.Markdown("""
    This version provides a **cleaner visual output** by removing bounding boxes and 
    focusing on semantic segmentation masks and classification summaries.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Upload Aerial Image")
            submit_btn = gr.Button("🚀 Analyze Image", variant="primary")
            
        with gr.Column(scale=1):
            output_image = gr.Image(type="numpy", label="Segmentation Masks (Clean View)")
            detection_summary = gr.Markdown(label="Detection Summary")
            
    # Connect the logic
    submit_btn.click(
        fn=predict_image, 
        inputs=input_image, 
        outputs=[output_image, detection_summary]
    )
    
    # Examples section
    gr.Examples(
        examples=[
            [r"d:\AAE4203\project\aeroscapes\images\train\000001_004.jpg"],
            [r"d:\AAE4203\project\test2.jpg"]
        ],
        inputs=input_image
    )

if __name__ == "__main__":
    port = find_available_port(7861)
    if port:
        logger.info(f"Starting server on port {port}")
        # Passing theme to launch() as per Gradio 6.0 recommendation
        demo.launch(server_name="127.0.0.1", server_port=port, theme=gr.themes.Soft())
    else:
        logger.error("Could not find an available port to start the server.")
