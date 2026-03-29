import argparse
import cv2
import os
from ultralytics import YOLO

def test_single_image(image_path, model_path="best.pt", output_path="output.jpg"):
    """
    Test the trained YOLO segmentation model on a custom picture.
    """
    if not os.path.exists(image_path):
        print(f"Error: The image '{image_path}' does not exist.")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: The model file '{model_path}' was not found.")
        print("Please provide the correct path to your trained model weights (e.g., runs/segment/UAV-Landing-Segmentation/fold_0_final/weights/best.pt).")
        return

    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Run inference
    print(f"Running inference on {image_path}...")
    results = model.predict(source=image_path, conf=0.25, imgsz=640)
    
    # Process results
    for r in results:
        # plot() generates a BGR numpy array with the bounding boxes and masks drawn on it
        annotated_img = r.plot()
        
        # Save the result
        cv2.imwrite(output_path, annotated_img)
        print(f"Success! Annotated image saved to {output_path}")
        
        # Optionally show the image if running in an environment with GUI
        # cv2.imshow("Result", annotated_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test custom picture with YOLO segmentation model")
    parser.add_argument("--image", type=str, required=True, help="Path to your custom image")
    parser.add_argument("--model", type=str, default=r"d:\AAE4203\project\runs\segment\UAV-Landing-Segmentation\fold_0_final\weights\best.pt", help="Path to the trained .pt model")
    parser.add_argument("--output", type=str, default="test_result.jpg", help="Where to save the output image")
    
    args = parser.parse_args()
    test_single_image(args.image, args.model, args.output)
