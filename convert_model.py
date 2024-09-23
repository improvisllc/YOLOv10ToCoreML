import argparse
import os
from ultralytics import YOLO

def export_to_coreml(model_path):
    
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Export the model to CoreML format
    model.export(format="coreml")
    
    # Extract the name without extension
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # Generate CoreML output name based on the input model name
    coreml_model_name = f"{model_name}.mlpackage"
    
    # Load the exported CoreML model
    coreml_model = YOLO(coreml_model_name)
    
    # Run inference
    results = coreml_model("https://ultralytics.com/images/bus.jpg")
    
    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Export YOLO model to CoreML format")
    parser.add_argument("model_path", type=str, help="Path to the YOLO model file (.pt)")
    
    args = parser.parse_args()
    
    results = export_to_coreml(args.model_path)
    
    print(results)

