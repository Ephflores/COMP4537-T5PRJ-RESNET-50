# This code was developed with assistance from OpenAI's ChatGPT.

import sys
import json
import os
from transformers import ResNetForImageClassification, AutoImageProcessor
from PIL import Image
import torch

def classify_image(image_path, model_path):
    try:
        # Normalize paths for cross-platform compatibility
        image_path = os.path.normpath(image_path)
        model_path = os.path.normpath(model_path)
        
        # Load model and processor from the local folder
        model = ResNetForImageClassification.from_pretrained(model_path)
        processor = AutoImageProcessor.from_pretrained(model_path)
        
        # Load the image
        img = Image.open(image_path).convert("RGB")
        
        # Preprocess image for the model
        inputs = processor(images=img, return_tensors="pt")
        
        # Run inference (prediction)
        with torch.no_grad():
            logits = model(**inputs).logits
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits[0], dim=0)
            pred_class_id = logits.argmax(-1).item()
            probability = probabilities[pred_class_id].item()
        
        # Get the class label
        label = model.config.id2label[pred_class_id]
        
        # Return result as JSON
        result = {
            "label": label,
            "probability": probability,
            "classId": pred_class_id
        }
        
        print(json.dumps(result))
        return result
        
    except Exception as e:
        error_result = {
            "error": str(e)
        }
        print(json.dumps(error_result), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: python inference.py <image_path> <model_path>"}), file=sys.stderr)
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    
    classify_image(image_path, model_path)

