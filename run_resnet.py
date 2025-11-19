from transformers import ResNetForImageClassification, AutoImageProcessor
from PIL import Image
import torch

# Path to your local clone of the model
model_path = r"C:\Users\ephra\Desktop\HuggingFace\resnet-50"

# Load model and processor from the local folder
model = ResNetForImageClassification.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)

# Load an example image
img = Image.open("your_image.jpg").convert("RGB")

# Preprocess image for the model
inputs = processor(images=img, return_tensors="pt")

# Run inference (prediction)
with torch.no_grad():
    logits = model(**inputs).logits
    pred_class_id = logits.argmax(-1).item()

# Get the class label
label = model.config.id2label[pred_class_id]

print(f"Predicted class ID: {pred_class_id}")
print(f"Predicted label: {label}")
