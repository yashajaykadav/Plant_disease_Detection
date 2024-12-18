import torch
from torchvision import transforms, models
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
import os


base_dir = 'PlantVillage'  # Update this to your dataset path

class_to_idx = {}
for plant_name in os.listdir(base_dir):
    plant_dir = os.path.join(base_dir, plant_name)
    if not os.path.isdir(plant_dir):
        continue
    for disease_name in os.listdir(plant_dir):
        label = f"{plant_name}_{disease_name}"
        if label not in class_to_idx:
            class_to_idx[label] = len(class_to_idx)

# Load trained model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_to_idx))  
model.load_state_dict(torch.load("best_plant_disease_model.pth",weights_only=True))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted_class = torch.max(output, 1)
            class_name = list(class_to_idx.keys())[predicted_class.item()]
        return class_name
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error during prediction: {e}")
        return None
