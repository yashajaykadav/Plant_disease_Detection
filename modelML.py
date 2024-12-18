# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision import transforms, models
# from PIL import Image, UnidentifiedImageError
# from tqdm import tqdm

# Define base dataset directory (the root directory containing plant folders)
# base_dir = 'PlantVillage'  # Update this to your dataset path

# # Define transformations
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Custom dataset class for nested directory structure
# class PlantDiseaseDataset(Dataset):
#     def __init__(self, base_dir, transform=None):
#         self.base_dir = base_dir
#         self.transform = transform
#         self.image_paths = []
#         self.labels = []
#         self.class_to_idx = {}
#         self._prepare_dataset()

#     def _prepare_dataset(self):
#         # Traverse plant and disease folders
#         for plant_name in os.listdir(self.base_dir):
#             plant_dir = os.path.join(self.base_dir, plant_name)
#             if not os.path.isdir(plant_dir):
#                 continue
#             for disease_name in os.listdir(plant_dir):
#                 disease_dir = os.path.join(plant_dir, disease_name)
#                 if not os.path.isdir(disease_dir):
#                     continue
#                 label = f"{plant_name}_{disease_name}"
#                 if label not in self.class_to_idx:
#                     self.class_to_idx[label] = len(self.class_to_idx)
#                 label_idx = self.class_to_idx[label]
#                 # Collect image paths and labels
#                 for img_name in os.listdir(disease_dir):
#                     img_path = os.path.join(disease_dir, img_name)
#                     self.image_paths.append(img_path)
#                     self.labels.append(label_idx)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         label = self.labels[idx]
#         try:
#             image = Image.open(img_path).convert("RGB")
#         except (UnidentifiedImageError, IOError):
#             print(f"Skipping invalid image file: {img_path}")
#             return self.__getitem__((idx + 1) % len(self))  # Get next valid item

#         if self.transform:
#             image = self.transform(image)
#         return image, label

# # Load entire dataset
# dataset = PlantDiseaseDataset(base_dir=base_dir, transform=transform)

# # Split dataset into training and validation sets (e.g., 80% train, 20% validation)
# train_size = int(0.8 * len(dataset))
# valid_size = len(dataset) - train_size
# train_data, valid_data = random_split(dataset, [train_size, valid_size])

# # Create data loaders
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

# # Load MobileNetV2 and modify for custom classification
# num_classes = len(dataset.class_to_idx)
# model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
# model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# # Move model to device (GPU if available)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 10
# best_accuracy = 0.0

# for epoch in range(num_epochs):
#     print(f"Epoch {epoch + 1}/{num_epochs}")

#     # Training phase
#     model.train()
#     running_loss = 0.0
#     for images, labels in tqdm(train_loader):
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     avg_train_loss = running_loss / len(train_loader)
#     print(f"Training Loss: {avg_train_loss:.4f}")

#     # Validation phase
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in valid_loader:
#             images, labels = images.to(device), labels.to(device)

#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = 100 * correct / total
#     print(f"Validation Accuracy: {accuracy:.2f}%")

#     # Save model after every epoch
#     epoch_model_path = f"plant_disease_model_epoch_{epoch + 1}.pth"
#     torch.save(model.state_dict(), epoch_model_path)
#     print(f"Model saved as {epoch_model_path}")

#     # Update best model
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         torch.save(model.state_dict(), "best_plant_disease_model.pth")
#         print("Best model updated and saved as best_plant_disease_model.pth")

# print("Training complete!")
