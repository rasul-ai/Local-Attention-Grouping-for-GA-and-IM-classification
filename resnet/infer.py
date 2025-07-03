import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# =======================
# Setup
# =======================
BATCH_SIZE = 16
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 224
output_dir = "results_resnet152_aug"
model_path = os.path.join(output_dir, "resnet152_model_checkpoint.pth")
class_names = ['esophagitis', 'normal', 'polyps', 'ulcerative_colitis']

# =======================
# Transforms
# =======================
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# =======================
# Load Test Dataset
# =======================
test_dataset = datasets.ImageFolder(root='../dataset/test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =======================
# Load Model
# =======================
model = models.resnet152(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=DEVICE)['model_state'])
model.to(DEVICE)
model.eval()

# =======================
# Inference & Metrics
# =======================
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

# =======================
# Report
# =======================
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# =======================
# Confusion Matrix
# =======================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()
