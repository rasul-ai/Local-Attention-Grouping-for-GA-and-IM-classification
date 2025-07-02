import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# =======================
# Device
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# Parameters
# =======================
NUM_CLASSES = 4
BATCH_SIZE = 16
MODEL_PATH = "results/vit_endoscopy_classifier.pth"
TEST_DIR = "../dataset/test"

# =======================
# Transforms
# =======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# =======================
# Test Dataset & Loader
# =======================
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_dataset.classes

# =======================
# Load Model
# =======================
model = models.vit_b_16(pretrained=False)
model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =======================
# Inference
# =======================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# =======================
# Evaluation
# =======================
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# =======================
# Confusion Matrix
# =======================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.show()
