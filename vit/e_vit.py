import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# =======================
# Hyperparameters
# =======================
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_CLASSES = 4
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Load Full Dataset
# =======================
train_dataset = datasets.ImageFolder(root='../dataset/train', transform=transform)
val_dataset = datasets.ImageFolder(root='../dataset/val', transform=transform)

class_names = train_dataset.classes

# =======================
# Split Dataset (80% train, 20% val)
# =======================
# total_size = len(full_dataset)
# val_size = int(VAL_SPLIT * total_size)
# train_size = total_size - val_size

# train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =======================
# Load Pretrained ViT
# =======================
vit = models.vit_b_16(pretrained=True)
vit.heads.head = nn.Linear(vit.heads.head.in_features, NUM_CLASSES)
vit.to(DEVICE)

# =======================
# Loss & Optimizer
# =======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vit.parameters(), lr=LEARNING_RATE)

# =======================
# Training Loop
# =======================
for epoch in range(NUM_EPOCHS):
    vit.train()
    total_loss, total_correct = 0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = vit(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()

    acc = 100 * total_correct / len(train_dataset)
    print(f"Train Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

    # ====================
    # Validation
    # ====================
    vit.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = vit(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / len(val_dataset)
    print(f"Validation Accuracy: {val_acc:.2f}%")

# ====================
# Save the Model
# ====================
torch.save(vit.state_dict(), "vit_endoscopy_classifier.pth")
