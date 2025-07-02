import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

# =======================
# Hyperparameters
# =======================
BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_CLASSES = 4
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# Output Directory
# =======================
output_dir = "results_resnet"
os.makedirs(output_dir, exist_ok=True)

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
# Load Dataset
# =======================
train_dataset = datasets.ImageFolder(root='../dataset/train', transform=transform)
val_dataset = datasets.ImageFolder(root='../dataset/val', transform=transform)
class_names = train_dataset.classes

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =======================
# Load Pretrained ResNet152
# =======================
resnet152 = models.resnet152(pretrained=True)
resnet152.fc = nn.Linear(resnet152.fc.in_features, NUM_CLASSES)
resnet152.to(DEVICE)

# =======================
# Loss & Optimizer
# =======================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet152.parameters(), lr=LEARNING_RATE)

# =======================
# Tracking Lists
# =======================
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# =======================
# Training Loop
# =======================
for epoch in range(NUM_EPOCHS):
    resnet152.train()
    total_loss, total_correct = 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = resnet152(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    acc = 100 * total_correct / len(train_dataset)

    train_losses.append(avg_loss)
    train_accuracies.append(acc)
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

    # ====================
    # Validation
    # ====================
    resnet152.eval()
    val_correct = 0
    val_total_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = resnet152(images)
            loss = criterion(outputs, labels)
            val_total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    val_avg_loss = val_total_loss / len(val_loader)
    val_acc = 100 * val_correct / len(val_dataset)

    val_losses.append(val_avg_loss)
    val_accuracies.append(val_acc)
    print(f"Validation Loss: {val_avg_loss:.4f}, Accuracy: {val_acc:.2f}%")

# ====================
# Save Model
# ====================
torch.save(resnet152.state_dict(), os.path.join(output_dir, "resnet152_endoscopy_classifier.pth"))

# ====================
# Save Loss Plot
# ====================
plt.figure()
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "loss_curve.png"))

# ====================
# Save Accuracy Plot
# ====================
plt.figure()
plt.plot(train_accuracies, label='Train Accuracy', color='green')
plt.plot(val_accuracies, label='Validation Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))

# ====================
# Save CSV Log
# ====================
csv_path = os.path.join(output_dir, "training_log.csv")
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Train Accuracy', 'Validation Accuracy'])
    for i in range(NUM_EPOCHS):
        writer.writerow([i + 1, train_losses[i], val_losses[i], train_accuracies[i], val_accuracies[i]])

print(f"Training complete. Results saved to '{output_dir}'")
