import os
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

# from lag import LAG
# from lag_with_h_v_p_w_f_e import LAG
from lag_horizontal import LAG
from preprocess import preprocess_dataset


model = LAG()
loss_function = nn.BCEWithLogitsLoss()
# loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)

train_dataloader = preprocess_dataset.train_dataloader
val_dataloader = preprocess_dataset.validation_dataloader

path = "./models/trial_model.pth"
train_losses = []
val_losses = []
train_accuracy = []
val_accuracy = []

# Check if a checkpoint exists to resume training
if os.path.exists(path):
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint["model_state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
  train_losses = checkpoint["train_loss"]
  val_losses = checkpoint["valid_loss"]
  train_accuracy = checkpoint['t_accuracy']
  val_accuracy = checkpoint['v_accuracy']
  start_epoch = checkpoint["epoch"] + 1  # Start from the next epoch after the loaded checkpoint
  print("Resume training from epoch", start_epoch)
else:
  start_epoch = 1

# Define accuracy calculation function
def calculate_accuracy(predictions, labels):
    predicted_labels = (predictions > 0.5).int()
    correct = (predicted_labels == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

total_epochs = 50
for epoch in range(start_epoch, total_epochs + 1):
    model.train()
    s_time = time.time()
    train_loss = 0.0
    val_loss = 0.0
    train_correct = 0
    val_correct = 0

    for batch_idx, (image, label) in enumerate(train_dataloader):
        prediction = model(image)
        prediction = prediction.squeeze(1)
        tloss = loss_function(prediction, label.float())
        train_loss += tloss.item()
        train_correct += calculate_accuracy(prediction, label)

        optimizer.zero_grad()
        tloss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    t_accuracy = train_correct / len(train_dataloader)

    train_losses.append(train_loss)
    train_accuracy.append(t_accuracy)

    with torch.no_grad():
        for batch, (img, lbl) in enumerate(val_dataloader):
            prediction = model(img)
            prediction = prediction.squeeze(1)
            vloss = loss_function(prediction, lbl.float())
            val_loss += vloss.item()
            val_correct += calculate_accuracy(prediction, lbl)

    val_loss /= len(val_dataloader)
    v_accuracy = val_correct / len(val_dataloader)
    val_losses.append(val_loss)
    val_accuracy.append(v_accuracy)

    print(f'Epoch [{epoch}/{total_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {t_accuracy * 100:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {v_accuracy * 100:.2f}%, Time: {time.time() - s_time:.2f} sec')

    # Save model checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses,
        'valid_loss': val_losses,
        't_accuracy': train_accuracy,
        'v_accuracy': val_accuracy,
    }
    torch.save(checkpoint, path)

