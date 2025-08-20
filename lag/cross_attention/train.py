import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from preprocess import preprocess_dataset
from multi_head_attention import LAG

# ------------------- Setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

model = LAG().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, min_lr=1e-6)

# ------------------- Paths -------------------
root_dir = "./models_cross_attention_original"
os.makedirs(root_dir, exist_ok=True)

model_path = os.path.join(root_dir, "model.pth")
csv_path = os.path.join(root_dir, "metrics_log.csv")
loss_plot_path = os.path.join(root_dir, "loss_curve.png")
acc_plot_path = os.path.join(root_dir, "accuracy_curve.png")

# ------------------- Load Data -------------------
data = preprocess_dataset()
train_loader = data.train_dataloader
val_loader = data.validation_dataloader

# ------------------- Resume Checkpoint -------------------
train_losses, val_losses = [], []
train_accuracy, val_accuracy = [], []
metrics_list = []
start_epoch = 1
best_val_loss = float("inf")
early_stop_counter = 0
patience = 50

# Load previous CSV metrics if exists
if os.path.exists(csv_path):
    old_df = pd.read_csv(csv_path)
    metrics_list = old_df.to_dict('records')
    start_epoch = old_df['Epoch'].max() + 1
    print(f"Resuming metrics logging from epoch {start_epoch}")

# Load checkpoint if exists
if os.path.exists(model_path):
    print("Checkpoint found. Loading...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    train_losses = checkpoint.get("train_loss", [])
    val_losses = checkpoint.get("valid_loss", [])
    train_accuracy = checkpoint.get("t_accuracy", [])
    val_accuracy = checkpoint.get("v_accuracy", [])
    best_val_loss = min(val_losses) if val_losses else float("inf")
    print(f"Resumed from epoch {start_epoch}")

# ------------------- Training Loop -------------------
total_epochs = 100
for epoch in range(start_epoch, total_epochs + 1):
    model.train()
    s_time = time.time()
    train_loss = 0.0
    val_epoch_loss = 0.0
    train_preds, train_labels = [], []
    val_preds, val_labels = [], []

    # --- Training ---
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels.long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        train_loss += loss.item()
        train_preds += torch.argmax(outputs, dim=1).cpu().tolist()
        train_labels += labels.cpu().tolist()

    # --- Validation ---
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels.long())

            val_epoch_loss += loss.item()
            val_preds += torch.argmax(outputs, dim=1).cpu().tolist()
            val_labels += labels.cpu().tolist()

    # --- Metrics ---
    train_loss /= len(train_loader)
    val_epoch_loss /= len(val_loader)

    train_acc = precision_score(train_labels, train_preds, average='macro', zero_division=0)
    val_acc = precision_score(val_labels, val_preds, average='macro', zero_division=0)

    train_precision = precision_score(train_labels, train_preds, average='macro', zero_division=0)
    train_recall = recall_score(train_labels, train_preds, average='macro', zero_division=0)
    train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)

    val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
    val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

    scheduler.step(val_epoch_loss)

    train_losses.append(train_loss)
    val_losses.append(val_epoch_loss)
    train_accuracy.append(train_acc)
    val_accuracy.append(val_acc)

    print(f"\nEpoch [{epoch}/{total_epochs}] - Time: {time.time() - s_time:.2f}s")
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
    print(f"Val   Loss: {val_epoch_loss:.4f}, Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    # --- Save Checkpoint ---
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses,
        'valid_loss': val_losses,
        't_accuracy': train_accuracy,
        'v_accuracy': val_accuracy,
    }, model_path)

    # --- Append Metrics ---
    metrics_list.append({
        'Epoch': epoch,
        'Train Loss': train_loss,
        'Val Loss': val_epoch_loss,
        'Train Accuracy': train_acc,
        'Val Accuracy': val_acc,
        'Train Precision': train_precision,
        'Val Precision': val_precision,
        'Train Recall': train_recall,
        'Val Recall': val_recall,
        'Train F1': train_f1,
        'Val F1': val_f1,
    })

    pd.DataFrame(metrics_list).to_csv(csv_path, index=False)

    # --- Early Stopping ---
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"EarlyStopping counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

# ------------------- Save Curves -------------------
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("Loss Curve")
plt.savefig(loss_plot_path)
plt.close()

plt.figure()
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Accuracy Curve")
plt.savefig(acc_plot_path)
plt.close()
