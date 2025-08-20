import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score

from preprocess import preprocess_dataset
from multi_head_attention import LAG

# ------------------- Setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Load model
model = LAG().to(device)
model_path = "./models_cross_attention_original/model.pth"
assert os.path.exists(model_path), f"No checkpoint found at {model_path}"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ------------------- Load Test Data -------------------
data = preprocess_dataset()
test_loader = data.test_dataloader

# ------------------- Inference -------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

# ------------------- Metrics -------------------
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print("\nTest Set Evaluation:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ------------------- Confusion Matrix -------------------
cm = confusion_matrix(all_labels, all_preds)
# class_names = ['cecum', 'barretts', 'esophagitis-b-d', 'retroflex-rectum', 'retroflex-stomach', 'bbps-2-3', 'bbps-0-1', 'dyed-resection-margins', 'ulcerative-colitis-grade-1-2', 'ulcerative-colitis-grade-2', 'ileum', 'pylorus', 'polyps', 'hemorrhoids', 'esophagitis-a', 'dyed-lifted-polyps', 'z-line', 'ulcerative-colitis-grade-0-1', 'ulcerative-colitis-grade-2-3', 'barretts-short-segment', 'ulcerative-colitis-grade-1', 'ulcerative-colitis-grade-3', 'impacted-stool']
class_names = ['cecum', 'retroflex-stomach', 'bbps-2-3', 'bbps-0-1', 'dyed-resection-margins', 'pylorus', 'polyps', 'z-line']


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("./models_cross_attention_original/confusion_matrix.png")
plt.close()

# Optionally save metrics to CSV
metrics_path = "./models_cross_attention_original/test_metrics.csv"
pd.DataFrame([{
    'Accuracy': acc,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}]).to_csv(metrics_path, index=False)

