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

from ensemble import CNNwithSE, SEResNet, Classifier


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return psi * x  # attention-weighted local features



class LAG(nn.Module):
    def __init__(self):
        super(LAG, self).__init__()
        self.cnn_se_model = CNNwithSE()
        self.backbone = SEResNet()
        self.classifier = Classifier()

        # Project local features to match global
        self.local_proj = nn.Conv2d(256, 64, kernel_size=1)
        self.att_gate = AttentionGate(F_g=64, F_l=64, F_int=32)  # global & local channels = 64

    def forward(self, image):
        if image.dim() == 3:
            image = image.unsqueeze(0)

        B, C, H, W = image.shape
        num_horizontal_parts = 4
        h_step = H // num_horizontal_parts

        sub_module_outputs = []
        for i in range(num_horizontal_parts):
            h_start = i * h_step
            h_end = (i + 1) * h_step if i < num_horizontal_parts - 1 else H
            part = image[:, :, h_start:h_end, :]
            output = self.cnn_se_model(part)
            sub_module_outputs.append(output)

        ensemble_sub_module = torch.cat(sub_module_outputs, dim=1)

        resized_image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        backbone_output = self.backbone(resized_image)

        if ensemble_sub_module.shape[2:] != backbone_output.shape[2:]:
            ensemble_sub_module = F.adaptive_avg_pool2d(ensemble_sub_module, backbone_output.shape[2:])

        # Project local to match global channels
        local_feat = self.local_proj(ensemble_sub_module)

        # Apply attention gate to local features using global context
        attended_local = self.att_gate(backbone_output, local_feat)

        # Fuse: attended local + global
        fused = attended_local + backbone_output

        return self.classifier(fused)
