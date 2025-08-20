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


class LAG(nn.Module):
    def __init__(self):
        super(LAG, self).__init__()
        self.cnn_se_model = CNNwithSE()
        self.backbone = SEResNet()
        self.classifier = Classifier()

        # Project local features to match global channels (assume CNNwithSE → 256)
        self.local_proj = nn.Conv2d(256, 64, kernel_size=1)

        # Residual gate (channel-wise, like SE)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.Sigmoid()
        )

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

        local_feat = torch.cat(sub_module_outputs, dim=1)  # (B, 256, h, w)

        resized_image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        global_feat = self.backbone(resized_image)  # (B, 64, h, w)

        if local_feat.shape[2:] != global_feat.shape[2:]:
            local_feat = F.adaptive_avg_pool2d(local_feat, global_feat.shape[2:])

        local_feat = self.local_proj(local_feat)  # Project to (B, 64, h, w)

        # Residual gate: weight (global - local)
        gate = self.gate(local_feat)  # (B, 64, 1, 1)

        # Residual fusion
        fused = local_feat + gate * (global_feat - local_feat)

        return self.classifier(fused)
