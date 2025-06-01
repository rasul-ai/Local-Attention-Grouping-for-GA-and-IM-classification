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

    def forward(self, image):
        # Ensure image is 4D (B, C, H, W); if 3D (C, H, W), add batch dim
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (1, C, H, W)

        B, C, H, W = image.shape
        num_horizontal_parts = 4
        h_step = H // num_horizontal_parts

        sub_module_outputs = []

        # Divide image into horizontal parts, no resizing
        for i in range(num_horizontal_parts):
            h_start = i * h_step
            h_end = (i + 1) * h_step if i < num_horizontal_parts - 1 else H  # handle last strip
            part = image[:, :, h_start:h_end, :]  # shape (B, C, h_part, W)

            output = self.cnn_se_model(part)  # Expect CNNwithSE to accept variable height input
            sub_module_outputs.append(output)

        # Concatenate all horizontal part outputs along channel dimension
        ensemble_sub_module = torch.cat(sub_module_outputs, dim=1)  # shape depends on CNNwithSE output

        # Global feature from SEResNet backbone
        resized_image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        backbone_output = self.backbone(resized_image)  # (B, 64, 8, 8)

        # Resize ensemble output to match spatial size of backbone output (e.g., 8x8)
        if ensemble_sub_module.shape[2:] != backbone_output.shape[2:]:
            ensemble_sub_module = F.adaptive_avg_pool2d(ensemble_sub_module, backbone_output.shape[2:])

        # Concatenate local + global features
        concatenated_output = torch.cat((backbone_output, ensemble_sub_module), dim=1)
        # print(concatenated_output.shape)

        lag_output = self.classifier(concatenated_output)  # (B, 1)
        return lag_output
