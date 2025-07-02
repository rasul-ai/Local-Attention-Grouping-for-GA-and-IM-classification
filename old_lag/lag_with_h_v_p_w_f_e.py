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
        grid_size = 4
        h_step = H // grid_size
        w_step = W // grid_size

        sub_module_outputs = []

        # Divide image into 4x4 grid (16 parts)
        for i in range(grid_size):
            for j in range(grid_size):
                h_start = i * h_step
                h_end = (i + 1) * h_step
                w_start = j * w_step
                w_end = (j + 1) * w_step
                part = image[:, :, h_start:h_end, w_start:w_end]  # shape (B, C, h_step, w_step)

                # Resize each part to 64x64 to match CNNwithSE input expectation
                part_resized = F.interpolate(part, size=(64, 64), mode='bilinear', align_corners=False)

                output = self.cnn_se_model(part_resized)  # output shape (B, 64, 8, 8)
                sub_module_outputs.append(output)

        # Concatenate all 16 part outputs along channel dimension
        ensemble_sub_module = torch.cat(sub_module_outputs, dim=1)  # shape (B, 64*16, 8, 8) = (B, 1024, 8, 8)

        # Global feature from SEResNet backbone
        resized_image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        backbone_output = self.backbone(resized_image)  # shape (B, 64, 8, 8)

        # Concatenate local (parts) + global features
        concatenated_output = torch.cat((backbone_output, ensemble_sub_module), dim=1)  # (B, 1088, 8, 8)
        print(concatenated_output.shape)

        lag_output = self.classifier(concatenated_output)  # (B, 1)
        # print
        return lag_output
