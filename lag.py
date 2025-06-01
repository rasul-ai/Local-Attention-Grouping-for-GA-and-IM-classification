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
        sub_module_outputs = []
        for i in range(10):
            crop_transform = transforms.RandomCrop(64)
            cropped_image = crop_transform(image)
            output = self.cnn_se_model(cropped_image)
            sub_module_outputs.append(output)

        ensemble_sub_module = torch.cat(sub_module_outputs, dim=1)

        resize_data = transforms.Resize((224, 224))
        resized_image = resize_data(image)
        backbone_output = self.backbone(resized_image)

        concatenated_output = torch.cat((backbone_output, ensemble_sub_module), dim=1)

        lag_output = self.classifier(concatenated_output)
        return lag_output