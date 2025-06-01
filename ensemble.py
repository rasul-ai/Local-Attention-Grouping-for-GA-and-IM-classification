import os
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

class CNNwithSE(nn.Module):
    def __init__(self):
        super(CNNwithSE, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(64 * 8 * 8, 2)  # Assuming output size 2 for binary classification
        self.se = SEBlock(64)  # SE block with 64 channels

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        # x = self.flatten(x)
        # x = self.fc(x)
        x = self.se(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel, height, width = x.size()
        y = self.avg_pool(x).view(batch_size, channel) #excitation block
        y = self.fc(y).view(batch_size, channel, 1, 1) #squeeze block
        return x * y.expand_as(x)  #scaling




class SEResNet(nn.Module):
    def __init__(self):
        super(SEResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Modify layers to preserve the output shape
        self.resnet.layer4[0].conv2.stride = (1, 1)
        self.resnet.layer4[0].downsample[0].stride = (1, 1)
        self.resnet.layer4[0].conv3.stride = (1, 1)

        self.additional_conv = nn.Conv2d(2048, 64, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8)) # Reshape the output from (1,64,14,14) to (1,64,8,8)
        self.se = SEBlock(64)  # SE block with 64 channels

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # Additional convolutional layer to adjust the number of output channels
        x = self.additional_conv(x)
        x = self.avgpool(x)
        x = self.se(x)

        return x
    

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(320 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 4)  # Output layer for 2-class classification
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x  # Use CrossEntropyLoss (no activation here)
