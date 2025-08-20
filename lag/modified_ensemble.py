import os
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

# --- CrossAttention Module ---
class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, output_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads # Ensure output_dim is divisible by num_heads

        if output_dim % num_heads != 0:
            raise ValueError(f"output_dim ({output_dim}) must be divisible by num_heads ({num_heads})")

        self.query_proj = nn.Linear(query_dim, output_dim)
        self.key_proj = nn.Linear(key_dim, output_dim)
        self.value_proj = nn.Linear(value_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)

    def forward(self, query, key, value):
        # query, key, value are typically (batch_size, sequence_length, feature_dim)

        B, N_q, D_q = query.shape
        _, N_kv, D_kv = key.shape # N_kv should be same as value's second dim

        # Project to output_dim and reshape for multi-head attention
        queries = self.query_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, N_q, head_dim)
        keys = self.key_proj(key).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)     # (B, num_heads, N_kv, head_dim)
        values = self.value_proj(value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)   # (B, num_heads, N_kv, head_dim)

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5) # (B, num_heads, N_q, N_kv)
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_values = torch.matmul(attention_weights, values) # (B, num_heads, N_q, head_dim)

        # Concatenate heads and project back to original dimension
        attended_values = attended_values.transpose(1, 2).contiguous().view(B, N_q, self.num_heads * self.head_dim)
        output = self.out_proj(attended_values)
        return output

# --- SEBlock (provided) ---
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
        y = self.avg_pool(x).view(batch_size, channel)
        y = self.fc(y).view(batch_size, channel, 1, 1)
        return x * y.expand_as(x)

# --- CNNwithSE (provided) ---
class CNNwithSE(nn.Module):
    def __init__(self):
        super(CNNwithSE, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.drop3 = nn.Dropout(0.3)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

        self.se = SEBlock(64)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        x = self.maxpool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.drop3(x)
        x = self.maxpool(x)

        x = self.se(x)
        return x # Output shape will be (B, 64, H_out, W_out) after 3 maxpools (image H, W / 8)

# --- SEResNet (provided, with corrected stride modification based on common practice) ---
class SEResNet(nn.Module):
    def __init__(self):
        super(SEResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        # Modify stride of layer4 to ensure spatial dimensions are not reduced further
        # Typical resnet50 output for layer3 is (B, 1024, H/16, W/16)
        # layer4 would typically reduce this to H/32, W/32.
        # To get 8x8 spatial resolution from 224x224 input, we need H/28, W/28.
        # This means 224 / 28 = 8.
        # Initial resnet layers:
        # conv1, maxpool: 224/4 = 56
        # layer1: 56/1 = 56
        # layer2: 56/2 = 28
        # layer3: 28/2 = 14
        # layer4: 14/2 = 7 (this is the default)
        # To get 8x8 from 224x224, we need total stride of 224/8 = 28.
        # Default ResNet50 strides: conv1 (2), maxpool (2), layer1 (1), layer2 (2), layer3 (2), layer4 (2) = 2*2*1*2*2*2 = 64
        # If avgpool target is 8x8 from 224x224, it means global features are desired at a spatial resolution.

        # Let's adjust the stride in layer4 so it doesn't downsample.
        # The original resnet50 layer4 has a stride of 2 in its first block.
        # By setting it to 1, we prevent further downsampling.
        # This will make the spatial size after layer4 same as after layer3.
        # Original ResNet: Output of layer3 (B, 1024, 14, 14) for 224x224 input.
        # With stride=1 in layer4's first conv and downsample: Output of layer4 (B, 2048, 14, 14)
        self.resnet.layer4[0].conv2.stride = (1, 1)
        self.resnet.layer4[0].downsample[0].stride = (1, 1)


        self.additional_conv = nn.Conv2d(2048, 64, kernel_size=1)
        self.bn = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.3)

        # AdaptiveAvgPool2d to get a consistent 8x8 spatial output for global features
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.se = SEBlock(64)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) # (B, 64, 56, 56)

        x = self.resnet.layer1(x) # (B, 256, 56, 56)
        x = self.resnet.layer2(x) # (B, 512, 28, 28)
        x = self.resnet.layer3(x) # (B, 1024, 14, 14)
        x = self.resnet.layer4(x) # (B, 2048, 14, 14) due to stride modification

        x = self.additional_conv(x) # (B, 64, 14, 14)
        x = self.bn(x)
        x = self.dropout(x)

        x = self.avgpool(x) # (B, 64, 8, 8) - This is the desired output spatial size
        x = self.se(x)
        return x


# --- LAG (modified for Cross-Attention and 8-class classifier) ---
class LAG(nn.Module):
    def __init__(self, num_classes=8): # Added num_classes argument
        super(LAG, self).__init__()
        self.cnn_se_model = CNNwithSE()
        self.backbone = SEResNet()

        # Determine feature dimensions for CrossAttention
        # CNNwithSE output is (B, 64, H_out, W_out)
        # SEResNet output is (B, 64, 8, 8)
        self.backbone_output_channels = 64
        # CNNwithSE output channels for ONE part are 64.
        # We divide image into 4 horizontal parts, then concatenate them.
        # So, LOCAL_FEATURE_CHANNELS = 4 * 64 = 256
        self.local_feature_channels = 64 * 4 # 256

        # Cross-attention where local features query global features
        self.local_to_global_attention = CrossAttention(
            query_dim=self.local_feature_channels, # Local features as query
            key_dim=self.backbone_output_channels,   # Global features as key
            value_dim=self.backbone_output_channels,  # Global features as value
            output_dim=self.backbone_output_channels # Output dimension of the attention module (64)
        )

        # Cross-attention where global features query local features
        self.global_to_local_attention = CrossAttention(
            query_dim=self.backbone_output_channels, # Global features as query
            key_dim=self.local_feature_channels,   # Local features as key
            value_dim=self.local_feature_channels,  # Local features as value
            output_dim=self.local_feature_channels # Output dimension of the attention module (256)
        )

        # The combined feature dimension after concatenation of both attention outputs
        # is (self.backbone_output_channels + self.local_feature_channels) = 64 + 256 = 320
        self.final_classifier_input_dim = self.backbone_output_channels + self.local_feature_channels

        # Initialize Classifier with the correct input dimension and num_classes
        self.classifier = Classifier(in_features=self.final_classifier_input_dim, num_classes=num_classes)


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

            output = self.cnn_se_model(part)
            sub_module_outputs.append(output)

        # Concatenate all horizontal part outputs along channel dimension
        # Output will be (B, 4 * 64, H_out, W_out) -> (B, 256, H_out, W_out)
        ensemble_sub_module = torch.cat(sub_module_outputs, dim=1)

        # Global feature from SEResNet backbone
        resized_image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        backbone_output = self.backbone(resized_image)  # (B, 64, 8, 8)

        # --- Prepare features for Cross-Attention ---
        # Apply Global Average Pooling to get (B, Channels) vectors
        global_features_pooled = F.adaptive_avg_pool2d(backbone_output, (1, 1)).squeeze(-1).squeeze(-1) # (B, 64)
        local_features_pooled = F.adaptive_avg_pool2d(ensemble_sub_module, (1, 1)).squeeze(-1).squeeze(-1) # (B, 256)

        # Unsqueeze to add a sequence dimension (N=1) for the attention module
        global_features_seq = global_features_pooled.unsqueeze(1) # (B, 1, 64)
        local_features_seq = local_features_pooled.unsqueeze(1)   # (B, 1, 256)

        # --- Apply Cross-Attention ---
        # Local features querying global features
        attended_local_from_global = self.local_to_global_attention(
            query=local_features_seq,
            key=global_features_seq,
            value=global_features_seq
        ) # Output: (B, 1, 64)

        # Global features querying local features
        attended_global_from_local = self.global_to_local_attention(
            query=global_features_seq,
            key=local_features_seq,
            value=local_features_seq
        ) # Output: (B, 1, 256)

        # Squeeze out the sequence dimension (1) to get (B, D)
        attended_local_from_global = attended_local_from_global.squeeze(1) # (B, 64)
        attended_global_from_local = attended_global_from_local.squeeze(1) # (B, 256)

        # Concatenate the outputs from both cross-attention modules
        superimposed_features = torch.cat(
            (attended_local_from_global, attended_global_from_local),
            dim=1
        ) # (B, 64 + 256) = (B, 320)

        # Pass through the classifier
        lag_output = self.classifier(superimposed_features) # (B, num_classes)
        return lag_output


# --- Classifier (Modified for 8 classes and flexible input_dim) ---
class Classifier(nn.Module):
    def __init__(self, in_features, num_classes=8): # Added in_features and num_classes
        super(Classifier, self).__init__()
        # Removed flatten as the input to Classifier will now be (B, in_features)
        # from the concatenated attention outputs, not a feature map.
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features, 512), # Reduced first FC layer to match common practice post-attention
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256), # Adjusted for simpler architecture
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)  # Output layer for 8-class classification
        )

    def forward(self, x):
        # x is already (B, in_features), no need to flatten a 4D tensor.
        x = self.fc_layers(x)
        return x