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

# Assuming these are defined in 'ensemble.py'
from ensemble import CNNwithSE, SEResNet, Classifier


# Define a CrossAttention module
class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, output_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

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

class LAG(nn.Module):
    def __init__(self):
        super(LAG, self).__init__()
        self.cnn_se_model = CNNwithSE()
        self.backbone = SEResNet()
        self.classifier = Classifier() # This classifier needs to be adjusted based on the output of cross-attention

        # --- New: Define Cross-Attention Modules ---
        # Assuming backbone_output and ensemble_sub_module will be flattened
        # or pooled to sequence-like features for attention.
        # Let's assume after pooling/flattening, they have a certain feature dimension.
        # We need to know the feature dimensions from SEResNet and CNNwithSE.
        # For SEResNet (backbone_output), if it's (B, C, H, W), the feature dim is C*H*W if flattened, or C if avg pooled.
        # For CNNwithSE (ensemble_sub_module), similarly.

        # Let's assume for simplicity, the feature maps will be globally pooled
        # to get a single vector per image, or flattened to sequences of features.

        # Example dimensions (you'll need to verify these from your actual models):
        # If SEResNet outputs (B, 64, 8, 8), then backbone_feature_dim = 64
        # If CNNwithSE outputs (B, SOME_CHANNELS, 8, 8), then ensemble_feature_dim = SOME_CHANNELS
        # Or, if you flatten (B, C, H, W) to (B, C*H*W), then feature_dim = C*H*W.

        # Let's use the channel dimension as the feature dimension for queries, keys, values
        # after spatial pooling to 1x1.
        self.backbone_output_channels = 64 # From SEResNet (B, 64, 8, 8)
        # You need to determine the output channels of your CNNwithSE.
        # Let's assume CNNwithSE outputs a feature map that, after some processing,
        # can have its channels used as feature dimensions for attention.
        # For the sake of this example, let's estimate or set a placeholder.
        # You need to replace `SOME_CHANNELS_FROM_CNN_SE` with the actual output channels
        # of your `CNNwithSE` module after any necessary pooling to match `backbone_output_channels` if needed,
        # or determine the appropriate input dimension for the CrossAttention.
        # If `ensemble_sub_module` is concatenated, then its channels would be SUM of `cnn_se_model` outputs.
        # Let's assume `ensemble_sub_module` (after adaptive_avg_pool2d) has C_local channels.
        # You'll need to replace this with the actual channel count.
        # For now, let's assume `ensemble_sub_module` will be treated as the 'query' and `backbone_output` as 'key/value'.
        # Or vice-versa. A common approach is to use one as query and the other as key/value, and then switch.

        # Let's refine the feature dimensions. After `adaptive_avg_pool2d`, the features will be (B, C, 1, 1).
        # We'll squeeze them to (B, C).
        # So, query_dim, key_dim, value_dim will be the channel dimensions.

        # Assuming `SEResNet` output channels = 64 (backbone_output.shape[1])
        # Assuming `CNNwithSE` output channels for *one part* is 'X'. If you concatenate 4 parts, it's 4*X channels.
        # Let's call the total channels from `ensemble_sub_module` as `ensemble_total_channels`.
        # You need to find this out from your `CNNwithSE` and how you concatenate.
        # For now, let's use a placeholder `LOCAL_FEATURE_CHANNELS`.

        # Placeholder for `LOCAL_FEATURE_CHANNELS`. You MUST replace this with the actual value.
        # If your `CNNwithSE` outputs `(B, C_part, H_out, W_out)` for one part, and you concatenate 4 parts:
        # `ensemble_sub_module` will be `(B, 4 * C_part, H_out, W_out)`.
        # So, `LOCAL_FEATURE_CHANNELS = 4 * C_part`.
        LOCAL_FEATURE_CHANNELS = 256 # Example: if CNNwithSE outputs 64 channels per part, then 4*64=256

        # Cross-attention where local features query global features
        self.local_to_global_attention = CrossAttention(
            query_dim=LOCAL_FEATURE_CHANNELS, # Local features as query
            key_dim=self.backbone_output_channels,   # Global features as key
            value_dim=self.backbone_output_channels,  # Global features as value
            output_dim=self.backbone_output_channels # Output dimension of the attention module
        )

        # Cross-attention where global features query local features (optional, but often beneficial)
        self.global_to_local_attention = CrossAttention(
            query_dim=self.backbone_output_channels, # Global features as query
            key_dim=LOCAL_FEATURE_CHANNELS,   # Local features as key
            value_dim=LOCAL_FEATURE_CHANNELS,  # Local features as value
            output_dim=LOCAL_FEATURE_CHANNELS # Output dimension of the attention module
        )

        # Final projection layer after attention, if needed, before the classifier.
        # The output of cross-attention will be (B, 1, output_dim) after pooling,
        # so we'll concatenate the two attention outputs.
        # The total dimension for the classifier will be:
        # `local_to_global_attention.output_dim` + `global_to_local_attention.output_dim`
        # which is `self.backbone_output_channels + LOCAL_FEATURE_CHANNELS`.
        # Ensure your Classifier can handle this input dimension.
        # If Classifier expects a specific fixed size after concatenation, you might need an additional FC layer here.
        # For now, assume Classifier can adapt or we'll ensure the input matches.

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

        ensemble_sub_module = torch.cat(sub_module_outputs, dim=1) # (B, C_ensemble, H_out, W_out)

        resized_image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        backbone_output = self.backbone(resized_image) # (B, 64, 8, 8)

        # --- Prepare features for Cross-Attention ---
        # We need to transform the feature maps (B, C, H, W) into sequences (B, N, D) for attention.
        # Option 1: Global Average Pooling to get a single vector per feature map.
        #           This results in (B, C) which can be unsqueezed to (B, 1, C).
        global_features_pooled = F.adaptive_avg_pool2d(backbone_output, (1, 1)).squeeze(-1).squeeze(-1) # (B, 64)
        local_features_pooled = F.adaptive_avg_pool2d(ensemble_sub_module, (1, 1)).squeeze(-1).squeeze(-1) # (B, LOCAL_FEATURE_CHANNELS)

        # Unsqueeze to add a sequence dimension (N=1) for the attention module
        global_features_seq = global_features_pooled.unsqueeze(1) # (B, 1, 64)
        local_features_seq = local_features_pooled.unsqueeze(1)   # (B, 1, LOCAL_FEATURE_CHANNELS)

        # Option 2 (More complex): Flatten spatial dimensions to create sequences of tokens.
        # This is more common in vision transformers. E.g., (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C).
        # However, given your current setup with a single output from the classifier,
        # pooling to (B, C) and then to (B, 1, C) might be simpler and effective initially.
        # If you choose to flatten, make sure to adjust query/key/value dimensions in CrossAttention.

        # --- Apply Cross-Attention ---
        # Local features querying global features
        # Query: local_features_seq (B, 1, LOCAL_FEATURE_CHANNELS)
        # Key/Value: global_features_seq (B, 1, 64)
        attended_local_from_global = self.local_to_global_attention(
            query=local_features_seq,
            key=global_features_seq,
            value=global_features_seq
        ) # Output: (B, 1, self.backbone_output_channels)

        # Global features querying local features
        # Query: global_features_seq (B, 1, 64)
        # Key/Value: local_features_seq (B, 1, LOCAL_FEATURE_CHANNELS)
        attended_global_from_local = self.global_to_local_attention(
            query=global_features_seq,
            key=local_features_seq,
            value=local_features_seq
        ) # Output: (B, 1, LOCAL_FEATURE_CHANNELS)

        # Squeeze out the sequence dimension (1) to get (B, D)
        attended_local_from_global = attended_local_from_global.squeeze(1) # (B, self.backbone_output_channels)
        attended_global_from_local = attended_global_from_local.squeeze(1) # (B, LOCAL_FEATURE_CHANNELS)

        # Concatenate the outputs from both cross-attention modules
        # This combined representation now incorporates information flow in both directions.
        superimposed_features = torch.cat(
            (attended_local_from_global, attended_global_from_local),
            dim=1
        ) # (B, self.backbone_output_channels + LOCAL_FEATURE_CHANNELS)

        # Pass through the classifier
        # IMPORTANT: You need to ensure your Classifier `__init__` is updated
        # to accept `self.backbone_output_channels + LOCAL_FEATURE_CHANNELS` as its input `in_features`.
        lag_output = self.classifier(superimposed_features) # (B, 1)
        return lag_output