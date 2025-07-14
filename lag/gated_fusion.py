class LAG(nn.Module):
    def __init__(self):
        super(LAG, self).__init__()
        self.cnn_se_model = CNNwithSE()
        self.backbone = SEResNet()
        self.classifier = Classifier()

        # Gating network: input = local + global features (channel-wise concat)
        # Output = scalar gate for each channel (between 0 and 1)
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # Global average pooling to reduce spatial dims
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),  # Reduce channels
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),   # Output gate for 64 channels
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

        ensemble_sub_module = torch.cat(sub_module_outputs, dim=1)

        resized_image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        backbone_output = self.backbone(resized_image)  # Shape: (B, 64, H', W')

        # Match spatial dimensions
        if ensemble_sub_module.shape[2:] != backbone_output.shape[2:]:
            ensemble_sub_module = F.adaptive_avg_pool2d(ensemble_sub_module, backbone_output.shape[2:])

        # Ensure both outputs have 64 channels before gated fusion
        if ensemble_sub_module.shape[1] != 64:
            ensemble_sub_module = nn.Conv2d(ensemble_sub_module.shape[1], 64, kernel_size=1).to(image.device)(ensemble_sub_module)

        # Gated Fusion
        fusion_input = torch.cat([ensemble_sub_module, backbone_output], dim=1)  # (B, 128, H', W')
        gate = self.gate_fc(fusion_input)  # (B, 64, 1, 1)

        fused_output = gate * ensemble_sub_module + (1 - gate) * backbone_output  # (B, 64, H', W')

        lag_output = self.classifier(fused_output)
        return lag_output
