class LAG(nn.Module):
    def __init__(self):
        super(LAG, self).__init__()
        self.cnn_se_model = CNNwithSE()
        self.backbone = SEResNet()
        self.classifier = Classifier()

        self.local_proj = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)  # adjust input if needed
        self.global_proj = nn.Identity()  # assumes backbone output is already (B, 64, H, W)

        # Local feature gate: squeeze-excitation style
        self.local_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.Sigmoid()
        )

        # Global feature gate: squeeze-excitation style
        self.global_gate = nn.Sequential(
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

        ensemble_sub_module = torch.cat(sub_module_outputs, dim=1)  # (B, 256, h, w)

        resized_image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        backbone_output = self.backbone(resized_image)  # (B, 64, h, w)

        # Match spatial sizes
        if ensemble_sub_module.shape[2:] != backbone_output.shape[2:]:
            ensemble_sub_module = F.adaptive_avg_pool2d(ensemble_sub_module, backbone_output.shape[2:])

        # Project local features to match global channel size
        ensemble_proj = self.local_proj(ensemble_sub_module)  # (B, 64, h, w)
        backbone_proj = self.global_proj(backbone_output)     # (B, 64, h, w)

        # Apply separate gates
        gate_local = self.local_gate(ensemble_proj)           # (B, 64, 1, 1)
        gate_global = self.global_gate(backbone_proj)         # (B, 64, 1, 1)

        fused = gate_local * ensemble_proj + gate_global * backbone_proj  # (B, 64, h, w)

        output = self.classifier(fused)  # Final prediction
        return output
