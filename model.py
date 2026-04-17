import torch
from torch import nn
from torchvision import models


class ECGResNet(nn.Module):
    """
    EfficientNet-B0 fine-tuned for 5-class CAC score classification from ECG images.

    Strategy:
        - Lighter backbone vs ResNet152 — far fewer parameters, less overfitting on ~90 images
        - Freeze early feature blocks (0-5), fine-tune later blocks (6-8) + classifier head
        - Moderate dropout to regularize the small head
    """
    def __init__(self, num_classes=5, dropout=0.4):
        super().__init__()

        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Freeze blocks 0-5, fine-tune blocks 6-8
        for name, param in effnet.named_parameters():
            block_id = None
            if name.startswith('features.'):
                parts = name.split('.')
                try:
                    block_id = int(parts[1])
                except (IndexError, ValueError):
                    pass
            if block_id is not None and block_id <= 5:
                param.requires_grad = False

        in_features = effnet.classifier[1].in_features  # 1280
        effnet.classifier = nn.Identity()
        self.backbone = effnet

        # Compact classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

    def _get_name(self):
        return "ECGEfficientNetB0"

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
