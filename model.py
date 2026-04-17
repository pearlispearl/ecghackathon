import torch
from torch import nn
from torchvision import models


class ECGEfficientNet(nn.Module):
    """
    EfficientNet-B0 fine-tuned for 5-class CAC score classification
    from ECG PNG images.
    """
    def __init__(self, num_classes=5, dropout=0.4):
        super().__init__()

        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # Freeze early layers, fine-tune later features + classifier
        for param in backbone.features.parameters():
            param.requires_grad = False

        # Unfreeze last 2 feature blocks
        for block in backbone.features[-2:]:
            for param in block.parameters():
                param.requires_grad = True

        in_features = backbone.classifier[1].in_features

        backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

        self.backbone = backbone

    def _get_name(self):
        return "ECGEfficientNetB0"

    def forward(self, x):
        return self.backbone(x)