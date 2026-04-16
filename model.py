import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class ECGResNet(nn.Module):
    """
    ResNet50 fine-tuned for 5-class CAC score classification from ECG PNG images.
 
    Strategy:
        - Freeze early ResNet layers (layers 1-3) to preserve ImageNet features
        - Fine-tune only layer4 + classifier head
        - Strong dropout in head to prevent overfitting on small medical datasets
    """
    def __init__(self, num_classes=5, dropout=0.5):
        super().__init__()
 
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
 
        # Freeze everything except layer4
        for name, param in resnet.named_parameters():
            if not (name.startswith("layer4") or name.startswith("fc")):
                param.requires_grad = False
 
        in_features = resnet.fc.in_features  # 2048
        resnet.fc = nn.Identity()
        self.backbone = resnet
 
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
 
    def _get_name(self):
        return "ECGResNet50"
 
    def forward(self, x):
        features = self.backbone(x)        
        return self.classifier(features) 
 