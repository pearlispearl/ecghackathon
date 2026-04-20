# Import Python Packages
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.v2 as transforms_v2
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from PIL import Image

from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score,
)

from sklearn.metrics import roc_auc_score

from ecg_image_dataset import ECGImageDataset
from dl_utils import train_one_epoch, test
from model import ECGResNet


# Evaluate on the Test Set
# Load the best model
model_best = ECGResNet().to(device)
model_best.load_state_dict(torch.load("model_best_vloss.pth"))

model_best.eval() 

test_image_dir = "data/test_image"
if not os.path.exists(test_image_dir):
    print(f"Error: Folder {test_image_dir} not found!")
else:
    image_files = [f for f in os.listdir(test_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    predictions = []

    print(f"Starting Prediction on {len(image_files)} images...")

    with torch.no_grad():
        for fname in image_files:
            img_path = os.path.join(test_image_dir, fname)
            image = Image.open(img_path).convert("RGB")
            
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            outputs = model_best(img_tensor)
            _, predicted = torch.max(outputs, 1)
            
            hn = fname.split('_')[0]
            
            predictions.append({
                "HN": hn,
                "CaClass": predicted.item()
            })

    submission_df = pd.DataFrame(predictions)
    submission_df = submission_df.sort_values("HN")
    submission_df.to_csv("submission.csv", index=False)

    print("\n--- DONE ---")
    print(f"Created 'submission.csv' with {len(submission_df)} predictions.")
# ---------------------- End of your code --------------------