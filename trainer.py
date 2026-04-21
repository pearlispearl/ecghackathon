# Import Python Packages
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
import torchvision.transforms.v2 as transforms_v2
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score,
    mean_squared_error )
from PIL import Image
from ecg_image_dataset import ECGImageDataset 
from dl_utils import train_one_epoch, test
from wave_loader import load_waveforms
from wave_image_dataset import ECGWaveToImageDataset
from torch.utils.data import ConcatDataset


# Hyperparameters
learning_rate = 1e-4
batch_size = 16
epochs = 30
csv_file_path1 = "data/patient_info_dataset.csv"
csv_file_path2 = "data/patient_info_dataset2.csv"
image_dir_path = "data/ECG Signal Image"

# Dataset
transform = transforms_v2.Compose([
    transforms_v2.ToImage(),
    transforms_v2.Resize((128, 256)), 
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

df1 = pd.read_csv(csv_file_path1)
df2 = pd.read_csv(csv_file_path2)
df_full = pd.concat([df1, df2], ignore_index=True)
wave_dict = load_waveforms("data/matched_cta_ecg_waves.tsv")
summary_df = pd.read_excel("data/matched_summary.xlsx")
image_dataset = ECGImageDataset(df_full, image_dir_path, transform=transform)

wave_dataset = ECGWaveToImageDataset(
    summary_df,
    wave_dict,
    transform=transform
)

full_dataset = ConcatDataset([image_dataset, wave_dataset])

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, valid_ds = random_split(full_dataset, [train_size, val_size])

test_ds = valid_ds

train_dl = DataLoader(train_ds, batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size, shuffle=False)


# Model
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
    
from model import ECGResNet
model = ECGResNet().to(device)

print(model)


batch_x, batch_y = next(iter(train_dl))
y_hat = model(batch_x.to(device))

# Model Training
# Setup tensorboard
writer = SummaryWriter(f'./runs/ECG_CAC_{model._get_name()}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
# Specify loss function
weights = torch.tensor([1.0, 2.0, 1.5, 2.5, 4.0]).to(device)
loss_fn = nn.CrossEntropyLoss(weight=weights)
# Specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Training loop
best_vloss = 100000.
for epoch in range(epochs):
    print(f"Epoch {epoch+1} / {epochs}")
    
    # Run train_one_epoch 
    train_one_epoch(train_dl, model, loss_fn, optimizer, epoch, device, writer)
    
    # Evaluate the model on the training set 
    train_loss, train_y_preds, train_y_trues = test(train_dl, model, loss_fn, device)
    
    # Evaluate the model on the validation set 
    val_loss, val_y_preds, val_y_trues = test(valid_dl, model, loss_fn, device)
    
    # Performance metrics for training set
    train_perf = {
        'accuracy': multiclass_accuracy(train_y_preds, train_y_trues, num_classes=5).item(),
        'f1': multiclass_f1_score(train_y_preds, train_y_trues, num_classes=5).item(),
    }
    
    # Performance metrics for validation set
    val_perf = {
        'accuracy': multiclass_accuracy(val_y_preds, val_y_trues, num_classes=5).item(),
        'f1': multiclass_f1_score(val_y_preds, val_y_trues, num_classes=5).item(),
    }
    
    # Log model training performance
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/valid', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_perf['accuracy'], epoch)
    writer.add_scalar('Accuracy/valid', val_perf['accuracy'], epoch)
    writer.add_scalar('F1-Score/train', train_perf['f1'], epoch)
    writer.add_scalar('F1-Score/valid', val_perf['f1'], epoch)

    # Track best performance, and save the model's state
    if val_loss < best_vloss:
        best_vloss = val_loss
        torch.save(model.state_dict(), 'model_best_vloss.pth')
        print('Saved best model to model_best_vloss.pth')
print("Done!")


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