# Import Python Packages
import os
import torch
from torch import nn
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
)
from PIL import Image
from ecg_image_dataset import ECGImageDataset
from dl_utils import train_one_epoch, test


# Device — MPS for Apple Silicon, CUDA, or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# Hyperparameters
learning_rate = 5e-5
batch_size = 16
epochs = 50
csv_file_path1 = "data/patient_info_dataset.csv"
csv_file_path2 = "data/patient_info_dataset2.csv"
image_dir_path = "data/ECG Signal Image"
IMAGE_SIZE = (224, 224)  # match ImageNet pretraining size


# Transforms — augmented for train only
train_transform = transforms_v2.Compose([
    transforms_v2.ToImage(),
    transforms_v2.Resize(IMAGE_SIZE),
    transforms_v2.RandomHorizontalFlip(),
    transforms_v2.RandomRotation(10),
    transforms_v2.ColorJitter(brightness=0.2, contrast=0.2),
    transforms_v2.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms_v2.Compose([
    transforms_v2.ToImage(),
    transforms_v2.Resize(IMAGE_SIZE),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class TransformSubset(Dataset):
    """Wraps a Subset to apply a per-split transform after the split."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


# Load data — no transform here so train/val can get different augmentations
df1 = pd.read_csv(csv_file_path1)
df2 = pd.read_csv(csv_file_path2)
df_full = pd.concat([df1, df2], ignore_index=True)
full_dataset = ECGImageDataset(df_full, image_dir_path, transform=None)

print(full_dataset.df['label'].value_counts())

# Reproducible split
generator = torch.Generator().manual_seed(42)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

train_ds = TransformSubset(train_subset, train_transform)
valid_ds = TransformSubset(val_subset, val_transform)

train_dl = DataLoader(train_ds, batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size, shuffle=False)

# Compute class weights from actual training label distribution
train_labels = [full_dataset[i][1].item() for i in train_subset.indices]
class_counts = np.bincount(train_labels, minlength=5).astype(float)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * 5  # scale so weights sum to num_classes
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Class weights: {weights.tolist()}")


# Model
from model import ECGResNet
model = ECGResNet().to(device)
print(model)

# Sanity check forward pass
batch_x, batch_y = next(iter(train_dl))
_ = model(batch_x.to(device))


# Training setup
writer = SummaryWriter(f'./runs/ECG_CAC_{model._get_name()}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Training loop — checkpoint on best validation F1
best_vf1 = 0.0
for epoch in range(epochs):
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{epochs}  lr={current_lr:.2e}")

    train_one_epoch(train_dl, model, loss_fn, optimizer, epoch, device, writer)

    train_loss, train_y_preds, train_y_trues = test(train_dl, model, loss_fn, device)
    val_loss, val_y_preds, val_y_trues = test(valid_dl, model, loss_fn, device)

    train_perf = {
        'accuracy': multiclass_accuracy(train_y_preds, train_y_trues, num_classes=5).item(),
        'f1': multiclass_f1_score(train_y_preds, train_y_trues, num_classes=5).item(),
    }
    val_perf = {
        'accuracy': multiclass_accuracy(val_y_preds, val_y_trues, num_classes=5).item(),
        'f1': multiclass_f1_score(val_y_preds, val_y_trues, num_classes=5).item(),
    }

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/valid', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_perf['accuracy'], epoch)
    writer.add_scalar('Accuracy/valid', val_perf['accuracy'], epoch)
    writer.add_scalar('F1-Score/train', train_perf['f1'], epoch)
    writer.add_scalar('F1-Score/valid', val_perf['f1'], epoch)
    writer.add_scalar('LR', current_lr, epoch)

    print(f"  Train  loss={train_loss:.4f}  acc={train_perf['accuracy']:.4f}  f1={train_perf['f1']:.4f}")
    print(f"  Val    loss={val_loss:.4f}  acc={val_perf['accuracy']:.4f}  f1={val_perf['f1']:.4f}")

    scheduler.step()

    if val_perf['f1'] > best_vf1:
        best_vf1 = val_perf['f1']
        torch.save(model.state_dict(), 'model_best_f1.pth')
        print(f"  *** Saved best model (val F1={best_vf1:.4f}) ***")

print(f"\nDone! Best val F1: {best_vf1:.4f}")


# Evaluate on test images using best F1 model
model_best = ECGResNet().to(device)
model_best.load_state_dict(torch.load("model_best_f1.pth", map_location=device))
model_best.eval()

test_image_dir = "data/test_image"
if not os.path.exists(test_image_dir):
    print(f"Error: Folder {test_image_dir} not found!")
else:
    image_files = [f for f in os.listdir(test_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    predictions = []

    print(f"Starting prediction on {len(image_files)} images...")

    with torch.no_grad():
        for fname in image_files:
            img_path = os.path.join(test_image_dir, fname)
            image = Image.open(img_path).convert("RGB")

            img_tensor = val_transform(image).unsqueeze(0).to(device)

            outputs = model_best(img_tensor)
            _, predicted = torch.max(outputs, 1)

            hn = fname.split('_')[0]
            predictions.append({"HN": hn, "CaClass": predicted.item()})

    submission_df = pd.DataFrame(predictions)
    submission_df = submission_df.sort_values("HN")
    submission_df.to_csv("submission.csv", index=False)

    print("\n--- DONE ---")
    print(f"Created 'submission.csv' with {len(submission_df)} predictions.")
