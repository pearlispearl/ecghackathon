import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.v2 as transforms_v2
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from PIL import Image
from ecg_image_dataset import ECGImageDataset
from dl_utils import train_one_epoch, test


# ---------------------------------------------------------------------------
# Device — MPS (Apple Silicon) first, then CUDA, then CPU
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE = 5e-5
BATCH_SIZE    = 16
EPOCHS        = 50
N_FOLDS       = 5
IMAGE_SIZE    = (224, 224)
MIXUP_ALPHA   = 0.4
N_TTA         = 5          # test-time augmentation passes

CSV1 = "data/patient_info_dataset.csv"
CSV2 = "data/patient_info_dataset2.csv"
IMAGE_DIR = "data/ECG Signal Image"
TEST_DIR  = "data/test_image"


# ---------------------------------------------------------------------------
# Focal Loss — handles class imbalance better than CrossEntropy
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    FL(p) = -(1-p)^gamma * log(p)
    gamma=2 focuses learning on hard / minority-class examples.
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Transforms
# NOTE: RandomHorizontalFlip is intentionally EXCLUDED —
#       flipping an ECG waveform changes its clinical meaning.
# ---------------------------------------------------------------------------
train_transform = transforms_v2.Compose([
    transforms_v2.ToImage(),
    transforms_v2.Resize(IMAGE_SIZE),
    transforms_v2.Grayscale(num_output_channels=3),      # ECG is monochrome
    transforms_v2.RandomRotation(10),
    transforms_v2.ColorJitter(brightness=0.2, contrast=0.2),
    transforms_v2.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms_v2.GaussianBlur(kernel_size=3),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms_v2.RandomErasing(p=0.3),                  # simulate signal dropout
])

val_transform = transforms_v2.Compose([
    transforms_v2.ToImage(),
    transforms_v2.Resize(IMAGE_SIZE),
    transforms_v2.Grayscale(num_output_channels=3),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# TTA transform — mild augmentation used only at inference
tta_transform = transforms_v2.Compose([
    transforms_v2.ToImage(),
    transforms_v2.Resize(IMAGE_SIZE),
    transforms_v2.Grayscale(num_output_channels=3),
    transforms_v2.RandomRotation(5),
    transforms_v2.RandomAffine(degrees=0, translate=(0.02, 0.02)),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
class TransformSubset(Dataset):
    """Wraps a Subset so each split can have its own transform."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


def predict_tta(model, pil_image, n=N_TTA):
    """
    TTA inference: 1 clean pass + (n-1) mildly augmented passes.
    Returns averaged softmax probabilities — shape (1, num_classes).
    """
    model.eval()
    probs = []
    with torch.no_grad():
        t = val_transform(pil_image).unsqueeze(0).to(device)
        probs.append(torch.softmax(model(t), dim=1))
        for _ in range(n - 1):
            t = tta_transform(pil_image).unsqueeze(0).to(device)
            probs.append(torch.softmax(model(t), dim=1))
    return torch.stack(probs).mean(0)


# ---------------------------------------------------------------------------
# Load full dataset (transform=None so each fold applies its own)
# ---------------------------------------------------------------------------
df1 = pd.read_csv(CSV1)
df2 = pd.read_csv(CSV2)
df_full = pd.concat([df1, df2], ignore_index=True)
full_dataset = ECGImageDataset(df_full, IMAGE_DIR, transform=None)

all_labels = [full_dataset[i][1].item() for i in range(len(full_dataset))]
print(f"\nDataset size : {len(full_dataset)}")
print(f"Class counts : {np.bincount(all_labels).tolist()}  (classes 0-4)\n")


# ---------------------------------------------------------------------------
# K-Fold Cross-Validation
# ---------------------------------------------------------------------------
from model import ECGResNet

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_model_paths = []
fold_best_f1s    = []

for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(full_dataset)), all_labels)):
    print(f"\n{'='*55}")
    print(f"  FOLD {fold+1}/{N_FOLDS}   train={len(train_idx)}  val={len(val_idx)}")
    print(f"{'='*55}")

    train_ds = TransformSubset(Subset(full_dataset, train_idx), train_transform)
    valid_ds = TransformSubset(Subset(full_dataset, val_idx),   val_transform)

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  drop_last=False)
    valid_dl = DataLoader(valid_ds, BATCH_SIZE, shuffle=False)

    # Per-fold class weights from actual training label counts
    fold_labels  = [all_labels[i] for i in train_idx]
    class_counts = np.bincount(fold_labels, minlength=5).astype(float)
    cw = 1.0 / (class_counts + 1e-6)
    cw = cw / cw.sum() * 5
    weights = torch.tensor(cw, dtype=torch.float32).to(device)
    print(f"  Class weights: {[f'{w:.3f}' for w in weights.tolist()]}")

    model     = ECGResNet().to(device)
    loss_fn   = FocalLoss(gamma=2.0, weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    writer    = SummaryWriter(f'./runs/fold{fold+1}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    best_vf1   = 0.0
    model_path = f'model_fold_{fold+1}.pth'

    for epoch in range(EPOCHS):
        current_lr = scheduler.get_last_lr()[0]

        train_one_epoch(
            train_dl, model, loss_fn, optimizer,
            epoch, device, writer,
            mixup_alpha=MIXUP_ALPHA
        )

        train_loss, tp, tt = test(train_dl, model, loss_fn, device)
        val_loss,   vp, vt = test(valid_dl, model, loss_fn, device)

        t_f1 = multiclass_f1_score(tp, tt, num_classes=5).item()
        v_f1 = multiclass_f1_score(vp, vt, num_classes=5).item()

        writer.add_scalar('Loss/train',    train_loss, epoch)
        writer.add_scalar('Loss/valid',    val_loss,   epoch)
        writer.add_scalar('F1-Score/train', t_f1,      epoch)
        writer.add_scalar('F1-Score/valid', v_f1,      epoch)
        writer.add_scalar('LR',            current_lr, epoch)

        saved = ''
        if v_f1 > best_vf1:
            best_vf1 = v_f1
            torch.save(model.state_dict(), model_path)
            saved = f'  ✓ saved'

        print(f"  Ep {epoch+1:>2}/{EPOCHS}  lr={current_lr:.1e}  "
              f"train_f1={t_f1:.4f}  val_f1={v_f1:.4f}{saved}")

        scheduler.step()

    writer.close()
    fold_model_paths.append(model_path)
    fold_best_f1s.append(best_vf1)
    print(f"  Fold {fold+1} best val F1: {best_vf1:.4f}")

print(f"\n{'='*55}")
print(f"K-Fold complete!")
print(f"Per-fold val F1 : {[f'{f:.4f}' for f in fold_best_f1s]}")
print(f"Mean  val F1    : {np.mean(fold_best_f1s):.4f} ± {np.std(fold_best_f1s):.4f}")
print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Ensemble + TTA inference on test images
# ---------------------------------------------------------------------------
print("Loading all fold models for ensemble inference...")
fold_models = []
for path in fold_model_paths:
    m = ECGResNet().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    fold_models.append(m)

if not os.path.exists(TEST_DIR):
    print(f"Error: test folder '{TEST_DIR}' not found!")
else:
    image_files = [f for f in os.listdir(TEST_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    predictions = []

    print(f"Running ensemble + TTA on {len(image_files)} test images  "
          f"({N_FOLDS} models × {N_TTA} TTA passes each)...")

    with torch.no_grad():
        for fname in image_files:
            image = Image.open(os.path.join(TEST_DIR, fname)).convert("RGB")

            # Average TTA probs across all fold models
            all_probs = torch.stack([predict_tta(m, image) for m in fold_models])
            predicted = all_probs.mean(0).argmax(1).item()

            predictions.append({"HN": fname.split('_')[0], "CaClass": predicted})

    submission_df = pd.DataFrame(predictions).sort_values("HN")
    submission_df.to_csv("submission.csv", index=False)

    print("\n--- DONE ---")
    print(f"Created 'submission.csv' with {len(submission_df)} predictions.")
