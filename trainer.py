import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.v2 as transforms_v2
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score,
)

from ecg_image_dataset import ECGImageDataset
from dl_utils import train_one_epoch, evaluate
from model import ECGEfficientNet


# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# =========================
# Utility functions
# =========================
def safe_multiclass_auc(y_true, probs, num_classes=5):
    y_true_np = y_true.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()

    unique_classes = np.unique(y_true_np)
    if len(unique_classes) < 2:
        return float("nan")

    try:
        auc = roc_auc_score(
            y_true_np,
            probs_np,
            multi_class="ovr",
            average="macro",
            labels=list(range(num_classes))
        )
        return float(auc)
    except ValueError:
        return float("nan")


def apply_class_thresholds(probs, thresholds):
    thresholds = thresholds.to(probs.device)

    passed = probs >= thresholds.unsqueeze(0)
    normalized = probs / thresholds.unsqueeze(0)
    normalized = torch.where(
        passed,
        normalized,
        torch.full_like(normalized, -1.0)
    )

    preds = normalized.argmax(dim=1)
    fallback = probs.argmax(dim=1)
    no_class_passed = ~passed.any(dim=1)
    preds = torch.where(no_class_passed, fallback, preds)

    return preds


def find_best_thresholds_from_logits(
    logits,
    y_true,
    num_classes=5,
    metric_average="macro"
):
    probs = torch.softmax(logits, dim=1).cpu()
    y_true = y_true.cpu()

    candidate_thresholds = torch.arange(0.05, 0.96, 0.05)
    thresholds = torch.full((num_classes,), 0.50)

    best_preds = apply_class_thresholds(probs, thresholds)
    best_f1 = multiclass_f1_score(
        best_preds,
        y_true,
        num_classes=num_classes,
        average=metric_average
    ).item()

    improved = True
    max_rounds = 3
    round_idx = 0

    while improved and round_idx < max_rounds:
        improved = False
        round_idx += 1

        for class_idx in range(num_classes):
            local_best_threshold = thresholds[class_idx].item()
            local_best_f1 = best_f1

            for candidate in candidate_thresholds:
                trial_thresholds = thresholds.clone()
                trial_thresholds[class_idx] = candidate

                trial_preds = apply_class_thresholds(probs, trial_thresholds)
                trial_f1 = multiclass_f1_score(
                    trial_preds,
                    y_true,
                    num_classes=num_classes,
                    average=metric_average
                ).item()

                if trial_f1 > local_best_f1:
                    local_best_f1 = trial_f1
                    local_best_threshold = float(candidate.item())

            if local_best_f1 > best_f1:
                thresholds[class_idx] = local_best_threshold
                best_f1 = local_best_f1
                improved = True

    final_preds = apply_class_thresholds(probs, thresholds)
    final_f1 = multiclass_f1_score(
        final_preds,
        y_true,
        num_classes=num_classes,
        average=metric_average
    ).item()

    return thresholds, final_preds, final_f1


def compute_class_weights_from_labels(labels, num_classes=5):
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def summarize_metric(values):
    arr = np.array(values, dtype=float)
    return arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0


# =========================
# Hyperparameters
# =========================
learning_rate = 5e-5
batch_size = 16
epochs = 100
num_classes = 5
num_folds = 5

patience = 10
min_delta = 1e-4

csv_file_path1 = "data/patient_info_dataset.csv"
csv_file_path2 = "data/patient_info_dataset2.csv"
image_dir_path = "data/ECG Signal Image"


# =========================
# Dataset
# =========================
train_transform = transforms_v2.Compose([
    transforms_v2.ToImage(),
    transforms_v2.Resize((224, 224)),
    transforms_v2.RandomRotation(8),
    transforms_v2.ColorJitter(brightness=0.15, contrast=0.15),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms_v2.Compose([
    transforms_v2.ToImage(),
    transforms_v2.Resize((224, 224)),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

df1 = pd.read_csv(csv_file_path1)
df2 = pd.read_csv(csv_file_path2)
df_full = pd.concat([df1, df2], ignore_index=True)

full_dataset_for_labels = ECGImageDataset(df_full, image_dir_path, transform=eval_transform)
labels = full_dataset_for_labels.df["label"].values
print("Label counts:")
print(full_dataset_for_labels.df["label"].value_counts().sort_index())

# separate datasets so train/valid can use different transforms
train_dataset_full = ECGImageDataset(df_full, image_dir_path, transform=train_transform)
eval_dataset_full = ECGImageDataset(df_full, image_dir_path, transform=eval_transform)

assert len(train_dataset_full) == len(eval_dataset_full) == len(labels), "Dataset size mismatch"


# =========================
# Device
# =========================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# =========================
# TensorBoard
# =========================
writer = SummaryWriter(
    f'./runs/ECG_CAC_EfficientNetB0_KFold_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
)


# =========================
# K-Fold Training
# =========================
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

fold_results = []
best_fold_f1 = -1.0
best_fold_idx = None
best_fold_thresholds = None
best_model_path = None

for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(np.arange(len(labels)), labels), start=1):
    print(f"\n{'=' * 20} FOLD {fold_idx}/{num_folds} {'=' * 20}")

    train_ds = Subset(train_dataset_full, train_idx)
    valid_ds = Subset(eval_dataset_full, valid_idx)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    model = ECGEfficientNet(num_classes=num_classes).to(device)
    print(model)

    fold_train_labels = labels[train_idx]
    class_weights = compute_class_weights_from_labels(fold_train_labels, num_classes=num_classes).to(device)
    print(f"Fold {fold_idx} class weights: {class_weights.tolist()}")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    best_val_f1_thresholded = -1.0
    best_thresholds = torch.full((num_classes,), 0.50)
    early_stop_counter = 0

    fold_model_path = f"model_fold{fold_idx}_best_f1_threshold.pth"
    fold_thresholds_pt_path = f"best_thresholds_fold{fold_idx}.pt"
    fold_thresholds_json_path = f"best_thresholds_fold{fold_idx}.json"

    for epoch in range(epochs):
        print(f"\nFold {fold_idx} - Epoch {epoch + 1}/{epochs}")

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {current_lr:.8f}")

        train_one_epoch(
            train_dl, model, loss_fn, optimizer,
            epoch, device, writer=writer, fold_idx=fold_idx
        )

        train_loss, train_y_preds, train_y_trues, train_logits = evaluate(
            train_dl, model, loss_fn, device, return_logits=True
        )
        train_probs = torch.softmax(train_logits, dim=1)

        train_acc = multiclass_accuracy(
            train_y_preds, train_y_trues, num_classes=num_classes
        ).item()
        train_f1 = multiclass_f1_score(
            train_y_preds, train_y_trues, num_classes=num_classes, average="macro"
        ).item()
        train_auc = safe_multiclass_auc(
            train_y_trues, train_probs, num_classes=num_classes
        )

        val_loss, val_y_preds, val_y_trues, val_logits = evaluate(
            valid_dl, model, loss_fn, device, return_logits=True
        )
        val_probs = torch.softmax(val_logits, dim=1)

        val_acc_argmax = multiclass_accuracy(
            val_y_preds, val_y_trues, num_classes=num_classes
        ).item()
        val_f1_argmax = multiclass_f1_score(
            val_y_preds, val_y_trues, num_classes=num_classes, average="macro"
        ).item()
        val_auc = safe_multiclass_auc(
            val_y_trues, val_probs, num_classes=num_classes
        )

        threshold_tensor, val_y_preds_thresholded, val_f1_thresholded = find_best_thresholds_from_logits(
            val_logits, val_y_trues, num_classes=num_classes, metric_average="macro"
        )

        val_acc_thresholded = multiclass_accuracy(
            val_y_preds_thresholded, val_y_trues, num_classes=num_classes
        ).item()

        # AUC depends on probabilities, not thresholds
        val_auc_thresholded = val_auc

        writer.add_scalar(f"Fold_{fold_idx}/Loss/train", train_loss, epoch)
        writer.add_scalar(f"Fold_{fold_idx}/Loss/valid", val_loss, epoch)

        writer.add_scalar(f"Fold_{fold_idx}/Accuracy/train", train_acc, epoch)
        writer.add_scalar(f"Fold_{fold_idx}/Accuracy/valid_argmax", val_acc_argmax, epoch)
        writer.add_scalar(f"Fold_{fold_idx}/Accuracy/valid_thresholded", val_acc_thresholded, epoch)

        writer.add_scalar(f"Fold_{fold_idx}/F1/train", train_f1, epoch)
        writer.add_scalar(f"Fold_{fold_idx}/F1/valid_argmax", val_f1_argmax, epoch)
        writer.add_scalar(f"Fold_{fold_idx}/F1/valid_thresholded", val_f1_thresholded, epoch)

        if not np.isnan(train_auc):
            writer.add_scalar(f"Fold_{fold_idx}/AUC/train", train_auc, epoch)
        if not np.isnan(val_auc):
            writer.add_scalar(f"Fold_{fold_idx}/AUC/valid_argmax", val_auc, epoch)
            writer.add_scalar(f"Fold_{fold_idx}/AUC/valid_thresholded", val_auc_thresholded, epoch)

        writer.add_scalar(f"Fold_{fold_idx}/LR", current_lr, epoch)

        for class_idx in range(num_classes):
            writer.add_scalar(
                f"Fold_{fold_idx}/Threshold/class_{class_idx}",
                threshold_tensor[class_idx].item(),
                epoch
            )

        print(
            f"Train -> Loss: {train_loss:.4f}, ACC: {train_acc:.4f}, "
            f"F1: {train_f1:.4f}, AUC: {train_auc:.4f}" if not np.isnan(train_auc)
            else f"Train -> Loss: {train_loss:.4f}, ACC: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: nan"
        )

        print(
            f"Val(argmax) -> Loss: {val_loss:.4f}, ACC: {val_acc_argmax:.4f}, "
            f"F1: {val_f1_argmax:.4f}, AUC: {val_auc:.4f}" if not np.isnan(val_auc)
            else f"Val(argmax) -> Loss: {val_loss:.4f}, ACC: {val_acc_argmax:.4f}, F1: {val_f1_argmax:.4f}, AUC: nan"
        )

        print(
            f"Val(thresholded) -> ACC: {val_acc_thresholded:.4f}, "
            f"F1: {val_f1_thresholded:.4f}, AUC: {val_auc_thresholded:.4f}" if not np.isnan(val_auc_thresholded)
            else f"Val(thresholded) -> ACC: {val_acc_thresholded:.4f}, F1: {val_f1_thresholded:.4f}, AUC: nan"
        )
        print(f"Best thresholds this epoch: {threshold_tensor.tolist()}")

        scheduler.step(val_f1_thresholded)

        if val_f1_thresholded > best_val_f1_thresholded + min_delta:
            best_val_f1_thresholded = val_f1_thresholded
            best_thresholds = threshold_tensor.clone()
            early_stop_counter = 0

            torch.save(model.state_dict(), fold_model_path)
            torch.save(best_thresholds, fold_thresholds_pt_path)

            with open(fold_thresholds_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fold": fold_idx,
                        "best_val_f1_thresholded": best_val_f1_thresholded,
                        "thresholds": [round(x, 4) for x in best_thresholds.tolist()]
                    },
                    f,
                    indent=2
                )

            print("Improvement detected. Model saved.")

        else:
            early_stop_counter += 1
            print(f"No improvement ({early_stop_counter}/{patience})")

            if early_stop_counter >= patience:
                print("\nEarly stopping triggered!")
                break

    # =========================
    # Final fold evaluation with best saved model
    # =========================
    model_best = ECGEfficientNet(num_classes=num_classes).to(device)
    model_best.load_state_dict(torch.load(fold_model_path, map_location=device))
    model_best.eval()

    best_thresholds = torch.load(fold_thresholds_pt_path, map_location="cpu")

    fold_val_loss, fold_val_y_preds, fold_val_y_trues, fold_val_logits = evaluate(
        valid_dl, model_best, loss_fn, device, return_logits=True
    )
    fold_val_probs = torch.softmax(fold_val_logits, dim=1)

    fold_val_acc_argmax = multiclass_accuracy(
        fold_val_y_preds, fold_val_y_trues, num_classes=num_classes
    ).item()
    fold_val_f1_argmax = multiclass_f1_score(
        fold_val_y_preds, fold_val_y_trues, num_classes=num_classes, average="macro"
    ).item()
    fold_val_auc = safe_multiclass_auc(
        fold_val_y_trues, fold_val_probs, num_classes=num_classes
    )

    fold_val_y_preds_thresholded = apply_class_thresholds(fold_val_probs, best_thresholds)
    fold_val_acc_thresholded = multiclass_accuracy(
        fold_val_y_preds_thresholded, fold_val_y_trues, num_classes=num_classes
    ).item()
    fold_val_f1_thresholded = multiclass_f1_score(
        fold_val_y_preds_thresholded, fold_val_y_trues, num_classes=num_classes, average="macro"
    ).item()

    print(f"\n===== FOLD {fold_idx} FINAL RESULT =====")
    print(f"ACC (argmax)      : {fold_val_acc_argmax:.4f}")
    print(f"F1  (argmax)      : {fold_val_f1_argmax:.4f}")
    print(f"AUC               : {fold_val_auc:.4f}" if not np.isnan(fold_val_auc) else "AUC               : nan")
    print(f"ACC (thresholded) : {fold_val_acc_thresholded:.4f}")
    print(f"F1  (thresholded) : {fold_val_f1_thresholded:.4f}")
    print(f"Thresholds        : {best_thresholds.tolist()}")

    fold_results.append({
        "fold": fold_idx,
        "acc_argmax": fold_val_acc_argmax,
        "f1_argmax": fold_val_f1_argmax,
        "auc": fold_val_auc,
        "acc_thresholded": fold_val_acc_thresholded,
        "f1_thresholded": fold_val_f1_thresholded,
        "thresholds": best_thresholds.tolist()
    })

    if fold_val_f1_thresholded > best_fold_f1:
        best_fold_f1 = fold_val_f1_thresholded
        best_fold_idx = fold_idx
        best_fold_thresholds = best_thresholds.clone()
        best_model_path = fold_model_path

writer.close()


# =========================
# Cross-validation summary
# =========================
acc_argmax_values = [x["acc_argmax"] for x in fold_results]
f1_argmax_values = [x["f1_argmax"] for x in fold_results]
auc_values = [x["auc"] for x in fold_results if not np.isnan(x["auc"])]
acc_thresholded_values = [x["acc_thresholded"] for x in fold_results]
f1_thresholded_values = [x["f1_thresholded"] for x in fold_results]

acc_argmax_mean, acc_argmax_std = summarize_metric(acc_argmax_values)
f1_argmax_mean, f1_argmax_std = summarize_metric(f1_argmax_values)
acc_thresholded_mean, acc_thresholded_std = summarize_metric(acc_thresholded_values)
f1_thresholded_mean, f1_thresholded_std = summarize_metric(f1_thresholded_values)

print("\n" + "=" * 40)
print("FINAL CROSS-VALIDATION RESULT")
print("=" * 40)
print(f"ACC (argmax)      : {acc_argmax_mean:.4f} ± {acc_argmax_std:.4f}")
print(f"F1  (argmax)      : {f1_argmax_mean:.4f} ± {f1_argmax_std:.4f}")

if len(auc_values) > 0:
    auc_mean, auc_std = summarize_metric(auc_values)
    print(f"AUC               : {auc_mean:.4f} ± {auc_std:.4f}")
else:
    print("AUC               : nan")

print(f"ACC (thresholded) : {acc_thresholded_mean:.4f} ± {acc_thresholded_std:.4f}")
print(f"F1  (thresholded) : {f1_thresholded_mean:.4f} ± {f1_thresholded_std:.4f}")
print(f"Best fold         : {best_fold_idx}")
print(f"Best fold F1      : {best_fold_f1:.4f}")
print(f"Best thresholds   : {best_fold_thresholds.tolist() if best_fold_thresholds is not None else None}")

with open("kfold_results.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "num_folds": num_folds,
            "acc_argmax_mean": round(acc_argmax_mean, 4),
            "acc_argmax_std": round(acc_argmax_std, 4),
            "f1_argmax_mean": round(f1_argmax_mean, 4),
            "f1_argmax_std": round(f1_argmax_std, 4),
            "auc_mean": round(auc_mean, 4) if len(auc_values) > 0 else None,
            "auc_std": round(auc_std, 4) if len(auc_values) > 0 else None,
            "acc_thresholded_mean": round(acc_thresholded_mean, 4),
            "acc_thresholded_std": round(acc_thresholded_std, 4),
            "f1_thresholded_mean": round(f1_thresholded_mean, 4),
            "f1_thresholded_std": round(f1_thresholded_std, 4),
            "best_fold_idx": best_fold_idx,
            "best_fold_f1": round(best_fold_f1, 4),
            "best_fold_thresholds": [round(x, 4) for x in best_fold_thresholds.tolist()] if best_fold_thresholds is not None else None,
            "fold_results": fold_results
        },
        f,
        indent=2
    )


# =========================
# Optional inference on external unlabeled test images
# Uses the best fold model
# =========================
test_image_dir = "data/test_image"

if best_model_path is not None and os.path.exists(test_image_dir):
    print("\nRunning inference on external test images using best fold model...")

    model_best = ECGEfficientNet(num_classes=num_classes).to(device)
    model_best.load_state_dict(torch.load(best_model_path, map_location=device))
    model_best.eval()

    inference_thresholds = best_fold_thresholds.cpu()

    image_files = [
        f for f in os.listdir(test_image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    predictions = []

    with torch.no_grad():
        for fname in image_files:
            img_path = os.path.join(test_image_dir, fname)
            image = Image.open(img_path).convert("RGB")

            img_tensor = eval_transform(image).unsqueeze(0).to(device)
            logits = model_best(img_tensor).cpu()
            probs = torch.softmax(logits, dim=1)

            predicted = apply_class_thresholds(probs, inference_thresholds).item()
            confidence = probs[0, predicted].item()

            hn = fname.split("_")[0]
            predictions.append({
                "HN": hn,
                "CaClass": predicted,
                "confidence": round(confidence, 6)
            })

    submission_df = pd.DataFrame(predictions).sort_values("HN")
    submission_df.to_csv("submission.csv", index=False)

    print(f"Created 'submission.csv' with {len(submission_df)} predictions.")
else:
    print("\nSkip external test inference: best model not found or test_image folder not found.")