import torch
import numpy as np


# ---------------------------------------------------------------------------
# Mixup helpers
# ---------------------------------------------------------------------------

def mixup_data(x, y, alpha=0.4, device='cpu'):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    Mixup paper: https://arxiv.org/abs/1710.09412
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Interpolated loss between two target labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
        dataloader, model, loss_fn,
        optimizer, epoch,
        device, writer,
        log_step_interval=50,
        mixup_alpha=0.4):
    """One training epoch with optional Mixup augmentation."""
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.

    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Apply Mixup
        X_mix, y_a, y_b, lam = mixup_data(X, y, alpha=mixup_alpha, device=device)

        optimizer.zero_grad()
        pred = model(X_mix)
        loss = mixup_criterion(loss_fn, pred, y_a, y_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % log_step_interval == 0:
            last_loss = running_loss / log_step_interval
            current = (i + 1) * len(X)
            print(f"    loss: {last_loss:.6f}  [{current:>5d}/{size:>5d}]")
            writer.add_scalar(
                'Loss/train_running',
                running_loss / log_step_interval,
                epoch * len(dataloader) + i
            )
            running_loss = 0.


def test(dataloader, model, loss_fn, device):
    """Evaluate model on a dataloader; returns avg loss, predictions, true labels."""
    num_batches = len(dataloader)
    model.eval()

    loss = 0.
    y_preds, y_trues = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            y_preds.append(pred.argmax(1))
            y_trues.append(y)

    y_preds = torch.cat(y_preds)
    y_trues = torch.cat(y_trues)
    loss /= num_batches

    return loss, y_preds, y_trues
