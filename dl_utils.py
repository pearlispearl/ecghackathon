import torch


def train_one_epoch(
        dataloader, model, loss_fn,
        optimizer, epoch,
        device, writer=None,
        fold_idx=None,
        log_step_interval=50):
    size = len(dataloader.dataset)
    model.train()

    running_loss = 0.0

    for i, batch in enumerate(dataloader):
        X, y = batch
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % log_step_interval == 0:
            last_loss = running_loss / log_step_interval
            current = (i + 1) * len(X)
            print(f"loss: {last_loss:>7f}  [{current:>5d}/{size:>5d}]")

            if writer is not None:
                tag = "Loss/train_running" if fold_idx is None else f"Fold_{fold_idx}/Loss/train_running"
                writer.add_scalar(tag, last_loss, epoch * len(dataloader) + i)

            running_loss = 0.0


def evaluate(dataloader, model, loss_fn, device, return_logits=False):
    num_batches = len(dataloader)
    model.eval()

    loss = 0.0
    y_preds, y_trues = [], []
    logits_list = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            y_pred = pred.argmax(1)

            loss += loss_fn(pred, y).item()

            y_preds.append(y_pred.detach().cpu())
            y_trues.append(y.detach().cpu())

            if return_logits:
                logits_list.append(pred.detach().cpu())

    y_preds = torch.cat(y_preds)
    y_trues = torch.cat(y_trues)
    loss /= num_batches

    if return_logits:
        logits = torch.cat(logits_list)
        return loss, y_preds, y_trues, logits

    return loss, y_preds, y_trues