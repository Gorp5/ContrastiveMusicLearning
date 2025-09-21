import os
import time

from loss.loss_utils import *
from datasets import tqdm
from torch import optim


def train_contrastive(model, test_dataloader, train_dataloader, config, show_graph=False):
    # Training setup
    file_path = f".\\{config.save_path}\\Config.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(config, file_path)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = config.criterion
    model.to("cuda", config.dtype)

    torch.autograd.set_detect_anomaly(True)

    # Training loop
    step = 1
    for epoch in range(config.num_epochs):
        mid_epoch_total = 0
        batch_steps = 0
        epoch_loss = 0
        steps_in_batch = len(train_dataloader)
        for batch in tqdm(train_dataloader):
            indicies, inputs = batch
            # inputs = (inputs - inputs.mean(dim=[1, 2, 3], keepdim=True)) / (inputs.std(dim=[1, 2, 3], keepdim=True) + 1e-6)

            a, b = inputs

            a = a.to("cuda", config.dtype)
            b = b.to("cuda", config.dtype)

            za = model(a)
            zb = model(b)

            loss = criterion(za, zb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mid_epoch_loss = loss.item()
            epoch_loss += mid_epoch_loss

            step += 1
            batch_steps += 1

            print(f"Batch Loss [{batch_steps}/{steps_in_batch}]: {mid_epoch_loss:.4f}\n")

        contrastive_loss = evaluate_contrastive(model, test_dataloader, config)

        print(f"[Epoch {epoch}] Train:  {epoch_loss / batch_steps:.4f}\n"
              f"Test:  Contrastive Loss: {contrastive_loss:.4f}"
              )

        torch.save(model, f".\\{config.save_path}\\Classifier-Epoch-{epoch}.pt")


def evaluate_contrastive(model, dataloader, config):
    contrastive_loss = 0
    criterion = config.criterion

    with torch.no_grad():
        for batch in tqdm(dataloader):
            indicies, inputs = batch

            a, b = inputs

            a = a.to("cuda", config.dtype)
            b = b.to("cuda", config.dtype)

            za = model(a)
            zb = model(b)

            loss = criterion(za, zb)

            contrastive_loss += loss.item()

    return contrastive_loss / len(dataloader)