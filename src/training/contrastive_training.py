import os
import time

from loss import loss_utils
from loss.loss_utils import *
from datasets import tqdm
from torch import optim

def get_beta(percentage, cycles=8, coef=1, warmup=2):
    if percentage < warmup:
        return 1e-8

    percentage *= 2 * cycles
    percentage = (percentage + warmup) % 2

    if percentage > 1:
        percentage = 1

    return percentage * coef

def train_contrastive(model, test_dataloader, train_dataloader, config, variational=False, train_masked=False, test_masked=False):
    # Training setup
    file_path = f".\\{config.save_path}\\Config.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(config, file_path)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = config.criterion
    model.to("cuda", config.dtype)

    torch.autograd.set_detect_anomaly(True)

    f = open(f".\\{config.save_path}\\Loss.txt", "w")
    f.close()

    # Training loop
    step = 1
    for epoch in range(config.num_epochs):
        batch_steps = 0
        epoch_contrastive_loss = 0
        epoch_kld_loss = 0
        epoch_distribution_loss = 0
        batches = len(train_dataloader)
        for batch in tqdm(train_dataloader):
            indicies, inputs = batch
            # inputs = (inputs - inputs.mean(dim=[1, 2, 3], keepdim=True)) / (inputs.std(dim=[1, 2, 3], keepdim=True) + 1e-6)

            a, b = inputs

            a = a.to("cuda", config.dtype)
            b = b.to("cuda", config.dtype)

            za = model(a, masked=train_masked)
            zb = model(b, masked=train_masked)

            if variational:
                za, mean_a, logvar_a = za
                zb, mean_b, logvar_b = zb

            contrastive_loss = criterion(za, zb)

            if variational:
                kld_loss = distribution_normalizing_loss(mean_a, logvar_a)
                kld_loss += distribution_normalizing_loss(mean_b, logvar_b)
                kld_loss = kld_loss * 0.5 * get_beta(step/batches, cycles=config.cycles, coef=config.coef, warmup=config.warmup)

                distribution_loss = distribution_similarity_loss(mean_a, logvar_a, mean_b, logvar_b).mean()

                loss = contrastive_loss + kld_loss + distribution_loss
                epoch_kld_loss += kld_loss.item()
                epoch_distribution_loss += distribution_loss.item()
            else:
                loss = contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_contrastive_loss += contrastive_loss.item()

            step += 1
            batch_steps += 1

            term = f"Contrastive Loss [{batch_steps}/{batches}]: {contrastive_loss.item():.4f}"
            if variational:
                term += f"\t|\tKLD Loss [{batch_steps}/{batches}]: {kld_loss.item():.4f}"
                term += f"\t|\tDistribution Loss [{batch_steps}/{batches}]: {distribution_loss.item():.4f}"

            with open(f".\\{config.save_path}\\Loss.txt", "a") as f:
                term += "\n"
                f.write(term)

        contrastive_loss = evaluate_contrastive(model, test_dataloader, config, variational=variational, test_masked=test_masked)
        term = f"[Epoch {epoch}] Train: C = {epoch_contrastive_loss / batch_steps:.4f}"
        if variational:
            term += f"\t|\tKLD = {(epoch_kld_loss / config.coef) / batch_steps:.4f}"
            term += f"\t|\tDist Sim = {epoch_distribution_loss / batch_steps:.4f}\n"

            contrastive_loss, kld_loss = contrastive_loss
            term +=  f"Test:  C = {contrastive_loss:.4f}\t|\tKLD = {kld_loss / batch_steps:.4f}"
            #term +=  f"\t|\tDist Sim = {kld_loss / batch_steps:.4f}"
        else:
            term += f"Test:  C = {contrastive_loss:.4f}"
        print(term)

        torch.save(model, f".\\{config.save_path}\\Classifier-Epoch-{epoch}.pt")


def evaluate_contrastive(model, dataloader, config, variational=False, test_masked=False):
    contrastive_loss_total = 0
    kld_loss_total = 0

    criterion = config.criterion

    with torch.no_grad():
        for batch in tqdm(dataloader):
            indicies, inputs = batch

            a, b = inputs

            a = a.to("cuda", config.dtype)
            b = b.to("cuda", config.dtype)

            za = model(a, masked=test_masked)
            zb = model(b, masked=test_masked)

            if variational:
                za, mean_a, logvar_a = za
                zb, mean_b, logvar_b = zb

            contrastive_loss = criterion(za, zb)

            if variational:
                kld_loss = loss_utils.KLD(mean_a, logvar_a)
                kld_loss += loss_utils.KLD(mean_b, logvar_b)
                kld_loss = kld_loss * 0.5
                kld_loss_total += kld_loss.item()

            contrastive_loss_total += contrastive_loss.item()

    if variational:
        return contrastive_loss_total / len(dataloader), kld_loss_total / len(dataloader)
    return contrastive_loss_total / len(dataloader)