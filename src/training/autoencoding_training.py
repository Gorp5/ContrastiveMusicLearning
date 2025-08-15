import os
import time

import numpy as np
import torch

from loss.loss_utils import *
from utils.visualization import visualize_ROC_PR_AUC
from datasets import tqdm
from torch import optim


def train_autoencode(model, test_dataloader, train_dataloader, config, show_graph=False):
    # Training setup
    file_path = f".\\{config.save_path}\\Config.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(config, file_path)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    #scheduler = CosineAnnealingLR(optimizer, T_max=config.steps_per_cycle, eta_min=config.min_learning_rate )
    criterion = config.criterion
    model.to("cuda", config.dtype)

    total_steps = (len(train_dataloader) * config.num_epochs) * config.step_coefficient
    warmup_threshold = config.warmup_threshold

    torch.autograd.set_detect_anomaly(True)

    time_start = time.time()
    first_time = True

    # Training loop
    step = 1
    for epoch in range(config.num_epochs):
        previous_mid_epoch_total = 0
        mid_epoch_total = 0
        batch_steps = 0
        for batch in tqdm(train_dataloader):

            inputs, labels = batch
            inputs = (inputs - inputs.mean(dim=[1, 2, 3], keepdim=True)) / (inputs.std(dim=[1, 2, 3], keepdim=True) + 1e-6)

            inputs = inputs.squeeze(0)
            labels = labels.squeeze(0)

            num_chunks = int(inputs.shape[0] / config.max_batch_size) + 1

            data_minibatches = torch.chunk(inputs, num_chunks, dim=0)
            label_minibatches = torch.chunk(labels, num_chunks, dim=0)

            loss_per_batch = 0
            minibatch_len = len(data_minibatches)
            for i, (data_minibatch, label_minibatch) in enumerate(zip(data_minibatches, label_minibatches)):

                data_minibatch = data_minibatch.to("cuda", config.dtype)

                outputs, latent = model(data_minibatch)
                loss = criterion(outputs, data_minibatch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_per_batch += loss.item()

            step += 1

            mid_epoch_total += loss_per_batch / minibatch_len
            batch_steps += 1

            if batch_steps % 9 == 0 or step == 2:
                previous_mid_epoch_average = mid_epoch_total / batch_steps
                print(f"Mid Epoch Loss: {previous_mid_epoch_average:0.4}\n")
                mid_epoch_total = 0
                batch_steps = 0


        mse_loss, cos_loss, all_probs, all_labels = evaluate_autoencode(model, test_dataloader, config)

        print(f"[Epoch {epoch}] Train:  {previous_mid_epoch_average:.4f}\n"
              # f"KLD: {kld:.4f}, μ: {kld_mean:.4f}, σ²: {kld_var:.4f}, "
              f"Test:  Cos: {cos_loss:.4f}, MSE: {mse_loss:.4f}"
              #f"Genre: {genre_class:.4f}, Mood: {mood_class:.4f}"
        )

        torch.save(model, f".\\{config.save_path}\\Classifier-Epoch-{epoch}.pt")

def evaluate_autoencode(model, dataloader, config):
    mse_loss = 0
    cos_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, labels = batch

            inputs = (inputs - inputs.mean(dim=[1, 2, 3], keepdim=True)) / (inputs.std(dim=[1, 2, 3], keepdim=True) + 1e-6)

            inputs = inputs.squeeze(0)
            labels = labels.squeeze(0)

            num_chunks = int(inputs.shape[0] / config.max_batch_size) + 1

            data_minibatches = torch.chunk(inputs, num_chunks, dim=0)
            label_minibatches = torch.chunk(labels, num_chunks, dim=0)

            mse_loss_per_batch = 0
            cos_loss_per_batch = 0

            minibatch_len = len(data_minibatches)
            for i, (data_minibatch, label_minibatch) in enumerate(zip(data_minibatches, label_minibatches)):
                data_minibatch = data_minibatch.to("cuda", config.dtype)

                outputs, latents = model(data_minibatch)

                cos_loss_per_batch += cosine_similarity(outputs, data_minibatch).item()
                mse_loss_per_batch += mse(outputs, data_minibatch).item()

                #all_preds.extend(outputs.sigmoid().cpu().numpy())
                #all_labels.extend(label_minibatch.cpu().numpy())

            mse_loss += mse_loss_per_batch / minibatch_len
            cos_loss += cos_loss_per_batch / minibatch_len


    return mse_loss / len(dataloader), cos_loss / len(dataloader), all_preds, all_labels