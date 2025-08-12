import os

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

    # Training loop
    step = 1
    for epoch in range(config.num_epochs):
        train_loss_total = 0
        for batch in tqdm(train_dataloader):
            inputs, labels = batch

            inputs = inputs.squeeze(0)
            labels = labels.squeeze(0)

            num_chunks = int(inputs.shape[0] / config.max_batch_size) + 1

            data_minibatches = torch.chunk(inputs, num_chunks, dim=0)
            label_minibatches = torch.chunk(labels, num_chunks, dim=0)

            loss_per_batch = 0
            minibatch_len = len(data_minibatches)
            for i, (data_minibatch, label_minibatch) in enumerate(zip(data_minibatches, label_minibatches)):
                data_minibatch = data_minibatch.to("cuda", config.dtype).permute(0, 2, 1)
                label_minibatch = label_minibatch.to("cuda", config.dtype)

                outputs, latent = model(data_minibatch)
                loss = criterion(outputs, data_minibatch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_per_batch += loss.item()

            step += 1

            train_loss_total += loss_per_batch / minibatch_len

        mse_loss, cos_loss, all_probs, all_labels = evaluate_autoencode(model, test_dataloader, config)

        if show_graph:
            all_p_tensor = torch.stack([torch.tensor(x) for x in all_probs], dim=0).float()
            all_l_tensor = torch.stack([torch.tensor(x) for x in all_labels], dim=0).int()

            visualize_ROC_PR_AUC(all_p_tensor, all_l_tensor)

        train_loss_average = train_loss_total / len(train_dataloader)

        print(f"Epoch {epoch + 1}, Test Loss: {train_loss_average:.4f}")
        print(f"Epoch {epoch + 1}, Cos Loss: {mse_loss:.4f}")
        print(f"Epoch {epoch + 1}, MSE Loss: {cos_loss:.4f}")

        torch.save(model, f".\\{config.save_path}\\Classifier-Epoch-{epoch + 1}.pt")

def evaluate_autoencode(model, dataloader, config):
    mse_loss_per_batch = 0
    cos_loss_per_batch = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, labels = batch

            inputs = inputs.squeeze(0).unsqueeze(1).to("cuda", config.dtype)
            labels = labels.squeeze(0).to("cuda", config.dtype)

            num_chunks = int(inputs.shape[0] / config.max_batch_size) + 1

            data_minibatches = torch.chunk(inputs, num_chunks, dim=0)
            label_minibatches = torch.chunk(labels, num_chunks, dim=0)

            mse_loss_per_batch = 0
            cos_loss_per_batch = 0

            minibatch_len = len(data_minibatches)
            for i, (data_minibatch, label_minibatch) in enumerate(zip(data_minibatches, label_minibatches)):
                outputs = model(data_minibatch)
                cos_loss = cosine_similarity(outputs, label_minibatch)
                mse_loss = mse(outputs, label_minibatch)

                cos_loss_per_batch += cos_loss.item()
                mse_loss_per_batch += mse_loss.item()

                #all_preds.extend(outputs.sigmoid().cpu().numpy())
                #all_labels.extend(label_minibatch.cpu().numpy())

            mse_loss_per_batch += mse_loss_per_batch / minibatch_len
            cos_loss_per_batch += cos_loss_per_batch / minibatch_len


    return mse_loss_per_batch / len(dataloader), cos_loss_per_batch / len(dataloader), all_preds, all_labels