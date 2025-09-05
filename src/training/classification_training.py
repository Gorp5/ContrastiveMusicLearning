import os
from pathlib import Path

import numpy as np
import torch

from utils.visualization import visualize_ROC_PR_AUC
from datasets import tqdm
from torch import optim


def train_classifier(model, test_dataloader, train_dataloader, config, show_graph=False):
    # Training setup
    file_path = f".\\{config.save_path}\\Config.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    loss_file = f".\\{config.save_path}\\loss.txt"
    os.makedirs(os.path.dirname(loss_file), exist_ok=True)

    torch.save(config, file_path)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    #scheduler = CosineAnnealingLR(optimizer, T_max=config.steps_per_cycle, eta_min=config.min_learning_rate )
    criterion = config.criterion
    model.to("cuda", config.dtype)

    total_steps = (len(train_dataloader) * config.num_epochs) * config.step_coefficient
    warmup_threshold = config.warmup_threshold
    final_gamma = config.gamma

    # Training loop
    step = 1
    for epoch in range(config.num_epochs):
        train_loss_total = 0

        model.train()

        for batch in tqdm(train_dataloader):
            inputs, labels = batch

            inputs = inputs.squeeze(0).unsqueeze(1)
            labels = labels.squeeze(0)

            num_chunks = int(inputs.shape[0] / config.max_batch_size) + 1

            data_minibatches = torch.chunk(inputs, num_chunks, dim=0)
            label_minibatches = torch.chunk(labels, num_chunks, dim=0)

            loss_per_batch = 0
            minibatch_len = len(data_minibatches)
            losses_this_minibatch = []
            for i, (data_minibatch, label_minibatch) in enumerate(zip(data_minibatches, label_minibatches)):
                data_minibatch = data_minibatch.to("cuda", config.dtype)
                label_minibatch = label_minibatch.to("cuda", config.dtype)

                outputs = model(data_minibatch)
                loss = criterion(outputs, label_minibatch)

                # loss = loss.view(-1)
                # k = int(0.7 * len(loss))  # keep top 70%
                # topk = torch.topk(loss, k).values
                # loss = topk.mean()
                # loss.backward()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                l = loss.item()
                loss_per_batch += l
                losses_this_minibatch.append(l)

            step += 1
            loss_average = loss_per_batch / minibatch_len
            with open(loss_file, "a") as file:
                file.write(f"Batch {step} | ------------------------\n")
                for index, loss in enumerate(losses_this_minibatch):
                    file.write(f"Minibatch {index + 1}:\nLoss: {loss}\n")
                file.write(f"Batch {step} | Average: {loss_average} |------------------------\n")

            train_loss_total += loss_per_batch / minibatch_len

        model.eval()
        test_loss_average, all_probs, all_labels = evaluate_classification(model, test_dataloader, config)

        probs = np.vstack(all_probs)  # from evaluate_classification
        print("probs mean,std,min,max:", probs.mean(), probs.std(), probs.min(), probs.max())
        print("frac > 0.5:", (probs > 0.5).mean())

        roc = None
        pr = None

        if show_graph:
            all_p_tensor = torch.stack([torch.tensor(x) for x in all_probs], dim=0).float()
            all_l_tensor = torch.stack([torch.tensor(x) for x in all_labels], dim=0).int()

            roc, pr = visualize_ROC_PR_AUC(all_p_tensor, all_l_tensor)

        train_loss_average = train_loss_total / len(train_dataloader)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss_average:.4f}")
        print(f"Epoch {epoch + 1}, Test Loss: {test_loss_average:.4f}")

        with open(loss_file, "a") as file:
            file.write(f"Epoch {epoch + 1}:\nTraining Loss: {train_loss_average}\n Test Loss: {test_loss_average}\n")
            if roc or pr:
                file.write(f"ROC-AUC: {roc}\n PR-AUC: {pr}\n")
            file.write(f"\n")

        torch.save(model, f".\\{config.save_path}\\Classifier-Epoch-{epoch + 1}.pt")

def evaluate_classification(model, dataloader, config):
    test_loss_total = 0

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

            loss_per_batch = 0
            minibatch_len = len(data_minibatches)
            for i, (data_minibatch, label_minibatch) in enumerate(zip(data_minibatches, label_minibatches)):
                outputs = model(data_minibatch)
                loss = config.criterion(outputs, label_minibatch)

                loss_per_batch += loss.item()

                all_preds.extend(outputs.sigmoid().cpu().numpy())
                all_labels.extend(label_minibatch.cpu().numpy())

            test_loss_total += loss_per_batch / minibatch_len

    return test_loss_total / len(dataloader), all_preds, all_labels