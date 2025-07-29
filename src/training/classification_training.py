import numpy as np
import torch
from matplotlib import pyplot as plt

from utils.visualization import visualize_ROC_PR_AUC
from datasets import tqdm
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_classifier(model, test_dataloader, train_dataloader, config, show_graph=False):
    # Training setup
    torch.save(config, f".\\{config.save_path}\\Config.pt")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    #scheduler = CosineAnnealingLR(optimizer, T_max=config.steps_per_cycle, eta_min=config.min_learning_rate )
    criterion = config.criterion
    model.to("cuda", config.dtype)

    total_steps = (len(train_dataloader) * config.num_epochs) * config.step_coefficient
    warmup_threshold = config.warmup_threshold
    final_gamma = config.gamma

    torch.autograd.set_detect_anomaly(True)

    # Training loop
    step = 1
    for epoch in range(config.num_epochs):
        train_loss_total = 0
        for batch in tqdm(train_dataloader):
            inputs, labels = batch

            inputs = inputs.squeeze(0).unsqueeze(1)
            labels = labels.squeeze(0)

            num_chunks = int(inputs.shape[0] / config.max_batch_size) + 1

            data_minibatches = torch.chunk(inputs, num_chunks, dim=0)
            label_minibatches = torch.chunk(labels, num_chunks, dim=0)

            loss_per_batch = 0
            minibatch_len = len(data_minibatches)
            for i, (data_minibatch, label_minibatch) in enumerate(zip(data_minibatches, label_minibatches)):
                data_minibatch = data_minibatch.to("cuda", config.dtype)
                label_minibatch = label_minibatch.to("cuda", config.dtype)

                outputs = model(data_minibatch)
                loss = criterion(outputs, label_minibatch)

                optimizer.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    loss.backward()
                optimizer.step()

                loss_per_batch += loss.item()

            if step / total_steps > warmup_threshold:
                config.criterion.set_gamma(np.min((step / total_steps, 1)) * final_gamma)

            step += 1

            train_loss_total += loss_per_batch / minibatch_len

        test_loss_average, all_probs, all_labels = evaluate_classification(model, test_dataloader, config)

        if show_graph:
            all_p_tensor = torch.stack([torch.tensor(x) for x in all_probs], dim=0).float()
            all_l_tensor = torch.stack([torch.tensor(x) for x in all_labels], dim=0).int()

            visualize_ROC_PR_AUC(all_p_tensor, all_l_tensor)

        train_loss_average = train_loss_total / len(train_dataloader)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss_average:.4f}")
        print(f"Epoch {epoch + 1}, Test Loss: {test_loss_average:.4f}")

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