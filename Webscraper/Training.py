import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from tqdm import tqdm
import math
import torch.profiler


from Loss import combined_loss
from Loss import reconstruction_loss


def evaluate(model, dataloader, device="cuda"):
    model.eval()  # Set model to evaluation mode
    model.to(device)
    total_loss = 0.0
    total_c_loss = 0.0
    total_r_loss = 0.0

    num_batches = 0

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)  # Move batch to GPU

            # Forward pass
            reconstructed = model(batch)

            # Compute loss (only on masked positions)
            loss = combined_loss(reconstructed, batch)
            r_loss = reconstruction_loss(reconstructed, batch)
            c_loss = loss - r_loss

            total_c_loss += c_loss.mean()
            total_r_loss += r_loss.mean()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    total_c_loss = total_c_loss / num_batches
    total_r_loss = total_r_loss / num_batches

    print(f"Avg Reconstructive Loss: {total_r_loss:.10f}")
    print(f"Avg Contrastive Loss: {total_c_loss:.10f}")
    print(f"Avg Total Loss: {avg_loss:.4f}")

    return avg_loss

# Path
path = "E:\\Coding\\SongAnalyzer\\Analyzer\\Webscraper\\"

# ==== Training Function ====
def train(model, dataloader, test_dataloader, optimizer, num_epochs=20, device="cuda"):
    samples_per_batch = 8
    name = model.name
    directory = f"{path}{name}\\"

    if not os.path.exists(directory):
        os.makedirs(directory)

    loss_file = open(f"{directory}loss.txt", "a")

    #model.to(device)


    for epoch in range(num_epochs):
        total_loss = 0

        model.train()
        for batch in tqdm(dataloader):
            batch = batch.to(device)  # Move batch to GPU
            optimizer.zero_grad()

            # Forward pass
            # with torch.profiler.profile(record_shapes=True) as prof:
            reconstructed = model(batch)

            # Compute loss (only on masked positions)
            loss = combined_loss(reconstructed, batch)

            # Backpropagation
            loss.backward()
            optimizer.step()
            # print(prof.key_averages().table(sort_by="self_cuda_time_total"))

            ls = loss.item()
            total_loss += ls

        model.eval()

        total_loss_valid = 0
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                batch = batch.to(device)  # Move batch to GPU

                # Forward pass
                reconstructed = model(batch)

                # Compute loss (only on masked positions)
                loss = combined_loss(reconstructed, batch)
                total_loss_valid += loss.item()

        avg_loss_valid = total_loss_valid / len(test_dataloader)

        avg_loss = total_loss / len(dataloader)
        loss_file.write(f"Epoch-{epoch}\tTrain\t{avg_loss:.4f}\n")
        loss_file.write(f"Epoch-{epoch}\tValidation\t{avg_loss_valid:.4f}\n")
        torch.save(model, f"{directory}-Epoch-{epoch + 1}.pt")

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f} \t Validation Loss: {avg_loss:.4f}")

    loss_file.close()