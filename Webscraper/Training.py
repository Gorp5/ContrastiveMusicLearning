import torch
import os

import torch.nn
from tqdm import tqdm
import torch.profiler

from Analyzer.Webscraper.Loss import cross_corr_fft
from Loss import combined_loss, contrastive_loss
from Loss import reconstruction_loss
from transformers import get_cosine_schedule_with_warmup


def evaluate(model, dataloader, device="cuda"):
    model.eval()  # Set model to evaluation mode
    model.to(device)
    total_c_loss = 0.0
    total_r_loss = 0.0
    total_cc_loss = 0.0


    num_batches = 0

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)  # Move batch to GPU

            # Forward pass
            reconstructed = model(batch)

            # Compute loss (only on masked positions)
            r_loss = reconstruction_loss(reconstructed, batch)
            cc_loss = cross_corr_fft(reconstructed, batch)
            c_loss = r_loss + cc_loss

            total_r_loss += r_loss.mean()
            total_cc_loss += cc_loss.mean()
            total_c_loss += c_loss.mean()

            num_batches += 1

    total_cc_loss = total_cc_loss / num_batches
    total_c_loss = total_c_loss / num_batches
    total_r_loss = total_r_loss / num_batches

    print(f"Avg Reconstructive Loss: {total_r_loss:.10f}")
    print(f"Avg Cross FFT Loss: {total_cc_loss:.10f}")
    print(f"Avg Contrastive Loss: {total_c_loss:.10f}")

    return total_r_loss + total_cc_loss + total_c_loss

# Path
path = "E:\\Coding\\SongAnalyzer\\Analyzer\\Webscraper\\"

# ==== Training Function ====
def train(model, dataloader, test_dataloader, optimizer, num_epochs=20, device="cuda", loss_func=combined_loss):
    samples_per_batch = 8
    name = model.name
    directory = f"{path}{name}\\"

    if not os.path.exists(directory):
        os.makedirs(directory)

    loss_file = open(f"{directory}loss.txt", "a")

    warmup_steps = int(0.15 * num_epochs * len(dataloader))  # 10% warmup
    total_steps = num_epochs * len(dataloader)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


    for epoch in range(num_epochs):
        total_loss = 0

        model.train()
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            reconstructed = model(batch)

            # Compute loss (only on masked positions)
            loss = loss_func(reconstructed, batch)

            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()

            ls = loss.item()
            total_loss += ls

        model.eval()

        total_loss_valid = 0
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                batch = batch.to(device)

                # Forward pass
                reconstructed = model(batch)

                # Compute loss
                loss = loss_func(reconstructed, batch)
                total_loss_valid += loss.item()

        avg_loss_valid = total_loss_valid / len(test_dataloader)
        avg_loss = total_loss / len(dataloader)

        loss_file.write(f"Epoch-{epoch}\tTrain\t{avg_loss:.4f}\n")
        loss_file.write(f"Epoch-{epoch}\tValidation\t{avg_loss_valid:.4f}\n")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f} \t Validation Loss: {avg_loss_valid:.4f}")

        torch.save(model, f"{directory}-Epoch-{epoch + 1}.pt")

    loss_file.close()