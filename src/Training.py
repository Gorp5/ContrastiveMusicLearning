import math

import torch
import os

import torch.nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.profiler

import Loss
from Loss import cross_corr_fft, variational_loss, KLD, mse_fft
from Loss import combined_loss
from Loss import reconstruction_loss
from transformers import get_cosine_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_


def evaluate(model, dataloader, device="cuda"):
    model.eval()  # Set model to evaluation mode
    model.to(device)

    total_k_loss = 0.0
    total_r_loss = 0.0
    total_cc_loss = 0.0
    total_mse_loss = 0.0
    total_fft_loss = 0.0

    num_batches = 0

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)  # Move batch to GPU
            masks = None#masks.to(device)

            # Forward pass
            reconstructed, mean, logvar = model(batch, masks)

            # Compute loss (only on masked positions)
            rec_loss = reconstruction_loss(reconstructed, batch)
            cross_relation = Loss.cross_relation_2(reconstructed, batch)
            mse_loss = Loss.mse(reconstructed, batch)
            cross_fft = Loss.cross_relation_fft_2(reconstructed, batch)
            k_loss = KLD(mean, logvar)

            total_r_loss += rec_loss.mean().item()
            total_cc_loss += cross_relation.mean().item()
            total_mse_loss += mse_loss.mean().item()
            total_fft_loss += cross_fft.mean().item()
            total_k_loss += k_loss.item()

            num_batches += 1

    total_cc_loss = total_cc_loss / num_batches
    total_r_loss = total_r_loss / num_batches
    total_k_loss = total_k_loss / num_batches
    total_mse_loss = total_mse_loss / num_batches
    total_fft_loss = total_fft_loss / num_batches


    print(f"Avg Reconstructive Loss: {total_r_loss:.10f}")
    print(f"Avg Cross Relation Loss: {total_cc_loss:.10f}")
    print(f"Avg KLD Loss: {total_k_loss:.10f}")
    print(f"Avg MSE Loss: {total_mse_loss:.10f}")
    print(f"Avg Cross FFT Loss: {total_fft_loss:.10f}")

    return total_r_loss + total_cc_loss + total_k_loss

# Path
path = "E:\\Coding\\SongAnalyzer\\Analyzer\\src\\"

def getBetaCos(step_fraction):
    return (1 - math.cos(step_fraction * math.pi))/2

def getBetaLog(step_fraction):
    return 1/(1 + math.e ** (-20 * (step_fraction - 0.5)))

def getBetaCyclical(step_fraction, cycle_length=2, coeff=0.1):
    if step_fraction - 0.2 < (6.0/30.0):
        return 5e-7

    adjusted = step_fraction * 26
    mod = adjusted % cycle_length
    term = 1 / (1 + math.e ** (-10 * (mod - 1.5)))
    return (term * coeff) * (step_fraction / 10)

# ==== Training Function ====
def  train(model, dataloader, test_dataloader, optimizer, cycle_length=2, coeff=0.1, num_epochs=20, device="cuda", loss_func=combined_loss, masks=True, dtype=torch.float32):
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

    current_steps = 0
    model.to(device, dtype=dtype)
    max_grad_norm = 1.0  # You can tune this
    #torch.autograd.set_detect_anomaly(True)

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for batch in tqdm(dataloader):
            batch = batch.to(device, dtype=dtype)
            masks = None

            beta = getBetaCyclical(current_steps / total_steps, cycle_length=cycle_length, coeff=coeff)

            optimizer.zero_grad()

            # Forward pass
            #reconstructed = model(batch, masks)
            reconstructed, mean, logvar = model(batch, masks)
            loss = variational_loss(reconstructed, batch, mean, logvar, beta=beta, device=device)

            # Backpropagation
            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # clip_grad_norm_(model.parameters(), max_grad_norm)
            # scaler.step(optimizer)
            # scaler.update()
            # scheduler.step()

            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            ls = loss.item()
            total_loss += ls
            current_steps += 1

        model.eval()

        total_k_loss = 0.0
        total_r_loss = 0.0
        total_cc_loss = 0.0
        total_mse_loss = 0.0
        total_fft_loss = 0.0

        # Disable gradient computation for evaluation
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                batch = batch.to(device)  # Move batch to GPU
                masks = None  # masks.to(device)

                # Forward pass
                reconstructed, mean, logvar = model(batch, masks)

                # Compute loss (only on masked positions)
                rec_loss = reconstruction_loss(reconstructed, batch)
                #cross_relation = Loss.cross_relation_2(reconstructed, batch)
                mse_loss = Loss.mse(reconstructed, batch)
                cross_fft = Loss.cross_corr_fft(reconstructed, batch)
                k_loss = KLD(mean, logvar)

                total_r_loss += rec_loss.mean().item()
                #total_cc_loss += cross_relation.mean().item()
                total_mse_loss += mse_loss.mean().item()
                total_fft_loss += cross_fft.mean().item()
                total_k_loss += k_loss.item()

        # if epoch % 5 == 0:
        #     diagnose_posterior_collapse(model, test_dataloader, device)


        total_cc_loss = total_cc_loss / len(test_dataloader)
        total_r_loss = total_r_loss / len(test_dataloader)
        total_k_loss = total_k_loss / len(test_dataloader)
        total_mse_loss = total_mse_loss / len(test_dataloader)
        total_fft_loss = total_fft_loss / len(test_dataloader)

        avg_loss = total_loss / len(dataloader)

        loss_file.write(f"Epoch-{epoch}\tTrain\t{avg_loss:.4f}\n")
        loss_file.write(f"Epoch-{epoch}\tValidation\t{total_k_loss + total_r_loss + total_cc_loss + total_mse_loss + total_fft_loss:.4f}\n")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f} \t Validation Losses:\nKLD: {total_k_loss:.4f}\tCosine Similarity: {total_r_loss:.4f}\tCross Coorelation: {total_cc_loss:.4f}\tMSE: {total_mse_loss:.4f}\tCross FFT: {total_fft_loss:.4f}")


        torch.save(model, f"{directory}-Epoch-{epoch + 1}.pt")

    loss_file.close()

@torch.no_grad()
def diagnose_posterior_collapse(model, dataloader, device='cuda'):
    with torch.no_grad():
        model.eval()
        similarities = []
        latent_vars = []


        for batch in tqdm(dataloader):
            x = batch.to(device)

            # Encode
            z_mu, z_logvar = model.to_latent(x, None)
            z1 = model.reparameterization(z_mu, z_logvar)
            z2 = model.reparameterization(z_mu, z_logvar)  # second sample for same input

            # Decode
            x1 = model.from_latent(z1, None)
            x2 = model.from_latent(z2, None)

            # Flatten and compute cosine similarity
            sim = F.cosine_similarity(x1.view(x1.size(0), -1), x2.view(x2.size(0), -1), dim=1)
            similarities.append(sim.cpu())

            # Track variance of latent dimensions
            latent_vars.append(z1.cpu())

    similarities = torch.cat(similarities)
    latent_vars = torch.cat(latent_vars)

    print(f"\n🔍 Posterior Collapse Diagnostic:")
    print(f"  Avg cosine similarity (sampled reconstructions): {similarities.mean().item():.4f}")
    print(f"  Std of latent dimensions (should NOT all be ~1):")

    stds = latent_vars.std(dim=0)
    for i, std in enumerate(stds):
        print(f"    z[{i}]: std={std:.4f}")

    # Optional: visualize
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(similarities.numpy(), bins=30)
    plt.title("Cosine similarity between reconstructions")
    plt.subplot(1, 2, 2)
    plt.bar(range(len(stds)), stds)
    plt.title("Std of latent dimensions")
    plt.tight_layout()
    plt.show()