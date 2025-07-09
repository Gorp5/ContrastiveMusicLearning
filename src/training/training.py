import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup

from loss import loss_utils
from loss.loss_utils import compute_vae_loss, combined_loss
from training.evaluation import evaluate
import os
import torch.nn.functional as F

def train(
    model,
    train_loader,
    validation_loader,
    optimizer,
    config,
    beta_func,
    device="cuda"
):
    model.to(device)
    model.train()

    save_dir = os.path.join(config.save_path, model.name)
    os.makedirs(save_dir, exist_ok=True)

    total_steps = config.num_epochs * len(train_loader)
    warmup_steps = int(config.warmup_percent * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    current_step = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}"):
            inputs, masks = (batch if config.use_masks else (batch, None))
            inputs = inputs.to(device)
            if masks is not None:
                masks = masks.to(device)

            optimizer.zero_grad()

            if config.variational:
                training_frac = current_step / total_steps
                beta = beta_func(training_frac)

                if config.autoregressive:
                    outputs = model(inputs, masks)
                    inputs = inputs[:, 1:]
                else:
                    outputs = model(inputs, masks)

                recon, mean, logvar = outputs
                losses = compute_vae_loss(recon, inputs, mean, logvar, training_frac, beta)
                loss = losses["total_loss"]
            else:
                outputs = model(inputs, masks)
                if config.autoregressive:
                    inputs = inputs[:, 1:]
                loss = combined_loss(outputs, inputs)

            loss.backward()
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            current_step += 1

        avg_train_loss = epoch_loss / len(train_loader)

        validation_metrics = evaluate(model, validation_loader, beta=1.0, config=config, device=device)

        _log_epoch(epoch, avg_train_loss, validation_metrics, save_dir, model, config)


def trainTriplet(
    model,
    train_loader,
    validation_loader,
    optimizer,
    config,
    device="cuda"
):
    #assert train_loader is AudioDatasetTriplets

    model.to(device, dtype=config.dtype)
    model.train()

    save_dir = os.path.join(config.save_path, model.name)
    os.makedirs(save_dir, exist_ok=True)

    total_steps = config.num_epochs * len(train_loader)
    warmup_steps = int(config.warmup_percent * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    current_step = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}"):
            anchor, anchor_masks, positive, positive_masks, negative, negative_masks = batch

            anchor = anchor.to(device, dtype=config.dtype)
            anchor_masks = anchor_masks.to(device, dtype=config.dtype)
            positive = positive.to(device, dtype=config.dtype)
            positive_masks = positive_masks.to(device, dtype=config.dtype)
            negative = negative.to(device, dtype=config.dtype)
            negative_masks = negative_masks.to(device, dtype=config.dtype)

            B = anchor.shape[0]  # Original batch size

            optimizer.zero_grad()

            all_segments = torch.cat([anchor, positive, negative], dim=0)
            all_masks = torch.cat([anchor_masks, positive_masks, negative_masks], dim=0)

            reconstructed, latents = model(all_segments, all_masks)

            reconstruction_loss = loss_utils.combined_loss(reconstructed, all_segments)

            anchor_latents, positive_latents, negative_latents = torch.split(latents, B, dim=0)

            anchor_positive_loss = loss_utils.cosine_similarity(anchor_latents, positive_latents)
            anchor_negative_loss = loss_utils.cosine_similarity(anchor_latents, negative_latents)

            triplet_loss = F.relu(anchor_positive_loss - anchor_negative_loss + config.margin)
            loss = reconstruction_loss + triplet_loss

            loss.backward()
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            current_step += 1

        avg_train_loss = epoch_loss / len(train_loader)

        validation_metrics = evaluate(model, validation_loader, beta=1.0, config=config, device=device)

        _log_epoch(epoch, avg_train_loss, validation_metrics, save_dir, model, config)


def _log_epoch(epoch, train_loss, val_metrics, save_dir, model, config):
    cosine, mse, triplet = val_metrics

    print(f"[Epoch {epoch}] Train: {train_loss:.4f} | Triplet: {triplet:.4f} "
          #f"KLD: {kld:.4f}, μ: {kld_mean:.4f}, σ²: {kld_var:.4f}, "
          f"Cos: {cosine:.4f}, MSE: {mse:.4f}")

    with open(os.path.join(save_dir, "loss.txt"), "a") as f:
        f.write(f"Epoch-{epoch}\tTrain\t{train_loss:.4f}\n")
        f.write(f"Epoch-{epoch}\tVal\t{triplet:.4f}\n")
        #f.write(f"KLD: {kld:.4f}\tKLD μ: {kld_mean:.4f}\tKLD σ²: {kld_var:.4f}\t")
        f.write(f"Cos: {cosine:.4f}\tMSE: {mse:.4f}\n")

    if config.save_checkpoints:
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model.name}-epoch{epoch}.pt"))