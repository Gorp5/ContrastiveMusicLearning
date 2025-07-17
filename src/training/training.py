import torch
from torch import nn
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup

from loss import loss_utils
from training.evaluation import *
import os
import torch.nn.functional as F

def trainHybrid(
    model,
    train_loader,
    validation_loader,
    optimizer,
    config,
    device="cuda",
):
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
    label_loss = nn.BCEWithLogitsLoss(pos_weight=config.class_ratio)

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}"):
            data, masks, tags = batch

            genre_tags, mood_tags = tags

            genre_tags, is_dummy_genre_pos = genre_tags
            mood_tags, is_dummy_mood_pos = mood_tags

            is_dummy_genre_pos = is_dummy_genre_pos.to(device)
            is_dummy_mood_pos = is_dummy_mood_pos.to(device)

            genre_tags = genre_tags.to(device, dtype=config.dtype)
            mood_tags = mood_tags.to(device, dtype=config.dtype)

            genre_tags = genre_tags[is_dummy_genre_pos]
            mood_tags = mood_tags[is_dummy_mood_pos]

            data = data.to(device, dtype=config.dtype)
            masks = masks.to(device, dtype=config.dtype)

            optimizer.zero_grad()

            reconstructed, latents, genre_labels, mood_labels = model(data, masks)

            genre_labels = genre_labels[is_dummy_genre_pos]
            mood_labels = mood_labels[is_dummy_mood_pos]

            genre_loss = label_loss(genre_labels, genre_tags)
            mood_loss = label_loss(mood_labels, mood_tags)

            classification_loss = genre_loss + mood_loss

            #reconstruction_loss = loss_utils.combined_loss(reconstructed, data)
            reconstruction_loss = 0
            loss = reconstruction_loss + classification_loss

            loss.backward()
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            current_step += 1

        avg_train_loss = epoch_loss / len(train_loader)

        validation_metrics = evaluate(model, validation_loader, config=config, device=device)

        _log_epoch(epoch, avg_train_loss, validation_metrics, save_dir, model, config)


def trainTriplet(
    model,
    train_loader,
    validation_loader,
    optimizer,
    config,
    device="cuda",
    use_tags=False
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
    label_loss = nn.BCEWithLogitsLoss()

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}"):
            if use_tags:
                anchor, anchor_masks, positive, positive_masks, negative, negative_masks, anchor_tags, negative_tags = batch
                anchor_genre_tags, anchor_mood_tags = anchor_tags
                negative_genre_tags, negative_mood_tags = negative_tags

                anchor_genre_tags, is_dummy_genre_pos = anchor_genre_tags
                anchor_mood_tags, is_dummy_mood_pos = anchor_mood_tags

                negative_genre_tags, is_dummy_genre_neg = negative_genre_tags
                negative_mood_tags, is_dummy_mood_neg = negative_mood_tags

                is_dummy_genre_pos = is_dummy_genre_pos.to(device)
                is_dummy_mood_pos = is_dummy_mood_pos.to(device)
                is_dummy_genre_neg = is_dummy_genre_neg.to(device)
                is_dummy_mood_neg = is_dummy_mood_neg.to(device)

                anchor_genre_tags = anchor_genre_tags.to(device, dtype=config.dtype)
                anchor_mood_tags = anchor_mood_tags.to(device, dtype=config.dtype)
                negative_genre_tags = negative_genre_tags.to(device, dtype=config.dtype)
                negative_mood_tags = negative_mood_tags.to(device, dtype=config.dtype)

                anchor_genre_tags = anchor_genre_tags[is_dummy_genre_pos]
                anchor_mood_tags = anchor_mood_tags[is_dummy_mood_pos]
                negative_genre_tags = negative_genre_tags[is_dummy_genre_neg]
                negative_mood_tags = negative_mood_tags[is_dummy_mood_neg]
            else:
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

            if use_tags:
                reconstructed, latents, genre_labels, mood_labels = model(all_segments, all_masks)
                anchor_genre_labels, positive_genre_labels, negative_genre_labels = torch.split(genre_labels, B, dim=0)
                anchor_mood_labels, positive_mood_labels, negative_mood_labels = torch.split(mood_labels, B, dim=0)

                anchor_genre_labels = anchor_genre_labels[is_dummy_genre_pos]
                anchor_mood_labels = anchor_mood_labels[is_dummy_mood_pos]

                positive_genre_labels = positive_genre_labels[is_dummy_genre_pos]
                positive_mood_labels = positive_mood_labels[is_dummy_mood_pos]

                negative_genre_labels = negative_genre_labels[is_dummy_genre_neg]
                negative_mood_labels = negative_mood_labels[is_dummy_mood_neg]

                anchor_genre_loss = label_loss(anchor_genre_labels, anchor_genre_tags)
                positive_genre_loss = label_loss(positive_genre_labels,anchor_genre_tags)
                negative_genre_loss = label_loss(negative_genre_labels,negative_genre_tags)

                anchor_mood_loss = label_loss(anchor_mood_labels, anchor_mood_tags)
                positive_mood_loss = label_loss(positive_mood_labels, anchor_mood_tags)
                negative_mood_loss = label_loss(negative_mood_labels,negative_mood_tags)

                classification_loss = anchor_genre_loss + positive_genre_loss + negative_genre_loss + anchor_mood_loss + positive_mood_loss + negative_mood_loss
            else:
                reconstructed, latents = model(all_segments, all_masks)

            #reconstruction_loss = loss_utils.combined_loss(reconstructed, all_segments)
            reconstructed_loss = 0
            anchor_latents, positive_latents, negative_latents = torch.split(latents, B, dim=0)

            anchor_positive_loss = loss_utils.cosine_similarity(anchor_latents, positive_latents)
            anchor_negative_loss = loss_utils.cosine_similarity(anchor_latents, negative_latents)

            triplet_loss = F.relu(anchor_positive_loss - anchor_negative_loss + config.margin)
            loss = reconstruction_loss + triplet_loss

            if use_tags:
                loss = loss + classification_loss

            loss.backward()
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            current_step += 1

        avg_train_loss = epoch_loss / len(train_loader)

        validation_metrics = evaluate(model, validation_loader, beta=1.0, config=config, device=device, use_tags=use_tags)

        _log_epoch_triplet(epoch, avg_train_loss, validation_metrics, save_dir, model, config)


def _log_epoch(epoch, train_loss, val_metrics, save_dir, model, config):
    cosine, mse, genre_class, mood_class, _, _, _, _ = val_metrics

    print(f"[Epoch {epoch}] Train: {train_loss:.4f} "
          #f"KLD: {kld:.4f}, μ: {kld_mean:.4f}, σ²: {kld_var:.4f}, "
          f"Cos: {cosine:.4f}, MSE: {mse:.4f}"
          f"Genre: {genre_class:.4f}, Mood: {mood_class:.4f}")

    with open(os.path.join(save_dir, "loss.txt"), "a") as f:
        f.write(f"Epoch-{epoch}\tTrain\t{train_loss:.4f}\n")
        #f.write(f"KLD: {kld:.4f}\tKLD μ: {kld_mean:.4f}\tKLD σ²: {kld_var:.4f}\t")
        f.write(f"Cos: {cosine:.4f}\tMSE: {mse:.4f}\n")
        f.write(f"Genre: {genre_class:.4f}, Mood: {mood_class:.4f}")

    if config.save_checkpoints:
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model.name}-epoch{epoch}.pt"))


def _log_epoch_triplet(epoch, train_loss, val_metrics, save_dir, model, config):
    cosine, mse, triplet, genre_class, mood_class, _, _ = val_metrics

    print(f"[Epoch {epoch}] Train: {train_loss:.4f} | Triplet: {triplet:.4f} "
          #f"KLD: {kld:.4f}, μ: {kld_mean:.4f}, σ²: {kld_var:.4f}, "
          f"Cos: {cosine:.4f}, MSE: {mse:.4f}"
          f"Genre: {genre_class:.4f}, Mood: {mood_class:.4f}")

    with open(os.path.join(save_dir, "loss.txt"), "a") as f:
        f.write(f"Epoch-{epoch}\tTrain\t{train_loss:.4f}\n")
        f.write(f"Epoch-{epoch}\tVal\t{triplet:.4f}\n")
        #f.write(f"KLD: {kld:.4f}\tKLD μ: {kld_mean:.4f}\tKLD σ²: {kld_var:.4f}\t")
        f.write(f"Cos: {cosine:.4f}\tMSE: {mse:.4f}\n")
        f.write(f"Genre: {genre_class:.4f}, Mood: {mood_class:.4f}")

    if config.save_checkpoints:
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model.name}-epoch{epoch}.pt"))