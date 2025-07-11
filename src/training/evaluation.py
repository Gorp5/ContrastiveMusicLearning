import torch
from torch import nn
from tqdm import tqdm

from loss import loss_utils
import torch.nn.functional as F

def evaluate(
        model,
        dataloader,
        config,
        device="cuda",
):
    model.eval()

    cosine_total = 0.0
    mse_total = 0.0
    genre_classification_loss = 0.0
    mood_classification_loss = 0.0

    num_batches = 0

    all_mood_preds = []
    all_genre_preds = []

    all_mood_targets = []
    all_genre_targets = []

    label_loss = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader):
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

            reconstructed, latents, predicted_genre_labels, predicted_mood_labels = model(data, masks)

            genre_classification_loss += label_loss(genre_tags, predicted_genre_labels)
            mood_classification_loss += label_loss(mood_tags, predicted_mood_labels)

            cosine_total += loss_utils.cosine_similarity(reconstructed, data)
            mse_total += loss_utils.mse(reconstructed, data)

            all_genre_preds.append(predicted_genre_labels.cpu())
            all_mood_preds.append(predicted_mood_labels.cpu())

            all_genre_targets.append(genre_tags.cpu())
            all_mood_targets.append(mood_tags.cpu())

            num_batches += 1

    # Average over batches
    def safe_div(x):
        return x / num_batches if num_batches > 0 else 0.0

    return (
        safe_div(cosine_total),
        safe_div(mse_total),
        safe_div(genre_classification_loss),
        safe_div(mood_classification_loss),

        all_genre_preds, all_genre_targets,
        all_mood_preds, all_mood_targets,
    )

def evaluateTriplet(
        model,
        dataloader,
        config,
        beta=1.0,
        device="cuda",
        dtype=torch.float32,
        use_tags=False,
):
    model.eval()

    cosine_total = 0.0
    mse_total = 0.0
    triplets_total = 0.0
    genre_classification_loss = 0.0
    mood_classification_loss = 0.0

    total_loss = 0.0
    num_batches = 0

    all_mood_preds = []
    all_genre_preds = []

    all_mood_targets = []
    all_genre_targets = []

    all_targets = []

    label_loss = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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

            all_segments = torch.cat([anchor, positive, negative], dim=0)
            all_masks = torch.cat([anchor_masks, positive_masks, negative_masks], dim=0)

            if use_tags:
                reconstructed, latents, genre_labels, mood_labels = model(all_segments, all_masks)
                anchor_genre_labels, positive_genre_labels, negative_genre_labels = torch.split(genre_labels, B,
                                                                                                dim=0)
                anchor_mood_labels, positive_mood_labels, negative_mood_labels = torch.split(mood_labels, B, dim=0)

                predicted_anchor_genre_labels = anchor_genre_labels[is_dummy_genre_pos]
                predicted_anchor_mood_labels = anchor_mood_labels[is_dummy_mood_pos]

                predicted_positive_genre_labels = positive_genre_labels[is_dummy_genre_pos]
                predicted_positive_mood_labels = positive_mood_labels[is_dummy_mood_pos]

                predicted_negative_genre_labels = negative_genre_labels[is_dummy_genre_neg]
                predicted_negative_mood_labels = negative_mood_labels[is_dummy_mood_neg]

                all_genre_preds.append(predicted_anchor_genre_labels.cpu())
                all_genre_preds.append(predicted_positive_genre_labels.cpu())
                all_genre_preds.append(predicted_negative_genre_labels.cpu())
                all_mood_preds.append(predicted_anchor_mood_labels.cpu())
                all_mood_preds.append(predicted_positive_mood_labels.cpu())
                all_mood_preds.append(predicted_negative_mood_labels.cpu())


                all_genre_targets.append(anchor_genre_tags.cpu())
                all_genre_targets.append(anchor_genre_tags.cpu())
                all_genre_targets.append(negative_genre_tags.cpu())
                all_mood_targets.append(anchor_mood_tags.cpu())
                all_mood_targets.append(anchor_mood_tags.cpu())
                all_mood_targets.append(negative_mood_tags.cpu())

                anchor_genre_loss = label_loss(predicted_anchor_genre_labels, anchor_genre_tags)
                positive_genre_loss = label_loss(predicted_positive_genre_labels, anchor_genre_tags)
                negative_genre_loss = label_loss(predicted_negative_genre_labels, negative_genre_tags)

                anchor_mood_loss = label_loss(predicted_anchor_mood_labels, anchor_mood_tags)
                positive_mood_loss = label_loss(predicted_positive_mood_labels, anchor_mood_tags)
                negative_mood_loss = label_loss(predicted_negative_mood_labels, negative_mood_tags)

                genre_classification_loss += anchor_genre_loss + positive_genre_loss + negative_genre_loss
                mood_classification_loss += anchor_mood_loss + positive_mood_loss + negative_mood_loss
            else:
                reconstructed, latents = model(all_segments, all_masks)

            cosine_total += loss_utils.cosine_similarity(reconstructed, all_segments)
            mse_total += loss_utils.mse(reconstructed, all_segments)

            anchor_latents, positive_latents, negative_latents = torch.split(latents, B, dim=0)

            anchor_positive_loss = loss_utils.cosine_similarity(anchor_latents, positive_latents)
            anchor_negative_loss = loss_utils.cosine_similarity(anchor_latents, negative_latents)

            triplets_total += F.relu(anchor_positive_loss - anchor_negative_loss + config.margin)

            num_batches += 1

    # Average over batches
    def safe_div(x):
        return x / num_batches if num_batches > 0 else 0.0

    return (
        safe_div(cosine_total),
        safe_div(mse_total),
        safe_div(triplets_total),
        safe_div(genre_classification_loss),
        safe_div(mood_classification_loss),

        all_genre_preds, all_genre_targets,
        all_mood_preds, all_mood_targets,
    )

