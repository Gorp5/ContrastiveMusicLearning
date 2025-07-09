import torch
from tqdm import tqdm

from loss import loss_utils
import torch.nn.functional as F

def evaluate(
    model,
    dataloader,
    config,
    beta=1.0,
    device="cuda",
    dtype=torch.float32
):
    model.eval()

    cosine_total = 0.0
    mse_total = 0.0
    triplets_total = 0.0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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
        safe_div(triplets_total)
    )