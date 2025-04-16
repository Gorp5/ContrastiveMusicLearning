import torch
import torch.nn as nn


# ============ Reconstruction Loss Function ============ #
def reconstruction_loss(pred, target):
    # Range {0, 2}
    return (1 - nn.CosineSimilarity()(pred, target)).mean(dim=-1)

# ============ Contrastive Loss ============ #
def contrastive_loss(pred, target, temperature=0.5):
    batch_size = pred.shape[0]

    concat = torch.cat([pred, target], dim=0)
    norm = torch.norm(concat, dim=1)

    # Compute similarity matrix using cosine similarity.
    similarity_matrix = torch.matmul(norm, norm.T)

    # Scaling
    logits = similarity_matrix / temperature

    # Mask out self-similarity by setting the diagonal to a very negative value.
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=norm.device)
    logits.masked_fill_(mask, -1e9)

    # For each sample i in [0, batch_size-1], the positive pair is at index i + batch_size.
    # For each sample i in [batch_size, 2*batch_size-1], the positive pair is at index i - batch_size.
    positive_indices = torch.arange(batch_size, device=norm.device)
    labels = torch.cat([positive_indices + batch_size, positive_indices], dim=0)

    # Now logits has shape (2B, 2B) and labels is (2B,)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss  / 60000.0  # Normalize to a value close to 1


# ============ Combined Loss Function ============ #
def combined_loss(pred, target):
    rec_loss = reconstruction_loss(pred, target)
    contr_loss = contrastive_loss(pred, target)

    total_loss = rec_loss + contr_loss

    return total_loss.mean()




