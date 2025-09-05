import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
#import torchsort  # pip install torchsort

# Cosine Similarity as the Reconstructive Loss
def cosine_similarity(pred, target):
    # Range {0, 2}
    loss = 1 - torch.nn.functional.cosine_similarity(pred, target).mean(dim=-1)
    return loss.mean()



class DifferentiablePRAUCLoss(nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau  # temperature for softmax smoothing

    def forward(self, y_score, y_true):
        """
        y_score: [batch, num_labels] raw model scores (logits)
        y_true:  [batch, num_labels] binary {0,1} tensor
        """
        # Flatten multi-label
        scores = y_score.view(-1)
        targets = y_true.view(-1).float()
        pos_idx = torch.nonzero(targets == 1, as_tuple=False).view(-1)
        neg_idx = torch.nonzero(targets == 0, as_tuple=False).view(-1)
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            return torch.tensor(0., device=y_score.device, requires_grad=True)

        s_pos = scores[pos_idx]  # positive scores
        s_neg = scores[neg_idx]  # negative scores

        # Pairwise score differences
        diff = s_neg.unsqueeze(0) - s_pos.unsqueeze(1)  # shape [P, N]

        # Apply smoothing via log-sigmoid (differentiable version of Heaviside)
        P = torch.sigmoid(diff / self.tau)

        # For each positive, how many negatives outrank it
        rank_pos = 1 + P.sum(dim=1)
        # Ideal rank if scores perfectly separate: 1...P
        ideal = torch.arange(1, rank_pos.size(0) + 1, device=rank_pos.device, dtype=rank_pos.dtype)

        # Precision-like term: proportion of positives above each positive
        precision_at_k = ideal / rank_pos

        # Differentiate PR-AUC proxy: average of precision_at_k
        loss = 1 - precision_at_k.mean()
        return loss

def cross_relation_2(prediction: torch.Tensor, input: torch.Tensor, eps: float = 1e-9, lambda_offdiag: float = 0.005):
    """
    Cross-Relation Loss between prediction and input.

    Args:
        prediction: (B, D) - model output embeddings
        input: (B, D) - original input embeddings
        eps: small constant to prevent division by zero
        lambda_offdiag: weight for off-diagonal loss term

    Returns:
        Scalar tensor representing the cross-relation loss
    """
    B = prediction.size(0)
    # Flatten last two dims
    pred_flat = prediction.view(B, -1)
    target_flat = input.view(B, -1)

    # Normalize embeddings to zero mean and unit variance per feature dim
    pred_norm = (pred_flat - pred_flat.mean(dim=0)) / (pred_flat.std(dim=0) + eps)
    target_norm = (target_flat - target_flat.mean(dim=0)) / (target_flat.std(dim=0) + eps)

    # Cross-correlation matrix (D x D)
    c = torch.mm(pred_norm.T, target_norm) / B

    # On-diagonal should be 1
    on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
    # Off-diagonal should be 0
    off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).sum()

    loss = on_diag + lambda_offdiag * off_diag
    return loss

def cross_relation_fft_2(prediction, input, eps=1e-9, lambda_offdiag=0.005):
    # Compute FFT along the feature/time dimension, e.g. last dim
    pred_fft = torch.fft.fft(prediction, dim=-1).real  # or .abs(), depending on use case
    input_fft = torch.fft.fft(input, dim=-1).real

    # Then apply same cross relation loss on these transformed tensors
    return cross_relation_2(pred_fft, input_fft, eps, lambda_offdiag)

# Cross Correlation Loss
def cross_corr_fft(pred, target):
    B, T, F = pred.shape

    # reshape & z-score each (batch,feature) sequence
    pred = pred.transpose(1, 2).reshape(B * F, T)
    target = target.transpose(1, 2).reshape(B * F, T)

    # Z_Score Normalization
    pred = (pred - pred.mean(1,keepdim=True)) / (pred.std(1,keepdim=True) + 1e-8)
    target = (target - target.mean(1,keepdim=True)) / (target.std(1,keepdim=True) + 1e-8)

    # full FFT cross-correlation
    fft_size = 2 * T
    p_fft = torch.fft.rfft(pred,  n=fft_size)
    t_fft = torch.fft.rfft(target, n=fft_size)
    cc_full = torch.fft.irfft(p_fft.conj() * t_fft, n=fft_size)

    # valid lags: [-(T-1) .. +(T-1)], center lag=0
    cc = cc_full[:, :2 * T-1]
    cc = torch.roll(cc, shifts=T - 1, dims=1)

    # normalize by t
    cc = cc / T

    # take peak corr ∈ [–1,1], and form loss=1–peak
    max_corr = cc.max(dim=1).values       # [B*F]
    max_corr = max_corr.view(B, F)       # reshape
    loss = 1 - max_corr.mean(dim=1)      # [B]
    return loss.mean()                   # scalar

def mse(pred, target):
    return F.mse_loss(pred, target) / (target.var() + 1e-6)

def mse_fft(pred, target):
    return F.mse_loss(torch.fft.rfft(pred), torch.fft.rfft(target)) * 0.2


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
        z1, z2: [B, D] - Two views of the same batch
        Returns NT-Xent loss for a batch size B
        """
    B = z_i.size(0)

    # Normalize projections
    z1 = F.normalize(z_i, dim=1)
    z2 = F.normalize(z_j, dim=1)

    # Concatenate embeddings
    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    # Compute cosine similarity matrix
    sim = torch.matmul(z, z.T)  # [2B, 2B]
    sim = sim / temperature

    # Mask out self-similarities
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -float('inf'))

    # Positive pairs are (i, i + B) and (i + B, i)
    positives = torch.cat([
        torch.arange(B, 2 * B),
        torch.arange(0, B)
    ], dim=0).to(z.device)

    loss = F.cross_entropy(sim, positives)
    return loss.mean()


def combined_loss(pred, target):
    rec_loss = cosine_similarity(pred, target)
    #cross_relation = cross_corr_fft(pred, target)
    mse_loss = mse(pred, target)
    #contrastive = nt_xent_loss(pred, target)

    total_loss = rec_loss + mse_loss# cross_relation + # + cross_fft

    return total_loss



def variational_loss(pred, target, mean, logvar, device="cpu", beta=1):
    reconstruction = combined_loss(pred, target)
    kld_loss = KLD(mean, logvar)

    return reconstruction + kld_loss * beta

def KLD(mean, logvar):
    return - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

def kl_divergence_decomposed(mu, logvar):
    """
    Computes KL divergence and splits into mean/variance components.
    Returns per-sample KL [batch_size], and the mean and var terms separately.
    """
    kl_mean_term = 0.5 * mu.pow(2)  # [batch, latent_dim]
    kl_var_term = 0.5 * (logvar.exp() - logvar - 1)  # [batch, latent_dim]
    kl_total = kl_mean_term + kl_var_term  # [batch, latent_dim]

    return kl_total, kl_mean_term, kl_var_term


def compute_vae_loss(recon_x, x, mu, logvar, step_fraction, beta=1.0,
                     C_max=30.0, free_bits=0.1,
                     loss_weights=None):
    """
    recon_x: reconstructed input
    x: original input
    mu, logvar: VAE parameters
    step_fraction: current training step fraction (e.g., epoch / total_epochs)
    beta: scaling coefficient for KL term
    C_max: maximum KL capacity
    free_bits: minimum bits per dim (in nats)
    loss_weights: dict to weight different loss components
    """

    if loss_weights is None:
        loss_weights = {
            "recon": 1.0,
            "cosine": 1.0,
            "fft_corr": 1.0,
            "contrastive": 1.0
        }

    # --- Reconstruction Loss (e.g., MSE) ---
    mse_loss = mse(recon_x, x).mean()

    # --- FFT or Cross Correlation loss (add if needed) ---
    fft_corr = 0.0  # placeholder

    # --- Cosine similarity between inputs and reconstructions ---
    cosine_sim = cosine_similarity(recon_x, x).mean()

    # --- KL Divergence (decomposed) ---
    kl_total, kl_mean_term, kl_var_term = kl_divergence_decomposed(mu, logvar)

    # Optional: enforce free bits
    kl_clamped = torch.clamp(kl_total, min=free_bits)  # [batch, latent_dim]
    kl_sum = kl_clamped.sum(dim=1).mean()  # scalar

    # Capacity annealing
    C = C_max * step_fraction
    kl_loss = beta * torch.abs(kl_sum - C)

    # Combine losses
    total_loss = (
        loss_weights["recon"] * mse_loss +
        loss_weights["cosine"] * cosine_sim +
        kl_loss
    )

    return {
        "total_loss": total_loss,
        "kl_loss": kl_loss.item(),
        "kl_sum": kl_sum.item(),
        "kl_mean_term": kl_mean_term.mean().item(),
        "kl_var_term": kl_var_term.mean().item(),
        "mse": mse_loss.item(),
        "cosine": cosine_sim.item()
    }

def triplet_loss(anchor, positive, negative):
    anchor_positive_loss = cosine_similarity(anchor, positive).mean()
    anchor_negative_loss = cosine_similarity(anchor, negative).mean()

    return anchor_positive_loss + torch.abs(1 - anchor_negative_loss)