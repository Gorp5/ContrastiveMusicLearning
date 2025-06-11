import torch
import torch.nn as nn
#from soft_dtw_pytorch import SoftDTW
import torch.nn.functional as F

# Cosine Similarity as the Reconstructive Loss
def reconstruction_loss(pred, target):
    # Range {0, 2}
    loss = 1 - torch.nn.functional.cosine_similarity(pred, target).mean(dim=-1)
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
    return F.mse_loss(pred, target) * (target.var() + 1e-6)

def mse_fft(pred, target):
    return F.mse_loss(torch.fft.rfft(pred), torch.fft.rfft(target)) * 0.2


def contrastive_loss(latents):
    latents = F.normalize(latents, dim=-1)
    sim_matrix = torch.matmul(latents, latents.T) / 0.1
    labels = torch.arange(latents.size(0)).to(latents.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def combined_loss(pred, target):
    rec_loss = reconstruction_loss(pred, target)
    cross_relation = cross_corr_fft(pred, target)
    mse_loss = mse(pred, target)
    #cross_fft = cross_relation_fft_2(pred, target)

    total_loss = rec_loss + cross_relation + mse_loss# + cross_fft

    return total_loss.mean()



def variational_loss(pred, target, mean, logvar, device="cpu", beta=1):
    reconstruction = combined_loss(pred, target)
    kld_loss = KLD(mean, logvar)

    return reconstruction + kld_loss * beta

def KLD(mean, logvar):
    return - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())