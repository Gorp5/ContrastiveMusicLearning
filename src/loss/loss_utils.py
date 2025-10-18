import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
#import torchsort  # pip install torchsort

# Cosine Similarity as the Reconstructive Loss
def cosine_similarity(pred, target):
    loss = 1 - torch.nn.functional.cosine_similarity(pred, target).mean(dim=-1)
    return loss.mean()

class DifferentiablePRAUCLoss(nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau  # temperature for softmax smoothing

    def forward(self, y_score, y_true):
        scores = y_score.view(-1)
        targets = y_true.view(-1).float()
        pos_idx = torch.nonzero(targets == 1, as_tuple=False).view(-1)
        neg_idx = torch.nonzero(targets == 0, as_tuple=False).view(-1)
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            return torch.tensor(0., device=y_score.device, requires_grad=True)

        s_pos = scores[pos_idx]  # positive scores
        s_neg = scores[neg_idx]  # negative scores

        diff = s_neg.unsqueeze(0) - s_pos.unsqueeze(1)  # shape [P, N]

        P = torch.sigmoid(diff / self.tau)

        rank_pos = 1 + P.sum(dim=1)
        ideal = torch.arange(1, rank_pos.size(0) + 1, device=rank_pos.device, dtype=rank_pos.dtype)

        precision_at_k = ideal / rank_pos

        loss = 1 - precision_at_k.mean()
        return loss

def cross_relation_2(prediction: torch.Tensor, input: torch.Tensor, eps: float = 1e-9, lambda_offdiag: float = 0.005):
    B = prediction.size(0)
    pred_flat = prediction.view(B, -1)
    target_flat = input.view(B, -1)

    pred_norm = (pred_flat - pred_flat.mean(dim=0)) / (pred_flat.std(dim=0) + eps)
    target_norm = (target_flat - target_flat.mean(dim=0)) / (target_flat.std(dim=0) + eps)

    c = torch.mm(pred_norm.T, target_norm) / B

    on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
    off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).sum()

    loss = on_diag + lambda_offdiag * off_diag
    return loss

def cross_relation_fft_2(prediction, input, eps=1e-9, lambda_offdiag=0.005):
    pred_fft = torch.fft.fft(prediction, dim=-1).real  # or .abs(), depending on use case
    input_fft = torch.fft.fft(input, dim=-1).real

    return cross_relation_2(pred_fft, input_fft, eps, lambda_offdiag)

# Cross Correlation Loss
def cross_corr_fft(pred, target):
    B, T, F = pred.shape

    pred = pred.transpose(1, 2).reshape(B * F, T)
    target = target.transpose(1, 2).reshape(B * F, T)

    pred = (pred - pred.mean(1,keepdim=True)) / (pred.std(1,keepdim=True) + 1e-8)
    target = (target - target.mean(1,keepdim=True)) / (target.std(1,keepdim=True) + 1e-8)

    fft_size = 2 * T
    p_fft = torch.fft.rfft(pred,  n=fft_size)
    t_fft = torch.fft.rfft(target, n=fft_size)
    cc_full = torch.fft.irfft(p_fft.conj() * t_fft, n=fft_size)

    cc = cc_full[:, :2 * T-1]
    cc = torch.roll(cc, shifts=T - 1, dims=1)

    cc = cc / T

    max_corr = cc.max(dim=1).values       # [B*F]
    max_corr = max_corr.view(B, F)       # reshape
    loss = 1 - max_corr.mean(dim=1)      # [B]
    return loss.mean()                   # scalar

def mse(pred, target):
    return F.mse_loss(pred, target) / (target.var() + 1e-6)

def mse_fft(pred, target):
    return F.mse_loss(torch.fft.rfft(pred), torch.fft.rfft(target)) * 0.2


def nt_xent_loss(z_i, z_j, temperature=0.5):
    B = z_i.size(0)

    z1 = F.normalize(z_i, dim=1)
    z2 = F.normalize(z_j, dim=1)

    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    sim = torch.matmul(z, z.T)  # [2B, 2B]
    sim = sim / temperature

    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -float('inf'))

    positives = torch.cat([
        torch.arange(B, 2 * B),
        torch.arange(0, B)
    ], dim=0).to(z.device)

    loss = F.cross_entropy(sim, positives)
    return loss.mean()


class ConvexCombinationLoss(nn.Module):

    def __init__(self, num_augmentations: int, device="cuda"):
        super().__init__()
        self.num_augmentations = num_augmentations

        self.c_tilde = nn.Parameter(torch.randn(num_augmentations)).to(device=device)

    def forward(self, z_aug: torch.Tensor, z_orig: torch.Tensor):

        c = F.softmax(self.c_tilde, dim=0)  # shape: (p,)

        z_combined = torch.sum(c.view(1, -1, 1) * z_aug, dim=1)  # (B, D)
        loss = F.mse_loss(z_combined, z_orig)

        return loss

def distribution_similarity_loss(mean_1, logvar_1, mean_2, logvar_2):
    mean_means = torch.mean(mean_1 + mean_2) / 2
    mean_logvar = torch.mean(logvar_1.exp() + logvar_2.exp()) / 2

    return -0.5 * (logvar_1 - logvar_2) + 0.25 * (torch.square(mean_1 - mean_means) + torch.square(mean_2 - mean_means)) / torch.square(mean_logvar)

def distribution_normalizing_loss(mean, logvar, reduction='mean', free_bits=0.1):
    kld_per_sample = -0.5 * (1 + logvar - torch.square(mean) - logvar.exp())

    if reduction == 'sum':
        return kld_per_sample.sum()
    elif reduction == 'none':
        return kld_per_sample
    return kld_per_sample.mean()

def KLD(mean, logvar, reduction='mean', free_bits=0.1):
    # kld_per_dim = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)  # shape [B, D]
    # kld_per_sample = torch.clamp(kld_per_dim, min=free_bits).sum(dim=1)

    kld_per_sample = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    if reduction == 'sum':
        return kld_per_sample.sum()
    elif reduction == 'none':
        return kld_per_sample
    return kld_per_sample.mean()