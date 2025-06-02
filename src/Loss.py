import torch
import torch.nn as nn
#from soft_dtw_pytorch import SoftDTW

# Cosine Similarity as the Reconstructive Loss
def reconstruction_loss(pred, target):
    # Range {0, 2}
    return (1 - nn.CosineSimilarity()(pred, target)).mean(dim=-1)


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


def combined_loss(pred, target):
    rec_loss = reconstruction_loss(pred, target)
    contr_loss = cross_corr_fft(pred, target)

    total_loss = rec_loss + contr_loss

    return total_loss.mean()