import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=0.0, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def set_gamma(self, gamma):
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        eps = 1e-4
        pt = torch.where(targets == 1, probs, 1 - probs)
        pt = pt.clamp(min=eps, max=1.0 - eps) # Avoiding NaN gradients caused by floating point error

        modulating_factor = (1 - pt) ** self.gamma
        loss = modulating_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss