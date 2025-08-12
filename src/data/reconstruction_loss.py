from torch import nn

from loss.loss_utils import cosine_similarity, mse


class ReconstructionLoss(nn.Module):
    def __init__(self, gamma=0.0):
        super().__init__()
        self.gamma = gamma

    def set_gamma(self, gamma):
        self.gamma = gamma

    def forward(self, pred, target):
        rec_loss = cosine_similarity(pred, target)
        mse_loss = mse(pred, target)

        total_loss = rec_loss + mse_loss

        eps = 1e-6

        modulating_factor = (1 - pt) ** self.gamma
        loss = modulating_factor * total_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss