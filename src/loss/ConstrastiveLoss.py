# from libauc.losses import GCLoss_v1
import torch
import torch.nn.functional as F
import torch.nn as nn

# class InfoCNELoss(nn.Module):
#     def __init__(self, N=9900):
#         super().__init__()
#         self.criterion = GCLoss_v1(
#             N=N,
#             tau=0.1,
#             gamma=0.9,
#             gamma_schedule='constant',
#             distributed=False,  # args.world_size > 1,
#             gamma_decay_epochs=10,
#             eps=1e-8,
#             enable_isogclr='store_true'
#         )
#     def forward(self, indicies, a, b):
#         return self.criterion(a, b, indicies)

class InfoCNELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.1):
        super(InfoCNELoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Compute the Generalized Contrastive Loss v1.

        Args:
            embeddings1 (torch.Tensor): First set of embeddings of shape (B, F).
            embeddings2 (torch.Tensor): Second set of embeddings of shape (B, F).
            index (torch.Tensor): Indices of positive pairs of shape (B,).

        Returns:
            torch.Tensor: The contrastive loss value.
        """
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=-1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=-1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature

        # Create labels for positive pairs
        labels = index

        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
#
# class InfoCNELoss(nn.Module):
#     def __init__(self, N=9900):
#         super().__init__()
#         self.criterion = GCLoss_v1(
#             N=N,
#             tau=0.1,
#             gamma=0.9,
#             gamma_schedule='constant',
#             distributed=False,  # args.world_size > 1,
#             gamma_decay_epochs=10,
#             eps=1e-8,
#             enable_isogclr='store_true'
#         )
#     def forward(self, indicies, a, b):
#         return self.criterion(a, b, indicies)