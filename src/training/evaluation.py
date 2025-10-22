import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn, optim
from torch.utils.data import DataLoader

def local_coherence(embeddings: np.ndarray, labels: np.ndarray, k=50):
    N = embeddings.shape[0]

    # Find k+1 nearest neighbors because the first neighbor is the point itself
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(embeddings)
    distances, neighbors = nn.kneighbors(embeddings)

    # Remove self from neighbors
    neighbors = neighbors[:, 1:]

    # Compute fraction of neighbors with the same label
    matches = (labels[neighbors] == labels[:, None]).astype(float)
    fractions = matches.mean(axis=1)

    # Average over all samples
    return fractions.mean()
