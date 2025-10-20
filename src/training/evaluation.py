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

class MLP(nn.Module):
    def __init__(self, input=128, output=10, dropout=0.0):
        super().__init__()
        model = nn.Sequential(
            nn.Linear(input, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, output)
        )

    def forward(self, x):
        return self.model(x)

class LinearProbe(nn.Module):
    def __init__(self, input=128, output=10):
        super().__init__()
        self.model = nn.Linear(input, output)

    def forward(self, x):
        return self.model(x)

def train_epoch_embeddings(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, args: argparse.Namespace):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []

    for indices, inputs, labels in train_loader:
        inputs = inputs.to(args.device, dtype=torch.float)
        labels = labels.to(args.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        all_targets.append(labels.tolist())
        all_predictions.append(outputs.tolist())


    all_targets, all_predictions = cat_predictions_targets(all_targets, all_predictions)

    # Compute metrics
    train_metrics = compute_metrics(all_targets, all_predictions, args)
    primary_metric = select_primary_metric(train_metrics, args)

    return train_metrics, primary_metric