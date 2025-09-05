import random

import torch
import os
from torch.utils.data import Dataset


class StreamingSongDataset(Dataset):
    def __init__(self, song_dir: str, label_dir: str, transform=None, indices=None):
        self.song_files = sorted(os.listdir(song_dir))
        self.label_files = sorted(os.listdir(label_dir))
        if indices is not None:
            self.song_files = [self.song_files[i] for i in indices]
            self.label_files = [self.label_files[i] for i in indices]

        self.song_dir = song_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.song_files)

    def __getitem__(self, idx):
        song = torch.load(os.path.join(self.song_dir, self.song_files[idx]), map_location='cpu')
        label = torch.load(os.path.join(self.label_dir, self.label_files[idx]), map_location='cpu')

        if self.transform:
            song = self.transform(song.clone())

        return song, label


# The amount of data is just small enough to fit in memory :)
class AudioDataset(Dataset):
    def __init__(self, data_directory, tags_directory, transform=None):
        self.song_files = sorted(os.listdir(data_directory))
        self.label_files = sorted(os.listdir(tags_directory))
        self.transform = transform

        self.data = []
        self.tags = []

        for filename in self.song_files:
            self.data.extend(torch.load(os.path.join(data_directory, filename)))

        for filename in self.song_files:
            self.tags.extend(torch.load(os.path.join(tags_directory, filename)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.tags[idx]
        if self.transform:
            data = self.transform(data.clone())
        return data, labels


class AudioDatasetTriplets(Dataset):
    def __init__(self, data, masks, tags=None):
        """
        Args:
        - data: Tensor of shape (num_samples, seq_length, embed_dim)
        """
        self.data = data
        self.masks = masks
        self.tags = tags

        self.index_map = []
        for song_idx, segments in enumerate(self.data):
            for seg_idx in range(len(segments)):
                self.index_map.append((song_idx, seg_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        (song_idx, seg_idx) = self.index_map[idx]
        anchor_segment = self.data[song_idx][seg_idx]
        anchor_segment_mask = self.masks[song_idx][seg_idx]

        # Choose Random Segment within the same song
        num_segments = len(self.data[song_idx])
        possible_segment_indices = [i for i in range(num_segments) if i != seg_idx]
        positive_choice = random.choice(possible_segment_indices)

        positive_pair_segment = self.data[song_idx][positive_choice]
        positive_pair_segment_mask = self.masks[song_idx][positive_choice]

        # Choose Random Song and Random Segment within Song
        num_songs = len(self.data)
        possible_song_indicies = [i for i in range(num_songs) if i != song_idx]
        negative_choice_index = random.choice(possible_song_indicies)

        # Choose Random Segment within the same song
        num_segments = len(self.data[negative_choice_index])
        possible_segment_indices = [i for i in range(num_segments) if i != seg_idx]
        negative_choice = random.choice(possible_segment_indices)

        negative_pair_segment = self.data[negative_choice_index][negative_choice]
        negative_pair_segment_mask = self.masks[negative_choice_index][negative_choice]

        # Return Anchor, Positive, and Negative Choices
        return (anchor_segment, anchor_segment_mask,
         positive_pair_segment, positive_pair_segment_mask,
         negative_pair_segment, negative_pair_segment_mask,
                self.tags[song_idx],
                self.tags[negative_choice])

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * torch.normal(std=self.std, mean=self.mean)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x