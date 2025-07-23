import random

import torch
from torch.utils.data import Dataset
import os
import numpy as np


# ==== Custom Dataset ====
class AudioDataset(Dataset):
    def __init__(self, data):
        """
        Args:
        - data: Tensor of shape (num_samples, seq_length, embed_dim)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AudioDatasetMask(Dataset):
    def __init__(self, data, masks):
        """
        Args:
        - data: Tensor of shape (num_samples, seq_length, embed_dim)
        """
        self.data = data
        self.masks = masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.masks[idx]

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


class AudioDatasetClassification(Dataset):
    def __init__(self, data, masks, tags):
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

        return (anchor_segment, anchor_segment_mask, self.tags[song_idx])


def chunk_song(path, sample_length, features):
    data = np.load(path)

    # Fill with padding
    transposed = data.transpose()
    end_tokens = (transposed.shape[0] % sample_length)
    if end_tokens == 0:
        end_tokens = sample_length

    zeros = np.zeros((sample_length - end_tokens, features))
    padded_data = np.concatenate((transposed, zeros))

    return padded_data, len(zeros)


def read_data_from_folder(folder_stub, features=64, start=0, count=-1, keep_song_data_option=False, sample_length=256):
    dataset = []
    masks = []
    keys = []

    all_folders = os.listdir(folder_stub)
    if count > 0:
        all_folders = all_folders[start:start + count]

    for each_song in all_folders:
        path = os.path.join(folder_stub, each_song)

        padded_data, mask = chunk_song(path, sample_length, features)

        if not keep_song_data_option:
            dataset.extend(np.split(padded_data, padded_data.shape[0] / sample_length))
            final_mask = [0 for _ in range(padded_data.shape[0] // sample_length - 1)]
            final_mask.append(mask)
            masks.extend(final_mask)
        else:
            song_name = each_song[:-4]
            dataset.append(torch.tensor(np.array(np.split(padded_data, padded_data.shape[0] / sample_length))))
            final_mask = [0 for _ in range(padded_data.shape[0] // sample_length - 1)]
            final_mask.append(mask)
            index = padded_data.shape[0] // sample_length
            masks.append(final_mask)
            keys.extend([song_name] * index)

    tens = dataset
    masks = masks
    return tens, masks, keys


def retrieve_data(path_stub, features=64, start=0, count=-1, sample_length=256, keep_song_data_option=False):
    return read_data_from_folder(path_stub, features=features, start=start, count=count, sample_length=sample_length, keep_song_data_option=keep_song_data_option)

