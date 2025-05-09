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


def chunk_song(path, sample_length):
    data = np.load(path)

    # Fill with padding
    transposed = data.transpose()
    zeros = np.zeros((sample_length - (transposed.shape[0] % sample_length), 64))
    padded_data = np.concatenate((transposed, zeros))
    return padded_data


def read_data_from_folder(folder_stub, start=0, count=-1, keep_song_data_option=False, sample_length=256):
    dataset = []
    all_folders = os.listdir(folder_stub)
    if count > 0:
        all_folders = all_folders[start:start + count]

    for each_song in all_folders:
        path = os.path.join(folder_stub, each_song)

        padded_data = chunk_song(path, sample_length)

        if not keep_song_data_option:
            dataset.extend(np.split(padded_data, padded_data.shape[0] / sample_length))
        else:
            dataset.append(torch.tensor(np.split(padded_data, padded_data.shape[0] / sample_length)))

    tens = torch.tensor(dataset).to(torch.float32)
    return tens


def retrieve_data(path_stub, song_path_stub, start=0, count=-1, sample_length=256, keep_song_data_option=False):
    dataset = read_data_from_folder(path_stub + song_path_stub, start=start, count=count, sample_length=sample_length, keep_song_data_option=keep_song_data_option)
    return dataset

