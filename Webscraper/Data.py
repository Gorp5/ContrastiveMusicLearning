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


def read_data_from_folder(folder_stub, start=0, count=-1, keep_song_data_option=False, sample_length=256):
    dataset = []
    all_folders = os.listdir(folder_stub)
    if count > 0:
        all_folders = all_folders[start:start + count]

    for each_song in all_folders:
        data = np.load(os.path.join(folder_stub, each_song))

        if data.shape[0] == 8192:
            pass

        # Fill with padding
        transposed = data.transpose()
        zeros = np.zeros((sample_length - (transposed.shape[0] % sample_length), 64))
        padded_data = np.concatenate((transposed, zeros))

        if not keep_song_data_option:
            dataset.extend(np.split(padded_data, padded_data.shape[0] / sample_length))
        else:
            dataset.append(torch.tensor(np.split(padded_data, padded_data.shape[0] / sample_length)))
    
    #ar = np.array(dataset)
    #del dataset
    tens = torch.tensor(np.array(dataset)).to(torch.float32)
    return tens


def retrieve_data(path_stub, song_path_stub, start=0, count=-1, sample_length=256):
    dataset_saved_path = path_stub + "song-dataset.npy"

    # Read and Memoize Training and Test Datasets
    # if not os.path.exists(dataset_saved_path):
    dataset = read_data_from_folder(path_stub + song_path_stub, start=start, count=count, sample_length=sample_length)
        # np.save(dataset_saved_path, dataset)
    # else:
    #     dataset = torch.tensor(np.load(dataset_saved_path))

    # dataset = dataset.to(torch.float32)
    return dataset
    # reconstruction_path_stub = path_stub + "reconstruction_test_latents\\"
    # reconstruction_test_saved_path = path_stub + "reconstruction-dataset.npy"
    #
    # if not os.path.exists(reconstruction_test_saved_path):
    #     # Read and Memoize Training and Test Datasets
    #     reconstruction_dataset = read_data_from_folder(song_path_stub, sample_length=sample_length, keep_song_data_option=True)
    #     np.save(reconstruction_path_stub, reconstruction_dataset)
    # else:
    #     reconstruction_dataset = np.load(song_path_stub)
    # reconstruction_dataset = reconstruction_dataset.to(torch.float32)
    # #
    # return reconstruction_dataset

def getDatasets():
    full_dataset, reconstruction_examples = retrieve_data()
