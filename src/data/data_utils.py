import random

import librosa
import numpy as np
import torch
import os
import csv

from datasets import tqdm
from librosa.feature import melspectrogram
from torch.utils.data import Dataset
from collections import defaultdict
from mtgjamendodataset.scripts import commons

def get_melspec_from_file(full_path):
    audio, sr = librosa.load(full_path, sr=44100, mono=True)

    win_length = int(round(0.025 * sr))  # ~1103 samples
    hop_length = int(round(0.010 * sr))  # 441 samples
    n_fft = 2048

    data = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        n_mels=128, fmin=0, fmax=sr / 2, power=2.0
    )

    return librosa.amplitude_to_db(data, ref=np.max)

class LatentDataset(Dataset):
    def __init__(self, data_directory):
        files = sorted(os.listdir(data_directory))

        self.latents = []
        self.labels = []

        for file in files:
            type = file[-8:-3]

            if type == "label":
                self.labels.append(file)
            else:
                self.latents.append(file)
    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.labels[idx], self.latents[idx]

class GTZAN(Dataset):
    def __init__(self, data_directory):
        self.genre_folders = sorted(os.listdir(data_directory))

        self.data = []
        self.tags = []

        self.spectrograms = []
        self.tags = []

        for index in tqdm(range(1000)):

            if index == 554:
                continue

            tag = index // 100
            index_in_genre = index % 100

            self.tags.append(tag)
            folder_path = os.path.join(data_directory, self.genre_folders[tag])

            songs= sorted(os.listdir(folder_path))
            song_path = os.path.join(folder_path, songs[index_in_genre])

            mel_spec = get_melspec_from_file(song_path)

            self.spectrograms.append(mel_spec)

    def __len__(self):
        return 999

    def __getitem__(self, idx):
        return self.tags[idx], self.spectrograms[idx]

class MTAT(Dataset):
    def __init__(self, data_directory, transform=None):
        self.id_to_tags = {}
        self.id_to_path = {}
        self.ids = []

        self.spectrograms = {}

        with open(os.path.join(data_directory, "annotations_final.csv")) as file:
            csv_reader = csv.reader(file)

            for row in csv_reader:
                name = row[-1]
                id = int(row[0])

                self.id_to_tags[id] = [int(x) for x in row[1:-1]]
                self.id_to_path[id] = os.path.join(data_directory, name)

                self.ids.append(id)

                mel_spec = get_melspec_from_file(self.id_to_path[id])

                self.spectrograms[id] = mel_spec

        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        return self.id_to_tags[id], self.spectrograms[id]


class StreamViewDataset(Dataset):
    def __init__(self, data_directory: str, chunk_size=256, views=2):
        self.song_folders = sorted(os.listdir(data_directory))

        outer_folders = sorted(os.listdir(data_directory))

        self.count = 0
        self.chunk_size = chunk_size
        self.view_count = views

        self.ids = []
        self.paths = []

        for folder in outer_folders:
            directory_path = os.path.join(data_directory, folder)
            for file in sorted(os.listdir(directory_path)):
                id = file.split(".")[0]
                full_path = os.path.join(directory_path, file)

                self.ids.append(int(id))
                self.paths.append(full_path)
                self.count += 1

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        song_path = self.paths[idx]
        id = self.ids[idx]

        full_spectrogram = np.load(song_path)
        possible_starts = full_spectrogram.shape[1] - self.chunk_size

        views = []

        for view_i in range(self.view_count):
            start = random.randint(0, possible_starts)
            views.append(torch.from_numpy(full_spectrogram[:, start:start + self.chunk_size]))

        return idx, views


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