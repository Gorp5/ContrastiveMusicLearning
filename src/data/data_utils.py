import random

import librosa
import torch
import os
import csv
import numpy as np
import torch.nn.functional as F

from datasets import tqdm
from librosa.feature import melspectrogram
from torch.utils.data import Dataset

def get_melspec_from_file(full_path):
    #Same thing here
    try:
        audio, sr = librosa.load(full_path, sr=44100, mono=True)

        win_length = int(round(0.025 * sr))  # ~1103 samples
        hop_length = int(round(0.010 * sr))  # 441 samples
        n_fft = 2048

        data = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            n_mels=128, fmin=0, fmax=sr / 2, power=2.0
        )

        return librosa.amplitude_to_db(data, ref=np.max)
    except Exception as e:
        return None

def one_hot_encode(label, num_classes):
    return F.one_hot(torch.tensor(label), num_classes=num_classes).float()

class LatentDataset(Dataset):
    def __init__(self, data_directory, num_classes=10):
        files = sorted(os.listdir(data_directory))

        self.latents = []
        self.labels = []
        self.num_classes = num_classes

        for file in files:
            type = file[-8:-3]

            if type == "label":
                label = torch.load(os.path.join(data_directory, file), weights_only=False)
                self.labels.append(label)
            else:
                data = os.path.join(data_directory, file)
                self.latents.append(data)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latent = self.latents[idx]
        l = torch.load(latent, weights_only=False)

        if l.shape[0] == 1:
            l = l.squeeze(0)

        latent = np.array(l)

        return self.labels[idx], latent

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
        self.ids = []

        self.id_to_spectrograms = {}

        mp3_dir = os.path.join(data_directory, "mp3_all\\")
        mel_spec_dir = os.path.join(data_directory, "mel_spec_all\\")

        with open(os.path.join(data_directory, "annotations_final.csv")) as file:
            csv_reader = csv.reader(file)

            tags = []

            header = True
            tq = tqdm(csv_reader, total=25863)
            for row in csv_reader:
                tq.update(1)

                row = row[0].split('\t')
                if header:
                    header = False
                    tags = row
                    continue

                id = int(row[0])

                path = row[-1].strip("\"")

                tags = [int(x.strip("\"")) for x in row[1:-1]]

                mp3_path = os.path.join(mp3_dir, path)
                mel_spec_path = os.path.join(mel_spec_dir, path[:-4] + ".npy")

                if not os.path.exists(mel_spec_path):
                    mel_spec = get_melspec_from_file(mp3_path)
                    if mel_spec is None:
                        continue

                    np.save(mel_spec_path, mel_spec)

                self.id_to_tags[id] = tags
                self.ids.append(id)

                self.id_to_spectrograms[id] = mel_spec_path

        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        path = self.id_to_spectrograms[id]
        mel_spec = np.load(path)
        return self.id_to_tags[id], mel_spec


class StreamViewDataset(Dataset):
    def __init__(self, data_directory: str, chunk_size=256, views=2, min=-1, max=-1):
        self.song_folder = os.path.join(data_directory, "data\\")

        self.count = 0
        self.chunk_size = chunk_size
        self.view_count = views

        self.min = min
        self.max = max

        self.stochastic = self.max > 0 and self.min > 0

        self.ids = []
        self.paths = []

        for file in sorted(os.listdir(self.song_folder)):
            id = file.split(".")[0]
            full_path = os.path.join(self.song_folder, file)

            self.ids.append(int(id))
            self.paths.append(full_path)
            self.count += 1

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        song_path = self.paths[idx]
        id = self.ids[idx]

        full_spectrogram = torch.load(song_path, weights_only=False)

        views = []
        masks = []

        for view_i in range(self.view_count):
            chunk_size = self.chunk_size

            if self.stochastic:
                chunk_size = random.randint(self.min, self.max)

            possible_starts = full_spectrogram.shape[1] - chunk_size
            start = random.randint(0, possible_starts)
            spec = torch.from_numpy(full_spectrogram[:, start:start + chunk_size])

            if self.stochastic:
                pad_len = self.max - spec.size(-1)
                # pad to the right (last dimension)
                spec = F.pad(spec, (0, pad_len))
                # 1 = valid, 0 = pad
                mask = torch.zeros(self.max, dtype=torch.bool)
                mask[:chunk_size] = True
            else:
                # no padding, full mask
                mask = torch.ones(chunk_size, dtype=torch.bool)
                
            views.append(spec)
            masks.append(mask)

        return idx, views, masks


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