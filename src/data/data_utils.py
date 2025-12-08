import random

import librosa
import torch
import os
import csv
import numpy as np
import torch.nn.functional as F
import re

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

        latent = torch.from_numpy(l)

        return self.labels[idx], latent

class GTZAN(Dataset):
    def __init__(self, data_directory, split="train"):
        data_directory = os.path.join(data_directory, "genres_original\\")
        self.genre_folders = sorted(os.listdir(data_directory))

        self.data = []
        self.tags = []

        self.split = split

        self.idx_to_spectrograms = []
        self.genre_to_num = {}

        genre_index = 0

        mel_spec_dir = os.path.join(data_directory, "mel_spec_all\\")

        with open(os.path.join(data_directory, split + "_filtered.txt")) as f:
            for line in tqdm(f):

                parts = line.split("/")
                tag = parts[0]
                path = parts[1].strip('\n')

                if tag not in self.genre_to_num.keys():
                    self.genre_to_num[tag] = genre_index
                    genre_index += 1

                tag_index = self.genre_to_num[tag]

                folder_path = os.path.join(data_directory, self.genre_folders[tag_index])
                song_path = os.path.join(folder_path, path)

                mel_spec_path = os.path.join(mel_spec_dir, path[:-4] + ".npy")

                if not os.path.exists(mel_spec_path):
                    mel_spec = get_melspec_from_file(song_path)
                    if mel_spec is None:
                        continue

                    np.save(mel_spec_path, mel_spec)

                self.tags.append(tag_index)
                self.idx_to_spectrograms.append(mel_spec_path)

    def __len__(self):
        return len(self.idx_to_spectrograms)

    def __getitem__(self, idx):
        tag = self.tags[idx]

        path = self.idx_to_spectrograms[idx]
        mel_spec = np.load(path)

        return tag, torch.from_numpy(mel_spec)


class MTAT(Dataset):
    def __init__(self, data_directory, split="train"):
        self.id_to_tags = {}
        self.ids = []

        self.id_to_spectrograms = {}

        self.valid_dir_chars = ""
        if split == "train":
            self.valid_dir_chars = "0123456789ab"
        elif split == "valid":
            self.valid_dir_chars = "c"
        elif split == "test":
            self.valid_dir_chars = "def"

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

                directory_char = path[0]

                if directory_char not in self.valid_dir_chars:
                    continue

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

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        path = self.id_to_spectrograms[id]
        mel_spec = np.load(path)
        return self.id_to_tags[id], torch.from_numpy(mel_spec)


class GS(Dataset):
    def __init__(self, data_directory, split=None):
        self.id_to_key = {}
        self.ids = []
        self.key_to_num = {"a major": 0,
                        "a minor:": 1,
                        "a# major": 2,
                        "a# minor": 3,
                        "a-a# minor": 4,
                        "ab major": 5,
                        "ab minor": 6,
                        "b major": 7,
                        "b minor": 8,
                        "bb major": 9,
                        "bb minor": 10,
                        "c major": 11,
                        "c minor": 12,
                        "c# major": 13,
                        "c# minor": 14,
                        "d major": 15,
                        "d minor": 16,
                        "d# major": 17,
                        "d# minor": 18,
                        "db major": 19,
                        "db minor": 20,
                        "e major": 21,
                        "e minor": 22,
                        "eb major": 23,
                        "eb minor": 24,
                        "f major": 25,
                        "f minor": 26,
                        "f# major": 27,
                        "f# minor": 28,
                        "g major": 29,
                        "g minor": 30,
                        "g# major": 31,
                        "g# minor": 32,
                        "gb major": 33,
                        "gb minor": 34}

        self.id_to_spectrograms = {}

        mp3_dir = os.path.join(data_directory, "audio\\")
        key_dir = os.path.join(data_directory, "annotations\\key\\")

        mel_spec_dir = os.path.join(data_directory, "mel_spec_all\\")

        key_num = 35

        use_split = split is not None
        if split is not None:
            split_ids = []
            with open(os.path.join(data_directory, f"{split}_ids.txt")) as f:
                for line in f:
                    split_ids.append(int(line))

        for row in tqdm(os.listdir(mp3_dir)):
            mp3_path = os.path.join(mp3_dir, row)
            path_sections = row.split(".")
            id = int(path_sections[0])

            if use_split and id not in split_ids:
                continue

            path_snub = path_sections[0] + "." + path_sections[1]
            annotation_path = os.path.join(key_dir, path_snub + ".key")

            key = ""
            pattern = r'^.*?\b(major|minor)\b'
            with open(annotation_path, "r") as key_file:
                key += key_file.readline()
                match = re.search(pattern, key, flags=re.IGNORECASE)
                if match:
                    key = match.group(0)

                if key[-5:] != "major" and key[-5:] != "minor":
                    continue

                if not key[0].isalpha():
                    continue

                if '/' in key:
                    continue

            key = key.strip().lower()

            mel_spec_path = os.path.join(mel_spec_dir, path_snub + ".npy")

            if not os.path.exists(mel_spec_path):
                mel_spec = get_melspec_from_file(mp3_path)
                if mel_spec is None:
                    continue

                np.save(mel_spec_path, mel_spec)

            self.id_to_key[id] = key
            self.ids.append(id)

            if not self.key_to_num.keys().__contains__(key):
                self.key_to_num[key] = key_num
                key_num += 1

            self.id_to_spectrograms[id] = mel_spec_path

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        id = self.ids[idx]
        path = self.id_to_spectrograms[id]
        mel_spec = np.load(path)
        key = self.id_to_key[id]
        return self.key_to_num[key], torch.from_numpy(mel_spec)


class EmoMusic(Dataset):
    def __init__(self, data_directory, split="train"):
        self.id_to_arousal = {}
        self.ids = []
        self.id_to_valence = {}
        self.id_to_spectrograms = {}

        mp3_dir = os.path.join(data_directory, "clips_45seconds\\")
        subset_path = os.path.join(data_directory, split + ".tsv")
        mel_spec_dir = os.path.join(data_directory, "mel_spec_all\\")

        with open(subset_path, "r", encoding="utf-8") as f:
            header = 0
            for line in f:
                if header == 0:
                    header = 1
                    continue

                parts = line.split("\t")

                arousal = re.search(r'\'mean_arousal\':\s*\'(.+?)\'', parts[3]).group(1)
                valence = re.search(r'\'mean_valence\':\s*\'(.+?)\'', parts[3]).group(1)

                id = int(parts[0])

                self.id_to_arousal[id] = float(arousal)
                self.id_to_valence[id] = float(valence)

        for row in tqdm(os.listdir(mp3_dir)):
            mp3_path = os.path.join(mp3_dir, row)
            path_sections = row.split(".")
            id = int(path_sections[0])

            if id not in self.id_to_arousal.keys():
                continue

            path_snub = path_sections[0]
            mel_spec_path = os.path.join(mel_spec_dir, path_snub + ".npy")

            if not os.path.exists(mel_spec_path):
                mel_spec = get_melspec_from_file(mp3_path)
                if mel_spec is None:
                    continue

                np.save(mel_spec_path, mel_spec)

            self.ids.append(id)
            self.id_to_spectrograms[id] = mel_spec_path

        print(len(self.ids))


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        id = self.ids[idx]
        path = self.id_to_spectrograms[id]
        mel_spec = np.load(path)
        arousal = self.id_to_arousal[id]
        valence = self.id_to_valence[id]

        coord = torch.tensor([arousal, valence], dtype=torch.float)

        return coord, mel_spec

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