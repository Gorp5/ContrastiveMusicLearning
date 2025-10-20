import random

import librosa
import numpy as np
import torch
import os
import csv

from librosa.feature import melspectrogram
from torch.utils.data import Dataset
from collections import defaultdict
from mtgjamendodataset.scripts import commons

def get_melspec_from_wav(full_path):
    audio, sr = librosa.load(full_path, sr=44100, mono=True)

    win_length = int(round(0.025 * sr))  # ~1103 samples
    hop_length = int(round(0.010 * sr))  # 441 samples
    n_fft = 2048

    data = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        n_mels=128, fmin=0, fmax=sr / 2, power=2.0
    )

    return librosa.amplitude_to_db(data, ref=np.max)

class GTZAN(Dataset):
    def __init__(self, data_directory, transform=None):
        self.genre_folders = sorted(os.listdir(data_directory))
        self.transform = transform

        self.data = []
        self.tags = []

        self.spectrograms = []
        self.tags = []

        for index in range(1000):
            tag = index // 100
            index_in_genre = index % 100

            self.tags.append(tag)
            folder_path = os.path.join(data_directory, self.genre_folders[tag])

            songs= sorted(os.listdir(folder_path))
            song_path = os.path.join(folder_path, songs[index_in_genre])

            mel_spec = get_melspec_from_wav(song_path)

            self.spectrograms.append(mel_spec)

    def __len__(self):
        return 1000

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

                mel_spec = get_melspec_from_wav(self.id_to_path[id])

                self.spectrograms[id] = mel_spec

        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        return self.id_to_tags[id], self.spectrograms[id]


class StreamViewDataset(Dataset):
    def __init__(self, song_dir: str, label_dir: str, transform=None, pair_album=False, views=2):
        self.song_folders = sorted(os.listdir(song_dir))
        self.song_labels = sorted(os.listdir(label_dir))

        subset_file_name = "autotagging_top50tags"
        subset_file = f'E:/mtg-jamendo-dataset/data/{subset_file_name}.tsv'

        self.find_song = pair_album
        self.views = views

        if self.find_song:
            tracks, tags, extra = commons.read_file(subset_file)
            #tag_mapping = ReadStats(subset_file_name)

            self.tracks = tracks

            album_to_ids = defaultdict(list)
            for album, id in [(self.tracks[i]['album_id'], i) for i in
                              [int(self.song_folders[i]) for i in range(len(self.song_folders))]]:

                if str(id) in self.song_folders:
                    album_to_ids[album].append(id)

            artist_to_ids = defaultdict(list)
            for artist, id in [(self.tracks[i]['artist_id'], i) for i in
                              [int(self.song_folders[i]) for i in range(len(self.song_folders))]]:

                if str(id) in self.song_folders:
                    artist_to_ids[artist].append(id)

            self.album_to_ids = album_to_ids
        self.song_dir = song_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.song_folders)


    def __getitem__(self, idx):
        song_id = self.song_folders[idx]
        folder = os.path.join(self.song_dir, song_id)
        files = os.listdir(folder)

        rand_index_1 = random.randint(0, len(files) - 1)
        
        views = []
        view_1 = torch.load(os.path.join(self.song_dir, os.path.join(folder, files[rand_index_1])), map_location='cpu')
        views.append(view_1)
        
        if self.views > 1:
            excluded = []
            excluded.append(rand_index_1)
            #num_files = len(files)

            choices = files[:rand_index_1] + files[rand_index_1 + 1:]
            rand_index_indicies = random.sample(choices, k=self.views - 1)
            
            for index in rand_index_indicies:
                view_i = torch.load(os.path.join(self.song_dir, os.path.join(folder, index)), map_location='cpu')
                views.append(view_i)

        if self.transform:
            for index, view in enumerate(views):
                views[index] = self.transform(view.clone())

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