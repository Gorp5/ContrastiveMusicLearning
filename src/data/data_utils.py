import random
import torch
import os

from torch.utils.data import Dataset
from collections import defaultdict
from data.processing import ReadStats
from mtgjamendodataset.scripts import commons



# The amount of data is just small enough to fit in memory :)
class AudioDataset(Dataset):
    def __init__(self, data_directory, tags_directory, transform=None):
        self.song_files = sorted(os.listdir(data_directory))
        self.label_files = sorted(os.listdir(tags_directory))
        self.transform = transform

        self.data = []
        self.tags = []

        self.random = random.seed(42)

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
            num_files = len(files)
            
            choices = files[:rand_index_1] + files[rand_index_1 + 1:]
            rand_index_indicies = random.sample(choices, k=self.views - 1)
            
            for index in rand_index_indicies:
                view_i = torch.load(os.path.join(self.song_dir, os.path.join(folder, index)), map_location='cpu')
                views.append(view_i)
        
        return idx, views

        #if self.find_song:
        #    # Pick random song from songs in album
        #    album_id = self.tracks[int(song_id)]['album_id']
        #    ids_in_album = self.album_to_ids[album_id]
        #    positive_song_id = random.choice(ids_in_album)
        #    other_folder = os.path.join(self.song_dir, str(positive_song_id))
        #    other_files = os.listdir(other_folder)
        #    rand_index_3 = random.randint(0, len(other_files) - 1)
        #    view_3 = torch.load(os.path.join(self.song_dir, os.path.join(other_folder, other_files[rand_index_3])), map_location='cpu')

        #    if self.transform:
        #        view_3 = self.transform(view_3.clone())

        #    return idx, (view_1, view_3)


        if self.transform:
            view_1 = self.transform(view_1.clone())
            view_2 = self.transform(view_2.clone())

        return album_id, (view_1, view_2)

        # song_id = self.song_folders[idx]
        # folder = os.path.join(self.song_dir, song_id)
        # files = os.listdir(folder)
        # rand_index_1 = random.randint(0, len(files) - 1)
        #
        # if self.find_song:
        #     # Pick random song from songs in album
        #     album_id = self.tracks[int(song_id)]['album_id']
        #     ids_in_album = self.album_to_ids[album_id]
        #     positive_song_id = random.choice(ids_in_album)
        #     other_folder = os.path.join(self.song_dir, str(positive_song_id))
        #     other_files = os.listdir(other_folder)
        #     rand_index_2 = random.randint(0, len(other_files) - 1)
        # else:
        #     other_folder = folder
        #     other_files = files
        #     rand_index_2 = random.randint(0, len(files[:rand_index_1] + files[rand_index_1 + 1:]))
        #
        # view_1 = torch.load(os.path.join(self.song_dir, os.path.join(folder, files[rand_index_1])), map_location='cpu')
        # view_2 = torch.load(os.path.join(self.song_dir, os.path.join(other_folder, other_files[rand_index_2])), map_location='cpu')
        #
        # if self.transform:
        #     view_1 = self.transform(view_1.clone())
        #     view_2 = self.transform(view_2.clone())
        #
        # return idx, (view_1, view_2)


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