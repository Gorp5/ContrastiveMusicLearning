import os

import numpy as np
import torch
from torch.utils.data._utils.worker import get_worker_info

from tqdm import tqdm
from data.Data import retrieve_data
from torch.utils.data import Dataset, IterableDataset
from mtgjamendodataset.scripts import commons
from utils.data import createDictionary, makeClassLabels


def ReadStats(subset_file_name):
    subset_file = f'E:/mtg-jamendo-dataset/stats/{subset_file_name}/genre.tsv'
    genre_count = {}
    genre_index_dict = {}

    count = 0
    first_flag = True
    with open(subset_file) as f:
        for line in f.readlines():
            if first_flag:  # Skip headers
                first_flag = False
                continue

            chunks = line.split('\t')
            genre, artists, albums, songs = chunks
            genre_count[genre] = int(songs[:-1])
            genre_index_dict[genre] = count
            count += 1

    return genre_count, genre_index_dict

def ParseData(subset_file_name, data_location, output_directory, features=96, chunks_per_batch=128, chunk_size=256, per_label=None, labels_to_include=50, chunks_per_song=None):
    subset_file = f'E:/mtg-jamendo-dataset/data/{subset_file_name}.tsv'

    tracks, tags, extra = commons.read_file(subset_file)
    genre_count, genre_mapping = ReadStats(subset_file_name)

    num_genres = len(tags['genre'].keys())
    num_moods = len(tags['mood/theme'].keys())
    num_instruments = len(tags['instrument'].keys())

    print(f"There are {num_genres} genres in this partition.")
    print(f"There are {num_moods} moods/themes in this partition.")
    print(f"There are {num_instruments} instruments in this partition.")

    count = 0

    song_set = []
    label_set = []

    genre_running_total = [0] * num_genres

    for track_num in tqdm(tracks.keys()):
        metadata_dict = tracks[track_num]
        #artist_id = metadata_dict['artist_id']
        #album_id = metadata_dict['album_id']
        path_end = metadata_dict['path']
        #duration = metadata_dict['duration']
        genre = metadata_dict['genre']
        #instrument = metadata_dict['instrument']
        #mood = metadata_dict['mood/theme']

        full_path = data_location + path_end[:-3] + "npy"

        # Make a list of genres that song is tagged as
        labeled_genres = set()

        if per_label:
            if len([1 for x in genre if genre_running_total[genre_mapping[x]] > per_label]) > 0:
                continue

        continue_flag = False
        for g in genre:
            genre_index = genre_mapping[g]
            if genre_index > labels_to_include:
                continue_flag = True
                continue

            genre_running_total[genre_index] = genre_running_total[genre_index] + 1
            labeled_genres.add(genre_index)

        if continue_flag:
            continue

        labels = [0] * num_genres
        for genre_index in labeled_genres:
            labels[genre_index] = 1

        count += 1

        data = np.load(full_path)

        chunked_data, num_chunks = chunk_data(data, features=features, chunk_size=chunk_size)
        repeated_labels = [torch.tensor(labels)] * num_chunks

        if chunks_per_song:
            chunked_data = chunked_data[:chunks_per_song]
            repeated_labels = repeated_labels[:chunks_per_song]


        song_set.extend(chunked_data)
        label_set.extend(repeated_labels)

        if len(song_set) >= chunks_per_batch:
            remainder_data = chunked_data[chunks_per_batch:]
            remainder_labels = repeated_labels[chunks_per_batch:]

            random_indicies = torch.randperm(chunks_per_batch)

            torch.save(torch.stack(song_set[:chunks_per_batch])[random_indicies], f"{output_directory}/data/{count:04d}.pt")
            torch.save(torch.stack(label_set[:chunks_per_batch])[random_indicies], f"{output_directory}/genre_labels/{count:04d}.pt")

            song_set = remainder_data
            label_set = remainder_labels

    torch.save(torch.stack(song_set[:chunks_per_batch]), f"{output_directory}/data/{count:04d}.pt")
    torch.save(torch.stack(label_set[:chunks_per_batch]),f"{output_directory}/genre_labels/{count:04d}.pt")




def chunk_data(data, features=96, chunk_size=256):
    # Fill with padding
    data = torch.tensor(data)
    transposed = data.transpose(0, 1)
    num_chunks = transposed.shape[0] // chunk_size
    trimmed = transposed[: num_chunks * chunk_size]

    trimmed = trimmed.reshape(num_chunks, chunk_size, features)

    return list(trimmed), num_chunks


def ParseTaggedDataset(source_directory, new_directory, features=64, chunk_size=50, length=256):
    count = 0

    all_labels = []

    for start in tqdm(range(0, 10000, chunk_size)):
        mtg_dataset, masks, keys = retrieve_data(source_directory, features=features, start=start, count=chunk_size,
                                           sample_length=length, keep_song_data_option=True)

        dictionary = createDictionary()
        all_genre_labels, all_mood_labels, genre_indicies, mood_indicies = makeClassLabels(keys, dictionary)

        genre_indicies = set(genre_indicies)

        filtered_songs = [mtg_dataset[i] for i in genre_indicies]
        filtered_songs = torch.cat(filtered_songs)

        all_labels.append(all_genre_labels)
        all_genre_labels = torch.stack(all_genre_labels)

        torch.save(filtered_songs, f"E:\\SongsDataset\\{new_directory}\\data\\dataset{count:04d}.pt")
        torch.save(keys, f"E:\\SongsDataset\\{new_directory}\\keys\\dataset{count:04d}-keys.pt")
        torch.save(all_genre_labels, f"E:\\SongsDataset\\{new_directory}\\genre_labels\\dataset{count:04d}-labels.pt")
        torch.save(all_mood_labels, f"E:\\SongsDataset\\{new_directory}\\mood_labels\\dataset{count:04d}-labels.pt")
        torch.save(masks, f"E:\\SongsDataset\\{new_directory}\\masks\\dataset{count :04d}-masks.pt")
        count += 1

    return all_labels


def filter_data(genre_indicies, mtg_dataset):
    filtered_data = []
    for index in genre_indicies:
        filtered_data.append(mtg_dataset[index])
    return filtered_data


class StreamingSongDataset(Dataset):
    def __init__(self, song_dir: str, label_dir: str, indices=None):
        self.song_files = sorted(os.listdir(song_dir))
        self.label_files = sorted(os.listdir(label_dir))
        if indices is not None:
            self.song_files = [self.song_files[i] for i in indices]
            self.label_files = [self.label_files[i] for i in indices]

        self.song_dir = song_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.song_files)

    def __getitem__(self, idx):
        song = torch.load(os.path.join(self.song_dir, self.song_files[idx]), map_location='cpu')
        label = torch.load(os.path.join(self.label_dir, self.label_files[idx]), map_location='cpu')

        return song, label


class IterableStreamingDataset(IterableDataset):
    def __init__(self, song_dir: str, label_dir: str,):
        self.song_files = sorted(os.listdir(song_dir))
        self.label_files = sorted(os.listdir(label_dir))

        self.song_dir = song_dir
        self.label_dir = label_dir
    def __len__(self):
        return len(self.song_files)

    def __getitem__(self, idx):
        song = torch.load(os.path.join(self.song_dir, self.song_files[idx]), map_location='cpu')
        label = torch.load(os.path.join(self.label_dir, self.label_files[idx]), map_location='cpu')

        return song, label

    def __iter__(self):
        worker_info = get_worker_info()
        per_worker = len(self.song_files) // worker_info.num_workers
        start = worker_info.id * per_worker
        end = start + per_worker

        for song_file, label_file in zip(self.song_files[start:end], self.label_files[start:end]):
            song = torch.load(song_file, map_location="cpu")
            label = torch.load(label_file, map_location="cpu")
            yield song, label