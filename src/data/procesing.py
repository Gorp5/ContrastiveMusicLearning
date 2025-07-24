import os
import random

import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from data.Data import retrieve_data
from torch.utils.data import Dataset, IterableDataset
from mtgjamendodataset.scripts import commons
from utils.data import createDictionary, makeClassLabels
from torch.utils.data._utils.worker import get_worker_info
from librosa.feature import melspectrogram


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


# TODO: Implement K-Fold Cross Validation
def ParseData(subset_file_name, data_location, output_directory, features=96, chunks_per_batch=128, chunk_size=256, songs_per_label=None, labels_to_include=50, chunks_per_song=None, test_prob=0.1, convert=False):
    subset_file = f'E:/mtg-jamendo-dataset/data/{subset_file_name}.tsv'

    tracks, tags, extra = commons.read_file(subset_file)
    genre_count, genre_mapping = ReadStats(subset_file_name)

    subset_genre_mapping = {}
    if labels_to_include == 50:
        subset_genre_mapping = genre_mapping
    else:
        for genre, index in genre_mapping.items():
            if index in range(labels_to_include):
                subset_genre_mapping[genre] = genre_mapping[genre]


    num_genres = len(subset_genre_mapping.keys())
    num_moods = len(tags['mood/theme'].keys())
    num_instruments = len(tags['instrument'].keys())

    print(f"There are {num_genres} genres in this partition.")
    print(f"There are {num_moods} moods/themes in this partition.")
    print(f"There are {num_instruments} instruments in this partition.")

    count = 0

    song_set = []
    label_set = []

    validate_song_set = []
    validate_label_set = []

    # test_song_set = []
    # test_label_set = []

    genre_running_total = [0] * num_genres

    for track_num in tqdm(tracks.keys()):
        metadata_dict = tracks[track_num]
        path_end = metadata_dict['path']
        genre = metadata_dict['genre']

        full_path = data_location + path_end[:-3] + "npy"

        # Make a list of genres that song is tagged as
        labeled_genres = set()

        if labels_to_include != 50:
            if len([1 for x in genre if x not in subset_genre_mapping]) > 0:
                continue

        if songs_per_label:
            if len([1 for x in genre if genre_running_total[subset_genre_mapping[x]] > songs_per_label]) > 0:
                continue

        continue_flag = False
        for g in genre:
            genre_index = subset_genre_mapping[g]
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

        if not convert:
            data = np.load(full_path)
        else:
            audio, sr = librosa.load(full_path, sr=44100, mono=True)
            data = librosa.feature.melspectrogram(y=audio, sr=sr)
            #s_chroma = librosa.feature.chroma_stft(y=s, sr=sr)

        chunked_data, num_chunks = chunk_data(data, features=features, chunk_size=chunk_size)
        repeated_labels = [torch.tensor(labels)] * num_chunks

        if chunks_per_song:
            chunked_data = chunked_data[:chunks_per_song]
            repeated_labels = repeated_labels[:chunks_per_song]

        count += 1

        random.shuffle(chunked_data)
        if random.random() > test_prob:
            song_set.extend(chunked_data)
            label_set.extend(repeated_labels)
        else:
            validate_song_set.extend(chunked_data)
            validate_label_set.extend(repeated_labels)

        if len(song_set) >= chunks_per_batch:
            # Randomly sample between both sets for the chunks from the song we want to use
            combined = list(zip(song_set, label_set))
            random.shuffle(combined)
            song_set, label_set = zip(*combined)

            remainder_data = chunked_data[chunks_per_batch:]
            remainder_labels = repeated_labels[chunks_per_batch:]

            save_file(torch.stack(song_set[:chunks_per_batch]), f"{output_directory}/train_set/data/{count:04d}.pt")
            save_file(torch.stack(label_set[:chunks_per_batch]), f"{output_directory}/train_set/genre_labels/{count:04d}.pt")

            song_set = remainder_data
            label_set = remainder_labels

        if len(validate_song_set) >= chunks_per_batch:
            combined = list(zip(validate_song_set, validate_label_set))
            random.shuffle(combined)
            validate_song_set, validate_label_set = zip(*combined)

            remainder_data = chunked_data[chunks_per_batch:]
            remainder_labels = repeated_labels[chunks_per_batch:]

            save_file(torch.stack(validate_song_set[:chunks_per_batch]), f"{output_directory}/test_set/data/{count:04d}.pt")
            save_file(torch.stack(validate_label_set[:chunks_per_batch]), f"{output_directory}/test_set/genre_labels/{count:04d}.pt")

            validate_song_set = remainder_data
            validate_label_set = remainder_labels

    save_file(torch.stack(song_set), f"{output_directory}/train_set/data/{count:04d}.pt")
    save_file(torch.stack(label_set), f"{output_directory}/train_set/genre_labels/{count:04d}.pt")
    save_file(torch.stack(validate_song_set[:chunks_per_batch]), f"{output_directory}/test_set/data/{count:04d}.pt")
    save_file(torch.stack(validate_label_set[:chunks_per_batch]), f"{output_directory}/test_set/genre_labels/{count:04d}.pt")


def save_file(object, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(object, file_path)


def show_mel_before(mel):
    plt.imshow(mel.squeeze(), origin='lower', aspect='auto', cmap='magma')
    plt.title("Before")
    #plt.title(f"Genres: {torch.nonzero(label).squeeze().tolist()}")
    plt.colorbar()
    plt.show()


def show_mel_after(mel):
    num_plots = len(mel)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))

    if num_plots == 1:
        axes = [axes]

    for i, mel in enumerate(mel):
        ax = axes[i]
        mel = mel.squeeze()  # remove channel if present
        im = ax.imshow(mel, origin='lower', aspect='auto', cmap='magma')
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


def chunk_data(data, features=96, chunk_size=256):
    data = torch.tensor(data)

    F, T = data.shape
    T_trunc = T - (T % chunk_size)

    data = data[:, :T_trunc]
    N = T_trunc // chunk_size
    data = data.reshape(F, N, chunk_size)
    data = data.permute(1, 0, 2)

    return list(data), N
