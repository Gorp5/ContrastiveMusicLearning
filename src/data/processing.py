import os
import random

import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from mtgjamendodataset.scripts import commons
from librosa.feature import melspectrogram

from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary, LpStatus, value, PULP_CBC_CMD, LpMinimize


def ReadStats(subset_file_name):
    subset_file = f'E:/mtg-jamendo-dataset/stats/{subset_file_name}/all.tsv'
    tag_index_dict = {}

    first_flag = True
    with open(subset_file) as f:
        for index, line in enumerate(f.readlines()):
            if first_flag:  # Skip headers
                first_flag = False
                continue

            chunks = line.split('\t')
            tags, artists, albums, tracks = chunks
            tag_index_dict[tags] = index - 1

    return tag_index_dict


def lp_solver_2(all_tracks, all_tags, songs_per_tag=1024):
    TARGET = songs_per_tag
    song_vars = {sid: LpVariable(f"select_{sid}", cat=LpBinary) for sid in all_tracks}
    prob = LpProblem("UniformTagCoverage", LpMinimize)

    # Create a tag variable for how many songs are selected per tag
    tag_vars = {tag: LpVariable(f"tag_count_{tag}", lowBound=0, upBound=TARGET, cat='Integer') for tag in all_tags}

    # Link tag_vars to selected songs
    for tag, song_ids in all_tags.items():
        prob += tag_vars[tag] == lpSum(song_vars[sid] for sid in song_ids)

    # Compute average tag count
    num_tags = len(all_tags)
    avg_tag_count = lpSum(tag_vars.values()) / num_tags

    # Add deviation variables for each tag
    deviations = {
        tag: LpVariable(f"dev_{tag}", lowBound=0, cat='Continuous') for tag in all_tags
    }

    for tag in all_tags:
        prob += tag_vars[tag] - avg_tag_count <= deviations[tag]
        prob += avg_tag_count - tag_vars[tag] <= deviations[tag]

    # Objective: minimize total deviation (i.e., uniform tag distribution)
    # and slightly reward more songs to break ties
    total_deviation = lpSum(deviations.values())
    total_songs = lpSum(song_vars.values())
    prob += total_deviation - 0.001 * total_songs  # Prefer uniformity, but gently encourage more songs

    # Solve
    solver = PULP_CBC_CMD(
        msg=True,
        timeLimit=300,
        gapRel=0.001,
        options=[
            "sec=60",  # Alternative to timeLimit
            "mipgap=0.005",  # CBC's own relative gap setting
            "maxNodes=1000000",
            "maxSolutions=10000",
            "randomSeed=42"
        ]
    )

    prob.solve(solver)

    # Collect selected songs
    selected_songs = [sid for sid, var in song_vars.items() if var.varValue == 1]
    tag_breakdown = {tag: int(tag_vars[tag].varValue) for tag in all_tags}

    return selected_songs, tag_breakdown


def lp_solver(all_tracks, all_tags, songs_per_tag=250):
    TARGET = songs_per_tag # Max samples per tag

    # Create binary variables for each song
    song_vars = {sid: LpVariable(f"select_{sid}", cat=LpBinary) for sid in all_tracks}

    # ILP setup
    prob = LpProblem("BalanceGenres", LpMaximize)

    # Build the tag coverage score per song
    song_tag_count = {sid: len(all_tracks[sid]) for sid in all_tracks}

    # Objective: maximize tag coverage
    prob += lpSum(song_tag_count[sid] * song_vars[sid] for sid in all_tracks)

    # Constraint: each genre appears at most TARGET times
    for tag, song_ids in all_tags.items():
        prob += lpSum(song_vars[sid] for sid in song_ids) <= TARGET

    # Solve and show results
    solver = PULP_CBC_CMD(
        msg=True,
        timeLimit=60,
        gapRel=0.0001,
        options=[
            "sec=60",  # Alternative to timeLimit
            "mipgap=0.0001",  # CBC's own relative gap setting
            "maxNodes=1000000",
            "maxSolutions=10000",
            "randomSeed=42"
        ]
    )

    prob.solve(solver)
    selected_songs = [sid for sid, var in song_vars.items() if var.varValue == 1]

    tag_breakdown = {tag: 0 for tag in all_tags.keys()}
    for song_id in selected_songs:
        tags = all_tracks[song_id]
        for tag in tags:
            tag_breakdown[tag] += 1

    return selected_songs, tag_breakdown


def ParseBalanced(subset_file_name, data_location, output_directory, convert, target_per_genre=256):
    subset_file = f'E:/mtg-jamendo-dataset/data/{subset_file_name}.tsv'

    tracks, tags, extra = commons.read_file(subset_file)
    tag_mapping = ReadStats(subset_file_name)

    all_tags = {}
    all_tags.update(tags['genre'])
    all_tags.update(tags['instrument'])
    all_tags.update(tags['mood/theme'])

    #min_tag_count = min([len(t) for t in all_tags.values()])
    chunks_per_batch = 4096
    chunks_per_song = None
    test_prob = 0.1
    chunk_size = 256
    num_genres = 50

    count = 0

    song_set = []
    label_set = []
    id_set = []

    validate_song_set = []
    validate_label_set = []
    validation_id_set = []

    missed_songs = []
    all_tracks = {}

    for track, data in tracks.items():
        all_tracks[track] = [f.split("---")[1] for f in data['tags']]

    selected_track, tag_breakdown = lp_solver_2(all_tracks, all_tags, songs_per_tag=target_per_genre)
    random.shuffle(selected_track)

    vals = list(tag_breakdown.values())
    min_value = min(vals)
    max_value = max(vals)
    std = np.std(vals)
    mean = sum(vals) / len(vals)

    plt.plot(vals)
    plt.show()

    print(f"Min Samples per Genre: {min_value}\nMax Samples per Genre: {max_value}\n Standard Deviation: {std}\n Mean: {mean}\nTracks in Total: {len(selected_track)}")

    for track_id in tqdm(selected_track):
        tags = all_tracks[track_id]
        path_end = tracks[track_id]['path']

        path_stem = "npy" if not convert else "mp3"
        full_path = data_location + path_end[:-3] + path_stem

        labels = [0] * num_genres
        for t in tags:
            genre_index = tag_mapping[t]
            labels[genre_index] = 1

        if not convert:
            data = np.load(full_path)
        else:
            if os.path.exists(full_path):
                audio, sr = librosa.load(full_path, sr=44100, mono=True)
                data = librosa.feature.melspectrogram(y=audio, sr=sr)
                data = librosa.amplitude_to_db(data, ref=np.max)
                # s_chroma = librosa.feature.chroma_stft(y=s, sr=sr)
            else:
                missed_songs.append(full_path)
                continue

        chunked_data, num_chunks = chunk_data(data, chunk_size=chunk_size)
        repeated_labels = [torch.tensor(labels)] * num_chunks

        if chunks_per_song:
            num_samples = min(chunks_per_song, len(chunked_data) - 1)
            repeated_labels = repeated_labels[:num_samples]
            chunked_data = random.sample(chunked_data, num_samples)
        count += 1

        if random.random() > test_prob:
            song_set.extend(chunked_data)
            label_set.extend(repeated_labels)
            id_set.append(track_id)
        else:
            validate_song_set.extend(chunked_data)
            validate_label_set.extend(repeated_labels)
            validation_id_set.append(track_id)

        if len(song_set) >= chunks_per_batch * 2:
            # Randomly sample between both sets for the chunks from the song we want to use
            combined = list(zip(song_set, label_set))
            random.shuffle(combined)
            song_set, label_set = zip(*combined)

            remainder_data = chunked_data[chunks_per_batch:]
            remainder_labels = repeated_labels[chunks_per_batch:]

            save_file(torch.stack(song_set[:chunks_per_batch]), f"{output_directory}/train_set/data/{count:04d}.pt")
            save_file(torch.stack(label_set[:chunks_per_batch]),
                      f"{output_directory}/train_set/genre_labels/{count:04d}.pt")

            song_set = remainder_data
            label_set = remainder_labels

        if len(validate_song_set) >= chunks_per_batch * 2:
            combined = list(zip(validate_song_set, validate_label_set))
            random.shuffle(combined)
            validate_song_set, validate_label_set = zip(*combined)

            remainder_data = chunked_data[chunks_per_batch:]
            remainder_labels = repeated_labels[chunks_per_batch:]

            save_file(torch.stack(validate_song_set[:chunks_per_batch]),
                      f"{output_directory}/test_set/data/{count:04d}.pt")
            save_file(torch.stack(validate_label_set[:chunks_per_batch]),
                      f"{output_directory}/test_set/genre_labels/{count:04d}.pt")

            validate_song_set = remainder_data
            validate_label_set = remainder_labels

    save_file(torch.stack(song_set), f"{output_directory}/train_set/data/{count:04d}.pt")
    save_file(torch.stack(label_set), f"{output_directory}/train_set/genre_labels/{count:04d}.pt")
    save_file(torch.stack(validate_song_set[:chunks_per_batch]), f"{output_directory}/test_set/data/{count:04d}.pt")
    save_file(torch.stack(validate_label_set[:chunks_per_batch]),
              f"{output_directory}/test_set/genre_labels/{count:04d}.pt")

    save_file(missed_songs, f"{output_directory}missed_songs.pt")

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
    missed_songs = []

    # test_song_set = []
    # test_label_set = []

    genre_running_total = [0] * num_genres

    for track_num in tqdm(tracks.keys()):
        metadata_dict = tracks[track_num]
        path_end = metadata_dict['path']
        genre = metadata_dict['genre']

        path_stem = "npy" if not convert else "mp3"
        full_path = data_location + path_end[:-3] + path_stem

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
            if os.path.exists(full_path):
                audio, sr = librosa.load(full_path, sr=44100, mono=True)
                data = librosa.feature.melspectrogram(y=audio, sr=sr)
                data = librosa.amplitude_to_db(data, ref=np.max)
                # s_chroma = librosa.feature.chroma_stft(y=s, sr=sr)
            else:
                missed_songs.append(full_path)
                continue

        chunked_data, num_chunks = chunk_data(data, chunk_size=chunk_size)
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

        if len(song_set) >= chunks_per_batch * 2:
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

    save_file(missed_songs, f"{output_directory}missed_songs.pt")
    print(f"Couldn't find {len(missed_songs)} songs.")


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


def chunk_data(data, chunk_size=256):
    data = torch.tensor(data)

    F, T = data.shape
    T_trunc = T - (T % chunk_size)

    data = data[:, :T_trunc]
    N = T_trunc // chunk_size
    data = data.reshape(F, N, chunk_size)
    data = data.permute(1, 0, 2)

    return list(data), N

def show_mel(mel):
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