import os
import random
from asyncio import as_completed
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from mtgjamendodataset.scripts import commons
from librosa.feature import melspectrogram

from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary, LpStatus, value, PULP_CBC_CMD, LpMinimize


def ReadStats(subset_file):
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

    solver = PULP_CBC_CMD(
        msg=True,
        timeLimit=300,
        gapRel=0.001,
        options=[
            "sec=60",
            "mipgap=0.005",
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

def process_track(track_id, all_tracks, tracks, data_location,
                  num_genres, tag_mapping, test_prob, output_directory):

    missed_songs = []

    tags = all_tracks[track_id]
    path_end = tracks[track_id]['path']

    path_stem = "mp3"
    full_path = data_location + path_end[:-3] + path_stem

    # one-hot encode genre tags
    labels = [0] * num_genres
    for t in tags:
        genre_index = tag_mapping[t]
        labels[genre_index] = 1

    discography_labels = (tracks[track_id]['artist_id'], tracks[track_id]['album_id'])

    if os.path.exists(f"{output_directory}/train_set/data/{track_id:04d}.pt"):
        return None

    if os.path.exists(f"{output_directory}/test_set/data/{track_id:04d}.pt"):
        return None

    if os.path.exists(full_path):
        audio, sr = librosa.load(full_path, sr=44100, mono=True)

        win_length = int(round(0.025 * sr))  # ~1103 samples
        hop_length = int(round(0.010 * sr))  # 441 samples
        n_fft = 2048

        data = librosa.feature.melspectrogram(
                    y=audio, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                    n_mels=128, fmin=0, fmax=sr/2, power=2.0
                )
        data = librosa.amplitude_to_db(data, ref=np.max)
    else:
        missed_songs.append(full_path)
        return None

    # determine train/test split
    dataset_split = "train_set" if random.random() > test_prob else "test_set"

    outputs = []
    outputs.append((
        f"{output_directory}/{dataset_split}/data/{track_id:04d}.pt",
        data

    ))
    outputs.append((
        f"{output_directory}/{dataset_split}/genre_labels/{track_id:04d}.pt",
        labels
    ))
    outputs.append((
        f"{output_directory}/{dataset_split}/discography_labels/{track_id:04d}.pt",
        discography_labels
    ))

    return outputs

def parse_parallel(selected_tracks, output_dir, tag_mapping, data_location, all_tracks, tracks, test_prob=0.1):
    # This will collect all (path, data) from all track tasks
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit every task
        future_to_track = {
            executor.submit(
                process_track,
                track_id,
                all_tracks, tracks, data_location,
                50, tag_mapping, test_prob, output_dir
            ): track_id
            for track_id in selected_tracks
        }

        with tqdm(total=len(future_to_track)) as pbar:
            for future in as_completed(future_to_track):
                track_id = future_to_track[future]
                try:
                    res = future.result()
                    for (path_out, data_obj) in res:
                        try:
                            save_file(data_obj, path_out)
                        except Exception as e:
                            print(f"Error is saving, track {track_id} generated exception: {e}")
                except Exception as e:
                    print(f"[Error] track {track_id} generated exception: {e}")
                finally:
                    pbar.update(1)

def ParseBalanced(subset_file, read, data_location, output_directory, target_per_genre=1024):
    tracks, tags, extra = commons.read_file(subset_file)
    tag_mapping = ReadStats(read)

    all_tags = {}
    all_tags.update(tags['genre'])
    all_tags.update(tags['instrument'])
    all_tags.update(tags['mood/theme'])

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

    with open("tag_balanced_subset.csv", "w") as f:
        f.write("id|tags\n")
        for track_id in tqdm(selected_track):
            tags = ",".join([str(x) for x in all_tracks[track_id]])
            f.write(f"{track_id}|{tags}\n")

    print(
        f"Min Samples per Genre: {min_value}\nMax Samples per Genre: {max_value}\n Standard Deviation: {std}\n Mean: {mean}\nTracks in Total: {len(selected_track)}")

    parse_sync(selected_track, output_directory, tag_mapping, data_location, all_tracks, tracks)


def parse_sync(selected_tracks, output_dir, tag_mapping, data_location, all_tracks, tracks, test_prob=0.1):
    # Submit every task
    for track_id in tqdm(selected_tracks):
        output = process_track(
            track_id,
            all_tracks, tracks, data_location,
            50, tag_mapping, test_prob, output_dir
        )

        try:
            res = output
            if res is None:
                continue

            for (path_out, data_obj) in res:
                try:
                    save_file(data_obj, path_out)
                except Exception as e:
                    print(f"Error is saving, track {track_id} generated exception: {e}")
        except Exception as e:
            print(f"[Error] track {track_id} generated exception: {e}")

def ParseAll(subset_file, read, data_location, output_directory):
    tracks, tags, extra = commons.read_file(subset_file)
    tag_mapping = ReadStats(read)

    all_tags = {}
    all_tags.update(tags['genre'])
    all_tags.update(tags['instrument'])
    all_tags.update(tags['mood/theme'])

    all_tracks = {}

    for track, data in tracks.items():
        all_tracks[track] = [f.split("---")[1] for f in data['tags']]

    print("Processing tracks...")
    parse_sync([x for x in all_tracks.keys()], output_directory, tag_mapping, data_location, all_tracks, tracks)


def save_file(object, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(object, file_path)


def chunk_data(data, chunk_size=256):
    #data = torch.tensor(data)

    F, T = data.shape
    T_trunc = T - (T % chunk_size)

    data = data[:, :T_trunc]
    N = T_trunc // chunk_size
    data = data.reshape(F, N, chunk_size)
    data = data.permute(1, 0, 2)

    return data, N

def chunk_data_pad(data, chunk_size=256):
    F_dim, T = data.shape
    remainder = T % chunk_size
    pad_len = 0

    if remainder != 0:
        pad_len = chunk_size - remainder
        data = F.pad(data, (0, pad_len))  # pad along time dimension
        T += pad_len

    N = T // chunk_size

    # Build attention mask before reshaping
    mask = torch.ones(T, dtype=torch.bool, device=data.device)
    if pad_len > 0:
        mask[-pad_len:] = False  # padded timesteps marked as False (0)

    # Reshape both data and mask
    data = data.reshape(F_dim, N, chunk_size).permute(1, 0, 2)  # [N, F, chunk_size]
    attn_mask = mask.reshape(N, chunk_size)  # [N, chunk_size]

    return data, attn_mask, N