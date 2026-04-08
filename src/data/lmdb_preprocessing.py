import lmdb
import pickle
import numpy as np
from tqdm import tqdm
import random
import librosa
import os
import multiprocessing as mp
import shutil
import gc

from data.processing import ReadStats
from mtgjamendodataset.scripts import commons


def process_track_worker(args):
    track_id, all_tracks, tracks, data_location, num_genres, tag_mapping, test_prob, base_seed = args

    tags = all_tracks[track_id]
    path_end = tracks[track_id]['path']
    path_stem = "mp3"
    full_path = data_location + path_end[:-3] + path_stem

    labels = [0] * num_genres
    for t in tags:
        genre_index = tag_mapping[t]
        labels[genre_index] = 1

    if not os.path.exists(full_path):
        return None

    # Load audio with memory-efficient settings
    audio, sr = librosa.load(full_path, sr=44100, mono=True, res_type='kaiser_fast')
    win_length = int(round(0.025 * sr))
    hop_length = int(round(0.010 * sr))
    n_fft = 2048

    data = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        n_mels=128, fmin=0, fmax=sr / 2, power=2.0
    )
    data = librosa.amplitude_to_db(data, ref=np.max)
    data = data.astype(np.float16)

    # Clear audio from memory immediately
    del audio

    random.seed(base_seed + track_id)

    split = "train" if random.random() > test_prob else "test"
    shard = track_id % 4
    key = f"{track_id:06d}"

    return (split, shard, key, data)


def parse_parallel(selected_tracks, output_dir, tag_mapping, data_location,
                   all_tracks, tracks, test_prob=0.1, num_workers=4, batch_size=50):
    """
    Process tracks with batched writes to prevent transaction overhead and handle
    memory constraints properly.

    Args:
        batch_size: Number of tracks to buffer per shard before writing (reduces transaction overhead)
    """
    num_shards = 4
    num_genres = 50

    # Calculate realistic map sizes based on actual data
    # Approx 4MB per track (float16 mel spec) + 20% pickle overhead
    estimated_bytes_per_track = 5 * 1024 * 1024
    total_tracks = len(selected_tracks)

    train_ratio = 1 - test_prob
    train_per_shard = int((total_tracks * train_ratio / num_shards) * estimated_bytes_per_track * 1.2)
    test_per_shard = int((total_tracks * test_prob / num_shards) * estimated_bytes_per_track * 1.2)

    print(
        f"Allocating {train_per_shard / 1024 ** 2:.0f}MB per train shard, {test_per_shard / 1024 ** 2:.0f}MB per test shard")

    # Setup environments without writemap (Windows memory constraint)
    train_envs = []
    test_envs = []

    for i in range(num_shards):
        train_path = f'{output_dir}/train_shard{i}.lmdb'
        test_path = f'{output_dir}/test_shard{i}.lmdb'

        if os.path.exists(train_path):
            shutil.rmtree(train_path)
        if os.path.exists(test_path):
            shutil.rmtree(test_path)

        train_envs.append(lmdb.open(
            train_path,
            map_size=train_per_shard,
            sync=False,
            metasync=False,
            readahead=False,  # Prevent OS from caching entire file
            max_readers=1
        ))
        test_envs.append(lmdb.open(
            test_path,
            map_size=test_per_shard,
            sync=False,
            metasync=False,
            readahead=False,
            max_readers=1
        ))

    # Buffers for batching writes: buffers[split][shard] = [(key, data), ...]
    buffers = {
        'train': [[] for _ in range(num_shards)],
        'test': [[] for _ in range(num_shards)]
    }

    # COLLECT IDS HERE
    split_ids = {'train': [], 'test': []}

    def flush_buffer(split, shard, force=False):
        buffer = buffers[split][shard]
        if not buffer or (len(buffer) < batch_size and not force):
            return True

        env = train_envs[shard] if split == "train" else test_envs[shard]
        path = f'{output_dir}/{split}_shard{shard}.lmdb'

        try:
            with env.begin(write=True) as txn:
                for key, packed in buffer:
                    txn.put(key.encode(), packed)
            buffer.clear()
            return True

        except lmdb.MapFullError:
            env.close()
            current_size = train_per_shard if split == "train" else test_per_shard
            new_size = current_size + (500 * 1024 * 1024)

            print(f"\nResizing {split} shard {shard} to {new_size / 1024 ** 2:.0f}MB...")

            new_env = lmdb.open(
                path,
                map_size=new_size,
                sync=False,
                metasync=False,
                readahead=False,
                max_readers=1
            )

            if split == "train":
                train_envs[shard] = new_env
            else:
                test_envs[shard] = new_env

            with new_env.begin(write=True) as txn:
                for key, packed in buffer:
                    txn.put(key.encode(), packed)
            buffer.clear()
            return True

    BASE_SEED = 42

    worker_args = [
        (tid, all_tracks, tracks, data_location, num_genres, tag_mapping, test_prob, BASE_SEED)
        for tid in selected_tracks
    ]

    written = 0
    processed = 0

    with mp.Pool(num_workers, maxtasksperchild=10) as pool:
        iterator = pool.imap_unordered(process_track_worker, worker_args, chunksize=1)

        with tqdm(total=len(selected_tracks)) as pbar:
            for result in iterator:
                processed += 1

                if result is None:
                    pbar.update(1)
                    continue

                split, shard, key, data = result
                split_ids[split].append(int(key))  # STORE ID HERE

                packed = pickle.dumps(data, protocol=4)
                buffers[split][shard].append((key, packed))

                if len(buffers[split][shard]) >= batch_size:
                    flush_buffer(split, shard)

                del result, data, packed

                if processed % 100 == 0:
                    for s in ['train', 'test']:
                        for sh in range(num_shards):
                            flush_buffer(s, sh, force=True)

                    if processed % 1000 == 0:
                        for env in train_envs + test_envs:
                            env.sync()
                    gc.collect()

                written += 1
                pbar.update(1)

    print("Final flush...")
    for split in ['train', 'test']:
        for shard in range(num_shards):
            flush_buffer(split, shard, force=True)

    # SAVE THE ID FILES AT THE END
    for split_name, ids in split_ids.items():
        id_file = os.path.join(output_dir, f"{split_name}_ids.npy")
        np.save(id_file, np.array(ids, dtype=np.int32))
        print(f"Saved {len(ids)} IDs to {id_file}")

    for env in train_envs + test_envs:
        env.sync()
        env.close()

    print(f"Successfully wrote {written} tracks")


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

    # Check disk space
    total, used, free = shutil.disk_usage(output_directory)
    estimated_need = len(tracks) * 5 / 1024  # ~5MB per track
    print(f"Free space: {free / (1024 ** 3):.1f} GB")
    print(f"Estimated need: {estimated_need:.1f} GB")

    print("Processing tracks...")

    parse_parallel(
        selected_tracks=list(tracks.keys()),
        output_dir=output_directory,
        tag_mapping=tag_mapping,
        data_location=data_location,
        all_tracks=all_tracks,
        tracks=tracks,
        test_prob=0.1,
        num_workers=4,  # Reduced to prevent RAM exhaustion
        batch_size=50  # Write 50 tracks per transaction
    )


if __name__ == "__main__":
    subset_name = "autotagging_top50tags"
    subset_file = f"../mtgjamendodataset/data/{subset_name}.tsv"
    subset_data = f'../mtgjamendodataset/stats/{subset_name}/all.tsv'
    data_location = "E:/mtg-jamendo/"
    output_directory = "E:/SongsDataset/"

    ParseAll(subset_file, subset_data, data_location, output_directory)