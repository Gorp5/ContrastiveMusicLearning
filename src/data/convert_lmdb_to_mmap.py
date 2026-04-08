import random

import lmdb
import pickle
import numpy as np
import os
from pathlib import Path

import torch
from tqdm import tqdm

def convert_lmdb_to_sharded_memmap(
    lmdb_root,
    split="train",
    num_shards=4,
    output_dir=None,
    dtype=np.float16,
    num_output_shards=16,  # number of memmap shards
):
    """
    Convert LMDB shards to multiple memmap shards.
    Writes continuously to disk without keeping all data in memory.
    Returns list of memmap paths and the index path.
    """
    if output_dir is None:
        output_dir = os.path.join(lmdb_root, f"{split}_memmap")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load track IDs for ordering
    id_file = os.path.join(lmdb_root, f"{split}_ids.npy")
    track_ids = np.load(id_file)

    print(f"Converting {len(track_ids)} tracks → {num_output_shards} memmap shards")

    # ---------------------------------------------
    # PASS 1: Compute total frames and freq bins
    # ---------------------------------------------
    total_frames = 0
    freq_bins = None

    print("\n[PASS 1] Scanning dataset...")
    for shard_idx in range(num_shards):
        shard_path = f"{lmdb_root}/{split}_shard{shard_idx}.lmdb"

        env = lmdb.open(
            shard_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            for key, data_bytes in tqdm(cursor, desc=f"Shard {shard_idx} scan"):
                arr = pickle.loads(bytes(data_bytes))  # shape: (freq, time)
                if freq_bins is None:
                    freq_bins = arr.shape[0]
                else:
                    assert arr.shape[0] == freq_bins, "Inconsistent freq bins"
                total_frames += arr.shape[1]
                del arr, data_bytes
        env.close()

    print(f"Total frames: {total_frames:,}")
    print(f"Freq bins: {freq_bins}")

    # ---------------------------------------------
    # Determine frames per memmap shard
    # ---------------------------------------------
    frames_per_shard = int(total_frames / num_output_shards * 1.05)
    print(f"Frames per output shard: {frames_per_shard:,}")

    # ---------------------------------------------
    # PASS 2: Write data continuously
    # ---------------------------------------------
    print("\n[PASS 2] Writing memmaps...")
    
    CHUNK_SIZE = 2500  # how many tracks to process before refreshing LMDB

    index = []
    shard_id = 0
    cursor_pos = 0
    current_mm = None

    for shard_idx in range(num_shards):
        shard_path = f"{lmdb_root}/{split}_shard{shard_idx}.lmdb"

        # ---------------------------
        # Prefetch keys (tiny memory)
        # ---------------------------
        env = lmdb.open(
            shard_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            keys = [bytes(k) for k, _ in cursor]
        env.close()

        # ---------------------------
        # Process keys in chunks
        # ---------------------------
        for i in range(0, len(keys), CHUNK_SIZE):
            chunk_keys = keys[i:i + CHUNK_SIZE]

            env = lmdb.open(
                shard_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with env.begin(buffers=True) as txn:
                for k in chunk_keys:
                    data_bytes = txn.get(k)
                    track_id = int(k.decode())
                    arr = pickle.loads(bytes(data_bytes))  # (freq, time)
                    arr = arr.T.astype(dtype, copy=False)  # (time, freq)
                    length = arr.shape[0]

                    # -------------------
                    # Write to memmap
                    # -------------------
                    if current_mm is None or cursor_pos + length > frames_per_shard:
                        if current_mm is not None:
                            current_mm.flush()
                            del current_mm
                        if shard_id >= num_output_shards:
                            print(f"[INFO] Allocating extra shard {shard_id}")
                        memmap_path = os.path.join(output_dir, f"{split}_{shard_id}.memmap")
                        current_mm = np.memmap(
                            memmap_path,
                            dtype=dtype,
                            mode="w+",
                            shape=(frames_per_shard, freq_bins),
                        )
                        cursor_pos = 0
                        shard_id += 1

                    current_mm[cursor_pos:cursor_pos + length] = arr
                    index.append((track_id, shard_id - 1, cursor_pos, length))
                    cursor_pos += length

                    del arr, data_bytes

            env.close()  # free LMDB memory for this chunk

    # Flush last memmap
    if current_mm is not None:
        current_mm.flush()
        del current_mm

    # ---------------------------------------------
    # Save index
    # ---------------------------------------------
    index_array = np.array(
        index,
        dtype=[
            ("track_id", "i8"),
            ("file_id", "i4"),
            ("start", "i8"),
            ("length", "i4"),
        ],
    )
    index_path = os.path.join(output_dir, f"{split}_index.npy")
    np.save(index_path, index_array)

    print("\n✅ Conversion complete!")
    print(f"Index: {index_path}")

    total_size = sum(os.path.getsize(os.path.join(output_dir, f"{split}_{i}.memmap"))
                     for i in range(num_output_shards))
    print(f"Total size: {total_size / 1e9:.2f} GB")

    memmap_paths = [os.path.join(output_dir, f"{split}_{i}.memmap") for i in range(num_output_shards)]
    return memmap_paths, index_path

class MemmapDataset:
    """
    Dataset for sharded memmap spectrogram storage.
    Each sample is retrieved via (file_id, start, length).
    """

    def __init__(
        self,
        memmap_root,
        split="train",
        chunk_size=256,
        views=2,
        min_chunk=-1,
        max_chunk=-1,
        dtype=np.float16,
        freq_bins=128,   # must match conversion
    ):
        self.memmap_root = memmap_root
        self.split = split
        self.chunk_size = chunk_size
        self.view_count = views
        self.stochastic = min_chunk > 0 and max_chunk > 0
        self.min_chunk, self.max_chunk = min_chunk, max_chunk
        self.dtype = dtype
        self.freq_bins = freq_bins

        # ---------------------------
        # Load index
        # ---------------------------
        index_path = os.path.join(memmap_root, f"{split}_index.npy")
        self.index = np.load(index_path)

        self.track_ids = self.index["track_id"]

        # ---------------------------
        # Lazy memmap cache
        # ---------------------------
        self.memmaps = {}

    def __len__(self):
        return len(self.index)

    def _get_memmap(self, file_id):
        """
        Lazily open memmap shard.
        Each worker will have its own cache.
        """
        if file_id not in self.memmaps:
            path = os.path.join(
                self.memmap_root,
                f"{self.split}_{file_id}.memmap"
            )

            # IMPORTANT: shape must match conversion
            # We don't know total rows, so use mode='r' and reshape via -1
            mm = np.memmap(
                path,
                dtype=self.dtype,
                mode="r"
            )

            mm = mm.reshape(-1, self.freq_bins)
            self.memmaps[file_id] = mm

        return self.memmaps[file_id]

    def __getitem__(self, idx):
        row = self.index[idx]

        track_id = int(row["track_id"])
        file_id = int(row["file_id"])
        start = int(row["start"])
        length = int(row["length"])

        # ---------------------------
        # Load slice (ZERO COPY)
        # ---------------------------
        mm = self._get_memmap(file_id)
        spec = mm[start:start + length]  # (time, freq)

        # convert to torch (still cheap)
        spec = torch.from_numpy(spec).float().T  # → (freq, time)

        # ---------------------------
        # Generate views
        # ---------------------------
        views, masks = [], []

        for _ in range(self.view_count):
            size = self.chunk_size

            if self.stochastic:
                size = random.randint(
                    self.min_chunk // 16,
                    self.max_chunk // 16
                ) * 16

            if spec.shape[1] <= size:
                pad = size - spec.shape[1]
                view = torch.nn.functional.pad(spec, (0, pad))
                mask = torch.cat([
                    torch.ones(spec.shape[1]),
                    torch.zeros(pad)
                ]).bool()

            else:
                start_idx = random.randint(0, spec.shape[1] - size)
                view = spec[:, start_idx:start_idx + size]
                mask = torch.ones(size, dtype=torch.bool)


            views.append(view)
            masks.append(mask)

        return track_id, torch.stack(views), torch.stack(masks)

if __name__ == "__main__":
    convert_lmdb_to_sharded_memmap("C:\SongsDataset",
    split="train",
    num_shards=4,
    output_dir="E:\SongsDataset\mtg-specs",
    dtype=np.float16,
    num_output_shards= 16)