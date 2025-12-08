import os
import json
import requests

def download_file(url, dest_path, chunk_size=8192):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        print(f"Downloaded: {dest_path}")
    except Exception as e:
        print(f"Failed to download {url} → {dest_path}: {e}")

def main(json_file, target_folder):
    with open(json_file, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    for key, info in manifest.items():
        if key.startswith("GIANTSTEPS_MTG_KEY_"):
            url = info.get("url")
            path_rel = info.get("path_rel")
            if not url or not path_rel:
                print(f"Skipping {key}: missing url or path_rel")
                continue

            dest_path = os.path.join(target_folder, path_rel)

            download_file(url, dest_path)


import csv
import random

def read_metadata(meta_path):
    did_to_meta = {}
    with open(meta_path, 'r', encoding='utf‑8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            did = int(row['ID'])

            artists = [a.strip() for a in row['ARTIST'].split(',')]
            did_to_meta[did] = {**row, 'artists': artists}
    return did_to_meta


def stratify_split(did_to_meta, valid_frac=0.2, seed=42):
    random.seed(seed)

    artist_set = set()
    for meta in did_to_meta.values():
        for a in meta['artists']:
            artist_set.add(a)
    artist_list = list(artist_set)

    random.shuffle(artist_list)
    num_valid_artists = int(len(artist_list) * valid_frac)
    valid_artists = set(artist_list[:num_valid_artists])
    train_artists = set(artist_list[num_valid_artists:])

    train_ids = []
    valid_ids = []
    for did, meta in did_to_meta.items():
        if any(a in valid_artists for a in meta['artists']):
            valid_ids.append(did)
        else:
            train_ids.append(did)
    return train_ids, valid_ids


if __name__ == '__main__':
    meta_path = 'D:\\SongsDataset\\GS-MTG\\annotations\\beatport_metadata.txt'
    did_to_meta = read_metadata(meta_path)
    train_ids, valid_ids = stratify_split(did_to_meta, valid_frac=0.2, seed=123)

    print(f"Total tracks: {len(did_to_meta)}")
    print(f"Train tracks: {len(train_ids)}")
    print(f"Valid tracks: {len(valid_ids)}")

    with open('train_ids.txt', 'w') as f:
        for did in train_ids:
            f.write(f"{did}\n")
    with open('valid_ids.txt', 'w') as f:
        for did in valid_ids:
            f.write(f"{did}\n")