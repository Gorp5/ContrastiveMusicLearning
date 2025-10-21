import math
import random
import librosa
import pandas as pd
import os
import torch
import numpy as np

from data.data_utils import get_melspec_from_file
from data.processing import chunk_data, ReadStats
from mtgjamendodataset.scripts import commons
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_song(song_path, chunking=True):
    chunks = load_and_parse_audio(song_path, convert=True, chunking=chunking)
    id = song_path

def get_latents(dataloader, model, chunking=True, averaging=False, chunk_size=256):
    all_latents = []
    all_labels = []

    model.to("cuda")

    with torch.no_grad():
        for label, data in tqdm(dataloader):
            if chunking:
                data, num_chunks = chunk_data(data.squeeze(0), chunk_size=chunk_size)
            else:
                data = torch.tensor(data)
                data = data.unsqueeze(0)

            latent = run_batch(model, data, averaging=averaging)
            all_latents.append(latent)
            all_labels.append(label)

    return all_latents, all_labels


def run_batch(model, batch, averaging=True):
    batch = batch.to("cuda")
    B, T, F = batch.shape

    batch = batch.unsqueeze(1)
    # Needs to be Broken into minibatches
    num_chunks = max(1, math.ceil(B // 16))

    mini_batches = torch.chunk(batch, num_chunks, dim=0)
    latents = []
    for mini_batch in mini_batches:
        latent = model(mini_batch)
        latents.append(latent)

    latents = torch.cat(latents, dim=0)

    if averaging:
        averages = latents.mean(dim=0).cpu().numpy()
    else:
        averages = latents.cpu().numpy()

    torch.cuda.empty_cache()
    return averages

def inference_on_directory(subset_file_name, directory_to_parse, config, model, has_labels=False, num_genres=50, num_songs=None, device="cuda", chunking=False):
    subset_file = f'E:/mtg-jamendo-dataset/data/{subset_file_name}.tsv'

    tracks, tags, extra = commons.read_file(subset_file)
    tag_mapping = ReadStats(subset_file_name)

    all_tags = {}
    all_tags.update(tags['genre'])
    all_tags.update(tags['instrument'])
    all_tags.update(tags['mood/theme'])

    reversed_tag_mapping = {v: k for k, v in tag_mapping.items()}

    # Define columns
    predicted_columns = [f'Confidence {reversed_tag_mapping[i]}' for i in range(50)]
    actual_columns = [f'Tag {reversed_tag_mapping[i]}' for i in range(50)]
    columns = ['Track ID', 'Loss'] + predicted_columns + actual_columns

    # Initialize empty DataFrame
    df = pd.DataFrame(columns=columns)
    all_files_in_directory = sorted(os.listdir(directory_to_parse))

    random.seed(config.seed)
    random.shuffle(all_files_in_directory)

    if num_genres is not None:
        all_files_in_directory = all_files_in_directory[:num_songs]

    for file_name in tqdm(all_files_in_directory):
        if has_labels:
            track_id = int(file_name[:-3])
            song_tags = tracks[track_id]['tags']
            actual_tags = [0] * num_genres
            for t in song_tags:
                genre_index = tag_mapping[t]
                actual_tags[genre_index] = 1
        else:
            track_id = file_name[:-4].split(" - ")[1]
            actual_tags = None

        directory = os.path.join(directory_to_parse, file_name)
        processed_audio = load_and_parse_audio(directory, chunk=chunking)

        if processed_audio is None:
            continue

        predictions, loss = make_inference(model, processed_audio.to(device), config,  actual_tags)
        if chunking:
            average_predictions = np.mean(predictions, axis=0)
        else:
            average_predictions = predictions

        if actual_tags is None:
            actual_tags = [-1] * num_genres

        row = [track_id, loss] + average_predictions.tolist() + actual_tags
        df.loc[len(df)] = row
    return df


def load_and_parse_audio(full_path, convert=True, chunking=True, chunk_size=256):
    if not convert:
        data = np.load(full_path)
    else:
        if os.path.exists(full_path):
            audio, sr = librosa.load(full_path, sr=44100, mono=True)
            data = librosa.feature.melspectrogram(y=audio, sr=sr)
            data = librosa.amplitude_to_db(data, ref=np.max)
            # s_chroma = librosa.feature.chroma_stft(y=s, sr=sr)
        else:
            return None

    if chunking:
        chunked_data, num_chunks = chunk_data(data, chunk_size=chunk_size)
        return chunked_data
    else:
        return torch.from_numpy(data)



def make_inference(model, input, config, labels=None):
    test_loss_total = 0

    all_preds = []

    with torch.no_grad():
        inputs = input.squeeze(0).unsqueeze(1).to("cuda", config.dtype)
        num_chunks = int(inputs.shape[0] / config.max_batch_size) + 1

        data_minibatches = torch.chunk(inputs, num_chunks, dim=0)

        if labels is not None:
            labels = labels.squeeze(0).to("cuda", config.dtype)
            label_minibatches = torch.chunk(labels, num_chunks, dim=0)
        else:
            label_minibatches = torch.zeros(data_minibatches[0].shape)

        loss_per_batch = 0
        minibatch_len = len(data_minibatches)
        for i, (data_minibatch, label_minibatch) in enumerate(zip(data_minibatches, label_minibatches)):
            outputs = model(data_minibatch)

            if labels is not None:
                loss = config.criterion(outputs, label_minibatch)
                loss_per_batch += loss.item()

            all_preds.extend(outputs.sigmoid().cpu().numpy())

        test_loss_total += loss_per_batch / minibatch_len

    return all_preds, test_loss_total