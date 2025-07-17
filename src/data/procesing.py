import os
import torch

from tqdm import tqdm
from data.Data import retrieve_data
from torch.utils.data import Dataset

from utils.data import createDictionary, makeClassLabels


def ParseTaggedDataset(new_directory):
    num_per = 100
    count = 0
    length = 256

    for start in tqdm(range(0, 16000, num_per)):
        mtg_dataset, masks, keys = retrieve_data("E:\SongsDataset\\mtg-jamendo\\", start=start, count=num_per,
                                           sample_length=length, keep_song_data_option=True)

        dictionary = createDictionary()
        all_genre_labels, all_mood_labels = makeClassLabels(keys, dictionary)

        torch.save(mtg_dataset, f"E:\\SongsDataset\\{new_directory}\\data\\dataset{count:04d}.pt")
        torch.save(keys, f"E:\\SongsDataset\\{new_directory}\\keys\\dataset{count:04d}-keys.pt")
        torch.save(all_genre_labels, f"E:\\SongsDataset\\{new_directory}\\genre_labels\\dataset{count:04d}-labels.pt")
        torch.save(all_mood_labels, f"E:\\SongsDataset\\{new_directory}\\mood_labels\\dataset{count:04d}-labels.pt")
        torch.save(masks, f"E:\\SongsDataset\\{new_directory}\\masks\\dataset{count :04d}-masks.pt")
        count += 1

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
        song_path = os.path.join(self.song_dir, self.song_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        song = torch.load(song_path, map_location='cpu')  # memory-mapped access
        label = torch.load(label_path, map_location='cpu')
        return song, label