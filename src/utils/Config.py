import os

import torch


class Config():
    def __init__(self, save_path, num_epochs, learning_rate, weight_decay, batch_size=512, num_workers=1, eval_batch_size=64, dtype=torch.float32):
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_batch_size = eval_batch_size
        self.dtype = dtype

    def save(self, filename=None):
        if filename is None:
            filename = os.path.join(self.save_path, "Config.pt")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.__dict__, filename)

    @classmethod
    def load(cls, filename):
        state = torch.load(filename, map_location='cpu')
        config = cls.__new__(cls)  # create instance without calling __init__
        config.__dict__.update(state)
        return config