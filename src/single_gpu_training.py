import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

from functools import partial
from info_nce import InfoNCE
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from functools import partial

from contrastive_training import train_contrastive
from data.data_utils import StreamViewDataset
from models.Myna import Myna
from utils.Config import Config

if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--latent_projection_method", type=str, required=False, default="cls")
    parser.add_argument("--mask_ratio", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--chunk_length", type=int, required=True)
    parser.add_argument("--rope_base", type=int, required=False, default=4096)
    parser.add_argument("--use_sinusoidal_raster", type=bool, required=False, default=False)
    parser.add_argument("--use_sinusoidal_x", type=bool, required=False, default=False)
    parser.add_argument("--use_sinusoidal_y", type=bool, required=False, default=False)

    parser.add_argument("--use_y_emb", type=bool, required=False, default=False)
    parser.add_argument("--use_x_emb", type=bool, required=False, default=False)

    parser.add_argument("--use_rope_x", type=bool, required=False, default=False)
    parser.add_argument("--use_rope_y", type=bool, required=False, default=False)
    parser.add_argument("--use_alibi_x", type=bool, required=False, default=False)
    parser.add_argument("--use_alibi_y", type=bool, required=False, default=False)
    parser.add_argument("--use_time_chunking", type=bool, required=False, default=False)

    args = parser.parse_args()

    per_gpu_batch = args.batch_size

    config = Config(
        save_path=args.save_dir,
        num_epochs=args.epochs,
        learning_rate=3e-4,
        weight_decay=1e-4,
        num_workers=2,
        batch_size=per_gpu_batch,
        eval_batch_size=per_gpu_batch,
        dtype=torch.float32
    )

    patch_size = (16, 16)
    if args.use_time_chunking:
        patch_size = (128, 1)

    # Determines which model is trained
    model = Myna(
        image_size=(128, args.chunk_length),
        channels=1,
        patch_size=(16, 16),
        latent_space=128,
        d_model=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        mask_ratio=args.mask_ratio,
        latent_projection_method=args.latent_projection_method,
        use_sinusoidal_x=args.use_sinusoidal_x,
        use_sinusoidal_y=args.use_sinusoidal_y,
        use_sinusoidal_raster=args.use_sinusoidal_raster,
        use_learned_encoding_y=args.use_y_emb,
        use_learned_encoding_x=args.use_x_emb,
        use_rope_x=args.use_rope_x,
        use_rope_y=args.use_rope_y,
        use_rope_double_frequency=False,
        use_alibi_x=args.use_alibi_x,
        use_alibi_y=args.use_alibi_y,
        rope_base=args.rope_base
    )

    train_dataset = StreamViewDataset(args.dataset_dir, views=2, chunk_size=args.chunk_length)
    test_dataset = StreamViewDataset(args.dataset_dir, views=2, chunk_size=args.chunk_length)

    train_contrastive(model, test_dataset, train_dataset, config)