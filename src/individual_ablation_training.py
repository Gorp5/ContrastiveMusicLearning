import argparse
import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm
from google.cloud import storage
from torch import optim

from models.Myna import Myna
from data.data_utils import MemmapDataset
from info_nce import InfoNCE


# ---------------------------
# SPEED FIX: multiprocessing safety
# ---------------------------
mp.set_start_method("spawn", force=True)
torch.multiprocessing.set_sharing_strategy("file_system")


# ---------------------------
# Model
# ---------------------------
def build_model(mask_ratio, chunk_length, embedding_params, device):
    return Myna(
        image_size=(128, chunk_length),
        channels=1,
        patch_size=(16, 16),
        latent_space=128,
        d_model=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        mask_ratio=mask_ratio,
        latent_projection_method="cls",
        use_sinusoidal_x=embedding_params.get("sinusoidal_x", False),
        use_sinusoidal_y=embedding_params.get("sinusoidal_y", False),
        use_sinusoidal_raster=embedding_params.get("sinusoidal_raster", False),
        use_learned_encoding_x=embedding_params.get("learned_x", False),
        use_learned_encoding_y=embedding_params.get("learned_y", False),
        use_rope_x=embedding_params.get("rope_x", False),
        use_rope_y=embedding_params.get("rope_y", False),
        use_alibi_x=embedding_params.get("alibi_x", False),
        use_alibi_y=embedding_params.get("alibi_y", False),
        use_learned_alibi_slopes=embedding_params.get("alibi_learned_slopes", False),
        rope_base=8192,
        device=device
    )


# ---------------------------
# DataLoader (FAST + SAFE)
# ---------------------------
def build_dataloader(dataset_path, batch_size, chunk_length):
    dataset = MemmapDataset(dataset_path, split="train", views=2, chunk_size=chunk_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=3
    )


# ---------------------------
# GPU WORKER
# ---------------------------
def gpu_worker(gpu_id, args, model_params_list):
    print(f"[GPU {gpu_id}] worker starting")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    # torch.cuda.set_device(device)

    models, optimizers = [], []

    for params in model_params_list[gpu_id]:
        model = build_model(
            params["mask_ratio"],
            params["chunk_length"],
            params["embedding_params"],
            device
        ).to(device)

        model.train()

        optimizers.append(
            optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        )

        models.append(model)

    criterion = InfoNCE()

    dataloader = build_dataloader(
        args.train_data_dir,
        args.batch_size,
        args.chunk_length
    )

    save_dir = os.path.join(args.save_dir, f"GPU-{gpu_id}")
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------
    # TRAIN LOOP
    # ---------------------------
    for epoch in range(args.epochs):
        losses = [0.0 for _ in models]

        pbar = tqdm(dataloader, desc=f"GPU {gpu_id} Epoch {epoch}")

        for batch in pbar:
            _, inputs, _ = batch
            inputs = inputs.to(device, non_blocking=True)

            zs = []
            T_full = 2048
            for model, optimizer, params in zip(models, optimizers, model_params_list[gpu_id]):
                chunk_len = params["chunk_length"]

                if chunk_len < T_full:
                    start = torch.randint(0, T_full - chunk_len + 1, (1,)).item()
                    sliced = inputs[:, :, start:start + chunk_len, :]
                else:
                    sliced = inputs

                stacked = sliced.view(B * 2, chunk_len, F).unsqueeze(1)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z = model(stacked, mask=None).squeeze(1).view(B, 2, -1)
                    loss = criterion(z[:, 0], z[:, 1])

                loss.backward()
                optimizer.step()

            for i, (z, model, optimizer) in enumerate(zip(zs, models, optimizers)):
                optimizer.zero_grad(set_to_none=True)

                loss = criterion(z[:, 0], z[:, 1])
                loss.backward()
                optimizer.step()

                losses[i] += loss.item()

        # ---------------------------
        # SAVE
        # ---------------------------
        for i, model in enumerate(models):
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizers[i].state_dict(),
                "loss": losses[i] / len(dataloader)
            }, os.path.join(save_dir, f"model_{i}.pt"))


# ---------------------------
# CONFIG
# ---------------------------
def determine_based_on_id(id):
    masking_ratio_array = [0.25, 0.5, 0.75, 0.9]
    training_chunk_length_array = [128, 256, 512, 1024, 2048]

    embedding_configs = [
        dict(name="alibi_2d_learned", alibi_x=True, alibi_y=True, alibi_learned_slopes=True),
        dict(name="alibi_2d", alibi_x=True, alibi_y=True),
        dict(name="rope_2d", rope_x=True, rope_y=True),
        dict(name="alibi_1d", alibi_x=True),
        dict(name="rope_1d", rope_x=True),
        dict(name="sinusoidal_raster", sinusoidal_raster=True),
        dict(name="learned_x", learned_x=True, learned_y=True),
        dict(name="none"),
        dict(name="sinusoidal_xy", sinusoidal_x=True, sinusoidal_y=True),
        dict(name="rope_double_frequency", rope_x=True, rope_y=True),
    ]

    config = embedding_configs[id % len(embedding_configs)]

    return (
        masking_ratio_array[id % len(masking_ratio_array)],
        training_chunk_length_array[(id // 4) % len(training_chunk_length_array)],
        config
    )


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--chunk_length", type=int, required=True)

    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--num_models", type=int, required=True)

    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    args = parser.parse_args()

    models_per_gpu = [[] for _ in range(args.num_gpus)]

    for i in range(args.num_models):
        i = i + args.id
        gpu_id = i % args.num_gpus

        mask_ratio, chunk_length, params = determine_based_on_id(i)

        models_per_gpu[gpu_id].append({
            "mask_ratio": mask_ratio,
            "chunk_length": chunk_length,
            "embedding_params": params
        })

    processes = []

    for gpu_id in range(args.num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, args, models_per_gpu)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()