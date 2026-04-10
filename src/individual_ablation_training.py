import argparse
import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from google.cloud import storage

from models.Myna import Myna
from data.data_utils import MemmapDataset
from info_nce import InfoNCE
from torch import optim


# ---------------------------
# Model
# ---------------------------
def build_model(mask_ratio, chunk_length, embedding_params):
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
        use_learned_encoding_y=embedding_params.get("learned_y", False),
        use_learned_encoding_x=embedding_params.get("learned_x", False),
        use_rope_x=embedding_params.get("rope_x", False),
        use_rope_y=embedding_params.get("rope_y", False),
        use_alibi_x=embedding_params.get("alibi_x", False),
        use_alibi_y=embedding_params.get("alibi_y", False),
        use_learned_alibi_slopes=embedding_params.get("alibi_learned_slopes", False),
        rope_base=8192
    )


# ---------------------------
# DataLoader (NO DISTRIBUTED SAMPLER)
# ---------------------------
def build_dataloader(dataset_path, batch_size, chunk_length):
    dataset = MemmapDataset(dataset_path, split="train", views=2, chunk_size=chunk_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )


# ---------------------------
# GCS upload
# ---------------------------
def upload_to_gcs(local_path, bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)


# ---------------------------
# GPU WORKER (NO DDP, NO NCCL)
# ---------------------------
def gpu_worker(gpu_id, args, model_params_list):
    print(f"[GPU {gpu_id}] worker starting")

    # Each process sees ONLY its GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    torch.cuda.set_device(device)

    models = []
    optimizers = []

    # Build models assigned to this GPU
    for params in model_params_list[gpu_id]:
        model = build_model(
            params["mask_ratio"],
            params["chunk_length"],
            params["embedding_params"]
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        models.append(model)
        optimizers.append(optimizer)

    criterion = InfoNCE()

    dataloader = build_dataloader(
        args.train_data_dir,
        args.batch_size,
        args.chunk_length
    )

    save_dir = os.path.join(args.save_dir, f"GPU-{gpu_id}")
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------
    # TRAINING LOOP
    # ---------------------------
    for epoch in range(args.epochs):
        epoch_losses = [0.0 for _ in models]

        pbar = tqdm(dataloader, desc=f"GPU {gpu_id} Epoch {epoch}")

        for batch in pbar:
            _, inputs, _ = batch
            inputs = inputs.to(device)

            B, _, T, F = inputs.shape

            for i, (model, optimizer) in enumerate(zip(models, optimizers)):
                optimizer.zero_grad(set_to_none=True)

                stacked = inputs.view(B * 2, T, F).unsqueeze(1)

                z = model(stacked, mask=None).squeeze(1).view(B, 2, -1)

                loss = 0.0
                for j in range(1, z.shape[1]):
                    loss += criterion(z[:, 0], z[:, j])

                loss = loss / (z.shape[1] - 1)

                loss.backward()
                optimizer.step()

                epoch_losses[i] += loss.item()

        # ---------------------------
        # SAVE CHECKPOINTS
        # ---------------------------
        for i, model in enumerate(models):
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizers[i].state_dict(),
                "loss": epoch_losses[i] / len(dataloader)
            }, os.path.join(save_dir, f"model_{i}.pt"))

    # ---------------------------
    # UPLOAD (rank 0 GPU only optional, but safe to let all do it)
    # ---------------------------
    if gpu_id == 0:
        print("Uploading to GCS...")
        for root, _, files in os.walk(args.save_dir):
            for f in files:
                local_path = os.path.join(root, f)
                rel_path = os.path.relpath(local_path, args.save_dir)

                upload_to_gcs(
                    local_path,
                    bucket_name="mtg-jamendo",
                    blob_name=f"checkpoints/{rel_path}"
                )


# ---------------------------
# CONFIG MAPPING
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
# MAIN SCHEDULER
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=512)
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

    # ---------------------------
    # Assign models to GPUs
    # ---------------------------
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

    # ---------------------------
    # Launch workers (NO DDP)
    # ---------------------------
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