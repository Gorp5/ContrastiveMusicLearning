import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import pickle
from google.cloud import storage
from models.Myna import Myna
from data.data_utils import StreamViewDataset, MemmapDataset
from info_nce import InfoNCE
from torch import optim

# ---------------------------
# Distributed helpers
# ---------------------------
def setup_process_group(rank, world_size, backend="nccl"):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(rank))
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def is_main_process(rank):
    return rank == 0

# ---------------------------
# Model & dataloader builders
# ---------------------------
def build_model(mask_ratio, chunk_length, embedding_params):
    model = Myna(
        image_size=(128, chunk_length),
        channels=1,
        patch_size=(16,16),
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
        use_rope_double_frequency=False,
        use_alibi_x=embedding_params.get("alibi_x", False),
        use_alibi_y=embedding_params.get("alibi_y", False),
        use_learned_alibi_slopes=embedding_params.get("alibi_learned_slopes", False),
        rope_base=8192
    )
    return model

def build_dataloader(dataset_path, batch_size, rank, world_size, chunk_length):
    dataset = MemmapDataset(dataset_path, split="test", views=2, chunk_size=chunk_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    return dataloader

def upload_to_gcs(local_path, bucket_name, blob_name):
    client = storage.Client()  # uses GOOGLE_APPLICATION_CREDENTIALS env var
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

# ---------------------------
# Training per GPU
# ---------------------------
def gpu_worker(rank, world_size, args, model_params_list):
    print(f"GPU {rank}: starting worker...")
    torch.cuda.set_device(rank)
    setup_process_group(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # 1) Instantiate models for this GPU
    models = []
    optimizers = []
    ddp_models = []
    streams = []
    for params in model_params_list[rank]:
        model = build_model(params["mask_ratio"], params["chunk_length"], params["embedding_params"]).to(device)
        ddp_model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        optimizer = optim.AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        stream = torch.cuda.Stream(device=device)

        models.append(model)
        ddp_models.append(ddp_model)
        optimizers.append(optimizer)
        streams.append(stream)

    # 2) Build dataloader
    dataloader = build_dataloader(args.train_data_dir, args.batch_size // world_size, rank, world_size, args.chunk_length)
    criterion = InfoNCE()

    # 3) Create save directories
    save_dir = os.path.join(args.save_dir, f"GPU-{rank}")
    os.makedirs(save_dir, exist_ok=True)
    if is_main_process(rank):
        with open(os.path.join(save_dir, "Losses.pkl"), "wb") as f:
            pickle.dump({}, f)

    # 4) Training loop
    for epoch in range(args.epochs):
        dataloader.sampler.set_epoch(epoch)
        epoch_losses = [0.0 for _ in ddp_models]

        pbar = tqdm(dataloader, desc=f"GPU {rank} Epoch {epoch}", disable=not is_main_process(rank))
        for batch in pbar:
            indices, inputs, masks = batch
            B, _, T, F = inputs.shape
            inputs = inputs.to(device)

            # Step 1: record forward/backward in separate streams
            handles = []
            for i, (ddp_model, optimizer, stream) in enumerate(zip(ddp_models, optimizers, streams)):
                handle = torch.cuda.StreamContext(stream)
                with torch.cuda.stream(stream):
                    optimizer.zero_grad(set_to_none=True)
                    stacked = inputs.view(B*2, T, F).unsqueeze(1)
                    z_list = ddp_model(stacked, mask=None).squeeze(1).view(B, 2, -1)
                    loss = 0.0
                    for idx in range(1, len(z_list)):
                        loss += criterion(z_list[0], z_list[idx])
                    loss /= (len(z_list)-1)
                    loss.backward()
                    optimizer.step()
                    epoch_losses[i] += loss.detach().item()

            # Step 2: synchronize all streams before next batch
            for stream in streams:
                stream.synchronize()

        # Save checkpoints
        for i, model in enumerate(models):
            model_name = f"Model-{rank}-{i}"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizers[i].state_dict(),
                "loss": epoch_losses[i]/len(dataloader)
            }, os.path.join(save_dir, f"{model_name}-Epoch{epoch}.pt"))

        # Save checkpoints
        for i, model in enumerate(models):
            model_name = f"Model-{rank}-{i}"
            local_path = os.path.join(save_dir, f"{model_name}-Epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model": model,
                "optimizer_state_dict": optimizers[i].state_dict(),
                "loss": epoch_losses[i] / len(dataloader)
            }, local_path)

            # upload to GCS
            upload_to_gcs(local_path, bucket_name="gs://mtg-jamendo",
                          blob_name=f"checkpoints/{model_name}-Epoch{epoch}.pt")

        # Save epoch losses globally
        if is_main_process(rank):
            local_losses = os.path.join(save_dir, "Losses.pkl")
            with open(local_losses, "wb") as f:
                pickle.dump(epoch_losses, f)
            upload_to_gcs(local_losses, bucket_name="gs://mtg-jamendo", blob_name="Losses.pkl")

    cleanup()


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, required=False, default=512)
    parser.add_argument("--epochs", type=int, required=False, default=16)
    parser.add_argument("--chunk_length", type=int, required=True)
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--num_models", required=True, type=int, default=1)

    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--train_data_dir", required=True, type=str)
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--lr", required=False, type=float, default=1e-4)
    parser.add_argument("--weight_decay", required=False, type=float, default=1e-4)

    args = parser.parse_args()

    # ---------------------------
    # Distribute models across GPUs
    # ---------------------------
    models_per_gpu = [[] for _ in range(args.num_gpus)]
    for i in range(args.num_models):
        gpu_id = i % args.num_gpus
        models_per_gpu[gpu_id].append({
            "mask_ratio": 0.9,
            "chunk_length": args.chunk_length,
            "embedding_params": {}
        })

    mp.spawn(gpu_worker, args=(args.num_gpus, args, models_per_gpu), nprocs=args.num_gpus, join=True)