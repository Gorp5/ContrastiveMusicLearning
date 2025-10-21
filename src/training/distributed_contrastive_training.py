import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from info_nce import InfoNCE
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from data.data_utils import StreamViewDataset
from models.Myna import ViTEncoder
from training.contrastive_training import evaluate_contrastive
from utils.Config import Config


def setup_process_group(rank, world_size, backend="nccl", master_addr=None, master_port=None):
    if master_addr:
        os.environ.setdefault("MASTER_ADDR", master_addr)
    if master_port:
        os.environ.setdefault("MASTER_PORT", master_port)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(rank))
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def reduce_mean(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def is_main_process(rank):
    return rank == 0

def ddp_train_worker(rank, world_size, model_fn, build_datasets_fn, config,
                     start_epoch=0, master_addr=None, master_port=None):

    # 1) init
    torch.cuda.set_device(rank)
    setup_process_group(rank, world_size, master_addr=master_addr, master_port=master_port)
    device = torch.device(f"cuda:{rank}")

    # 2) create model, move to device and wrap in DDP
    model = model_fn()
    model.to(device, dtype=getattr(torch, config.dtype))
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # 3) build dataloaders with DistributedSampler
    train_dataset, test_dataset = build_datasets_fn()
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=None
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size if hasattr(config, "eval_batch_size") else config.batch_size,
        sampler=test_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # 4) set up optimizer / loss
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = InfoNCE()

    if is_main_process(rank):
        file_path = os.path.join(".", config.save_path, "Config.pt")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(config, file_path)
        # Clear loss file
        with open(os.path.join(".", config.save_path, "Loss.txt"), "w") as f:
            pass

    torch.autograd.set_detect_anomaly(True)

    world_size_f = float(world_size)
    step = 1

    # 5) training loop
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        batch_steps = 0
        epoch_contrastive_loss = 0.0

        batches = len(train_dataloader)
        pbar = tqdm(train_dataloader, desc=f"Rank {rank} Epoch {epoch}", disable=(not is_main_process(rank)))

        for batch in pbar:
            indicies, inputs = batch

            # move inputs to device
            for index, view in enumerate(inputs):
                v = view.to(device).unsqueeze(1)
                inputs[index] = v
                B, _, T, F = v.shape

            stacked = torch.cat(inputs, dim=0)
            z_stacked = model(stacked)
            z_list = torch.split(z_stacked, B, dim=0)

            contrastive_loss = 0.0
            for i in range(1, len(z_list)):
                contrastive_loss = contrastive_loss + criterion(z_list[0], z_list[i])
            contrastive_loss = contrastive_loss / (len(z_list) - 1)
            loss = contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate local metrics
            epoch_contrastive_loss += contrastive_loss.detach().cpu()
            batch_steps += 1
            step += 1

            if is_main_process(rank):
                term = f"Contrastive Loss [{batch_steps}/{batches}]: {contrastive_loss.item():.4f}\n"
                with open(os.path.join(".", config.save_path, "Loss.txt"), "a") as f:
                    f.write(term)
                pbar.set_postfix({"train_loss": f"{contrastive_loss.item():.4f}"})

        # reduce/average losses across ranks to get global metric
        local_contrastive_sum = torch.tensor(epoch_contrastive_loss, dtype=torch.float32, device=device)
        local_batches = torch.tensor(batch_steps, dtype=torch.float32, device=device)

        # Sum across all ranks
        dist.all_reduce(local_contrastive_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_batches, op=dist.ReduceOp.SUM)

        avg_train_contrastive_loss = (local_contrastive_sum / local_batches).item() if local_batches.item() > 0 else float("nan")

        # synchronize before evaluation
        dist.barrier()

        # run evaluate on rank 0
        if is_main_process(rank):
            same_song_contrastive_loss = evaluate_contrastive(model.module, test_dataloader, config)
        else:
            same_song_contrastive_loss = None

        same_song_tensor = torch.tensor([same_song_contrastive_loss if same_song_contrastive_loss is not None else -1.0],
                                       dtype=torch.float32, device=device)
        dist.broadcast(same_song_tensor, src=0)
        same_song_contrastive_loss = float(same_song_tensor.item())

        # Save checkpoint on main process
        if is_main_process(rank):
            torch.save(model.module, os.path.join(".", config.save_path, f"Epoch-{epoch}.pt"))
            term = f"[Epoch {epoch}] Train: Same Song Contrastive Loss = {avg_train_contrastive_loss:.4f}"
            term += f"\nTest: Same Song Contrastive Loss = {same_song_contrastive_loss:.4f}\n"
            print(term)

    # cleanup
    cleanup()


def alibi_model_fn():
    return ViTEncoder(
        image_size=(128, 256),
        channels=1,
        patch_size=(16, 16),
        latent_space=128,
        d_model=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        mask_ratio=0.9,
        use_cls=True,
        alibi=True
    )


def sinusoidal_model_fn():
    return ViTEncoder(
        image_size=(128, 256),
        channels=1,
        patch_size=(16, 16),
        latent_space=128,
        d_model=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        mask_ratio=0.9,
        use_cls=True,
        alibi=False
    )


def make_build_datasets_fn(dataset_dir, views=2):
    def build_datasets_fn():
        train_dataset = StreamViewDataset(dataset_dir, views=views)
        test_dataset = StreamViewDataset(dataset_dir, views=views)
        return train_dataset, test_dataset
    return build_datasets_fn


if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--model", type=str, choices=["alibi", "sinusoidal"], default="alibi")
    args = parser.parse_args()

    config = Config(
        save_path=args.save_dir,
        num_epochs=512,
        learning_rate=3e-4,
        weight_decay=1e-4,
        num_workers=1,
        batch_size= args.batch_size,
        eval_batch_size=args.batch_size,
        dtype=torch.float32
    )

    # Determines which model is trained
    model_fn = alibi_model_fn if args.model == "alibi" else sinusoidal_model_fn

    build_datasets_fn = make_build_datasets_fn(args.dataset_dir)

    mp.spawn(ddp_train_worker,
             args=(world_size, model_fn, build_datasets_fn, config, 0,  None, None),
             nprocs=world_size,
             join=True)