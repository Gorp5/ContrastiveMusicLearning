import math
import os
import time

from libauc.losses import GCLoss_v1
from libauc.optimizers import SogCLR

from loss import loss_utils
from loss.loss_utils import *
from datasets import tqdm
from torch import optim
from torch.utils.data import DataLoader

def get_beta(percentage, cycles=8, coef=1, warmup=2):
    if percentage < warmup:
        return 1e-8

    percentage *= 2 * cycles
    percentage = (percentage + warmup) % 2

    if percentage > 1:
        percentage = 1

    return percentage * coef

def get_mask_schedule(current_step, current_epoch, current_batch, total_steps, total_epochs, total_batches, max=0.9):
    if current_epoch > 1:
        return max

    return (current_batch / total_batches) * max


def set_schedulers(epoch: int, criterion: nn.Module, optimizer: optim.Optimizer, args: None):
    if args is None:
        raise Exception("No args provided")

    # gamma schedule
    if isinstance(criterion, GCLoss_v1):
        criterion.adjust_gamma(epoch)

        # if args.rank == 0:
        #     print(f'Adjusted gamma according to schedule: {criterion.gamma:.5f}')

    # learning rate schedule
    if args.lr_schedule.lower() == 'cosine':
        # warmup
        if epoch < args.warmup_epochs:
            lr = args.learning_rate * float(epoch + 1) / args.warmup_epochs

        # cosine decay
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            lr = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def train_contrastive(model, test_dataloader, train_dataloader, config, variational=False,
                      train_masked=False, test_masked=False, album=False, convex=False, start_epoch=0, views=2):
    # Training setup
    file_path = f".\\{config.save_path}\\Config.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(config, file_path)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # optimizer = SogCLR(
    #     model.parameters(),
    #     lr=config.learning_rate,
    #     weight_decay=config.weight_decay,
    #     mode='adamw',
    #     amsgrad=False,
    # )

    criterion = config.criterion
    model.to("cuda", config.dtype)

    torch.autograd.set_detect_anomaly(True)

    f = open(f".\\{config.save_path}\\Loss.txt", "w")
    f.close()

    if convex:
        convex_loss = ConvexCombinationLoss(num_augmentations=views-1)

    total_steps = config.num_epochs * len(train_dataloader)

    # Training loop
    step = 1
    for epoch in range(start_epoch, config.num_epochs):
        batch_steps = 0
        epoch_same_song_contrastive_loss = 0
        epoch_kld_loss = 0
        epoch_distribution_loss = 0
        epoch_convex_loss = 0
        epoch_same_album_contrastive_loss = 0

        batches = len(train_dataloader)

        set_schedulers(epoch, criterion, optimizer, config)

        for batch in tqdm(train_dataloader):
            indicies, inputs = batch

            # masking = get_mask_schedule(step, epoch, batch_steps, total_steps, config.num_epochs, batches)
            # model.mask_ratio = masking
            
            for index, view in enumerate(inputs):
                v = view.to("cuda", config.dtype).unsqueeze(1)
                inputs[index] = v
                B, _, T, F = v.shape

            stacked = torch.cat(inputs, dim=0)

            z_stacked = model(stacked)

            z_list = torch.split(z_stacked, B, dim=0)

            contrastive_loss = 0

            for index in range(1, len(z_list)):
                contrastive_loss += criterion(z_list[0], z_list[index])
            
            contrastive_loss = contrastive_loss / (len(z_list) - 1)
            loss = contrastive_loss
            
            if album:
                album_loss = criterion(za, zc)
                album_loss += criterion(zb, zc)
                album_loss = album_loss * 0.5
                epoch_same_album_contrastive_loss += album_loss.item()
                contrastive_loss = contrastive_loss + album_loss

            if convex:
                zl = torch.stack(z_list[1:], dim=1)

                convex_loss_score = convex_loss(zl, z_list[0])
                epoch_convex_loss += convex_loss_score.item()
                loss += convex_loss_score

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_same_song_contrastive_loss += contrastive_loss.item()

            step += 1
            batch_steps += 1

            term = f"Contrastive Loss [{batch_steps}/{batches}]: {contrastive_loss.item():.4f}"
            if variational:
                term += f"\t|\tKLD Loss [{batch_steps}/{batches}]: {kld_loss.item():.4f}"
                term += f"\t|\tDistribution Loss [{batch_steps}/{batches}]: {distribution_loss.item():.4f}"

            with open(f".\\{config.save_path}\\Loss.txt", "a") as f:
                term += "\n"
                f.write(term)

        same_song_contrastive_loss = evaluate_contrastive(model, test_dataloader, config, variational=variational, test_masked=test_masked, album=album)

        if album:
            same_song_contrastive_loss, same_album_contrastive_loss = same_song_contrastive_loss

        term = f"[Epoch {epoch}] Train: Same Song Contrastive Loss = {epoch_same_song_contrastive_loss / batch_steps:.4f}"

        if album:
            term += f"\t|\tSame Album Contrastive Loss = {epoch_same_album_contrastive_loss / batch_steps:.4f}"

        if convex:
            term += f"\t|\tConvex Loss = {epoch_convex_loss / batch_steps:.4f}"

        term += "\n"

        term += f"Test: Same Song Contrastive Loss = {same_song_contrastive_loss:.4f}"

        if album:
            term += f"\t|Same Album Contrastive Loss = {same_album_contrastive_loss:.4f}"

        # if convex:
        #     term += f"\t|\tConvex Loss = {epoch_convex_loss / batch_steps:.4f}"

        term += "\n"

        print(term)

        torch.save(model, f".\\{config.save_path}\\Epoch-{epoch}.pt")


def evaluate_contrastive(model, dataloader, config, variational=False, test_masked=False, album=False):
    song_contrastive_loss_total = 0
    album_contrastive_loss_total = 0

    kld_loss_total = 0

    criterion = config.criterion

    with torch.no_grad():
        for batch in tqdm(dataloader):
            indicies, inputs = batch

            for index, view in enumerate(inputs):
                v = view.to("cuda", config.dtype).unsqueeze(1)
                inputs[index] = v
                B, _, T, F = v.shape

            stacked = torch.cat(inputs, dim=0)

            z_stacked = model(stacked)

            z_list = torch.split(z_stacked, B, dim=0)

            if variational:
                za, mean_a, logvar_a = za
                zb, mean_b, logvar_b = zb
            
            contrastive_loss = 0
            for index in range(1, len(z_list)):
                contrastive_loss += criterion(z_list[0], z_list[index])
            
            contrastive_loss = contrastive_loss / (len(z_list) - 1)
            loss = contrastive_loss
            

            song_contrastive_loss_total += contrastive_loss.item()

    if variational:
        return song_contrastive_loss_total / len(dataloader), kld_loss_total / len(dataloader)
    if album:
        return song_contrastive_loss_total / len(dataloader), album_contrastive_loss_total / len(dataloader)

    return song_contrastive_loss_total / len(dataloader)