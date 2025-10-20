import os

from loss.loss_utils import *
from datasets import tqdm
from torch import optim

def get_mask_schedule(current_step, current_epoch, current_batch, total_steps, total_epochs, total_batches, max=0.9):
    if current_epoch > 1:
        return max

    return (current_batch / total_batches) * max

def train_contrastive(model, test_dataloader, train_dataloader, config, variational=False,
                      test_masked=False, album=False, convex=False, start_epoch=0, views=2):
    # Training setup
    file_path = f".\\{config.save_path}\\Config.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(config, file_path)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    criterion = config.criterion
    model.to("cuda", config.dtype)

    torch.autograd.set_detect_anomaly(True)
    
    if start_epoch == 0:
        f = open(f".\\{config.save_path}\\Loss.txt", "w")
        f.close()

    if convex:
        convex_loss = ConvexCombinationLoss(num_augmentations=views-1)


    # Training loop
    step = 1
    for epoch in range(start_epoch, config.num_epochs):
        batch_steps = 0
        epoch_same_song_contrastive_loss = 0
        epoch_convex_loss = 0

        batches = len(train_dataloader)

        for batch in tqdm(train_dataloader):
            indicies, inputs = batch
            
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

            with open(f".\\{config.save_path}\\Loss.txt", "a") as f:
                term += "\n"
                f.write(term)

        same_song_contrastive_loss = evaluate_contrastive(model, test_dataloader, config, variational=variational, test_masked=test_masked, album=album)

        term = f"[Epoch {epoch}] Train: Same Song Contrastive Loss = {epoch_same_song_contrastive_loss / batch_steps:.4f}"

        if convex:
            term += f"\t|\tConvex Loss = {epoch_convex_loss / batch_steps:.4f}"

        term += "\n"
        term += f"Test: Same Song Contrastive Loss = {same_song_contrastive_loss:.4f}"

        # if convex:
        #     term += f"\t|\tConvex Loss = {epoch_convex_loss / batch_steps:.4f}"

        term += "\n"

        print(term)

        torch.save(model, f".\\{config.save_path}\\Epoch-{epoch}.pt")


def evaluate_contrastive(model, dataloader, config, variational=False, test_masked=False, album=False):
    song_contrastive_loss_total = 0

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
            song_contrastive_loss_total += contrastive_loss.item()

    return song_contrastive_loss_total / len(dataloader)