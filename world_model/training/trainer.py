import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import Batch, WorldModelDataset


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def save_model(
    model, losses,
    file_path,
    epoch
):
    os.makedirs(file_path, exist_ok=True)
    torch.save(
        {
            'model': model.state_dict(),
            'losses': losses
        },
        os.path.join(file_path, f'Epoches_{epoch}.pth')
    )


def train_one_epoch(
    denoiser,
    optimizer,
    dataloader,
    message,
    device
):
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"{message}")
    for ind, batch in enumerate(pbar, start=1):
        obs = batch['obs'].to(device)
        act = batch['act'].to(device)
        mask = batch['mask_padding'].to(device)

        optimizer.zero_grad()
        loss, logs = denoiser(Batch(obs=obs, act=act, mask_padding=mask))
        loss.backward()
        optimizer.step()

        total_loss += logs["loss_denoising"].item()

        avg_loss = total_loss / ind
        pbar.set_postfix({
            "avg_loss": f"{avg_loss:.4f}",
            "lr": optimizer.param_groups[0]["lr"]
        })
    return avg_loss



def train_world_model_full(
    denoiser,
    optimizer,
    scheduler,
    training_steps,
    dataset_path='data_collection',
    device=None,
    file_path='HK_diffusion_models',
    batch_size=8,
    context_len=4
):

    denoiser.train()
    losses = []

    epochs = 0
    epoch = 0
    for train_step in training_steps:
        epochs += train_step['epochs']

    print(f'Starting training model for {epochs}epochs with context_len = {context_len}')

    for train_step in training_steps:
        dataset = WorldModelDataset(dataset_path, context=context_len + train_step['seq_len'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(train_step['epochs']):
            epoch += 1

            loss = train_one_epoch(
                denoiser,
                optimizer,
                dataloader,
                f"|Epoch {epoch:>3}/{epochs:<3} | {train_step['name']:^30} | Sequence length {train_step['seq_len']:<3}|",
                device
            )

            losses.append(loss)
            save_model(denoiser, losses, file_path, epoch)

            scheduler.step()
