import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim import Adam, AdamW
from torch.utils.data import TensorDataset, random_split, DataLoader

from LolGarenEnv import GarenReplayEnv
from policy import GarenPolicy
from tqdm import tqdm

import argparse
import wandb 

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_val_bc(
    data_path: str,
    seq_len: int = 9,
    batch_size: int = 128,
    num_epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0, 
    device: str = "cuda", 
    checkpoint_dir: str = None, 
    resume_from: str = None,
    early_stop_epochs: int = 10,
    wandb_project: str = "garen_bc_training",
    wandb_run_name: str = "default_run", 
    logging_freq: int = 500,
    seed: int = 42
):
    """
    data_dir: directory containing many .pkl expert‐trajectory files.
    seq_len:  how many frames each training input uses (must match policy.seq_len).
    num_epochs: how many passes over all trajectories.
    """
    seed_everything(seed)
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "data_path": data_path,
            "lr": lr,
            "seq_len": seq_len,
            "num_epochs": num_epochs,
            "lr": lr,
            "device": device
        }
    )

    states, actions = torch.load(data_path)

    dataset = TensorDataset(states, actions)

    N = len(dataset)
    train_frac = 0.8
    train_size = int(train_frac * N)
    val_size   =  N - train_size

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=g
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory = True, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory = True)

    policy = GarenPolicy(device = device).to(device)
    policy.to(device)

    optimizer = AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    if resume_from is not None:
        # Load checkpoint before training
        ckpt = torch.load(resume_from, map_location=device)
        policy.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from checkpoint {resume_from}, starting at epoch {start_epoch+1}.")

    global_step = 0  # counts total frames seen across all epochs
    best_val_loss = float("inf")
    early_stop = 0 # counts epochs without improvement
    # training loop
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")
        # load data for this epoch
        policy.train()
        train_epoch_loss = 0.0
        count_steps = 0

        # train epoch
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} training", unit="batch"):
            states, expert_action = batch

            states = states.to(device)
            expert_action = expert_action.to(device)

            outputs = policy.forward(states) # (batch_size, action_dim)
            loss = policy.loss(outputs, expert_action)

            # 4) Backprop + step
            optimizer.zero_grad()
            loss.backward()

            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)

            optimizer.step()

            train_epoch_loss += loss.item()
            count_steps += 1
        
        if checkpoint_dir is not None:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch+1}.pth")

        # val epoch
        policy.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} validation", unit="batch"):
                states, expert_action = batch
                states = states.to(device)
                expert_action = expert_action.to(device)
                outputs = policy.forward(states)
                loss = policy.loss(outputs, expert_action)
                val_epoch_loss += loss.item()

        val_epoch_loss /= len(val_loader)

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            early_stop = 0
            print(f"New best validation loss: {best_val_loss:.6f}, saving model.")
            torch.save({
                "epoch": epoch,
                "model_state": policy.state_dict(),
                "optim_state": optimizer.state_dict()
            }, os.path.join(checkpoint_dir, "checkpoint_best.pth"))
        else:
            early_stop += 1
            

        avg_train_loss = train_epoch_loss / len(train_loader)
        print(f"Epoch {epoch} BC train loss = {avg_train_loss:.6f}")
        print(f"Epoch {epoch} validation loss: {val_epoch_loss:.6f}")

        wandb.log({
            "epoch_train_loss": avg_train_loss,
            "epoch_val_loss": val_epoch_loss,
            "epoch": epoch
        }, step = epoch)
        
        torch.save({
            "epoch": epoch,
            "model_state": policy.state_dict(),
            "optim_state": optimizer.state_dict()
        }, ckpt_path)

        print(f"Saved checkpoint: {ckpt_path}\n")

        if early_stop >= early_stop_epochs:
            print(f"Early stopping triggered after {early_stop_epochs} epochs without improvement.")
            break
           
    print("Training complete.")

def main():
    parser = argparse.ArgumentParser(description="Train a BC policy on Garen replay data.")
    parser.add_argument("--data_path", type=str, help="Path to .pt tensor dataset containing states and actions.")
    parser.add_argument("--seq_len", type=int, default=9, help="Length of input sequence for the policy.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer.")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="If > 0, clip gradient norm to this value.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run training on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--early_stop_epochs", type=int, default=10, help="Number of epochs without improvement before early stopping.")
    parser.add_argument("--wandb_project", type=str, default="garen_bc_training", help="WandB project name for logging.")
    parser.add_argument("--wandb_run_name", type=str, default="default_run", help="WandB run name for logging.")
    args = parser.parse_args()
    
    train_val_bc(
        data_path=args.data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=args.device, 
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        early_stop_epochs=args.early_stop_epochs,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )

if __name__ == "__main__":
    main()