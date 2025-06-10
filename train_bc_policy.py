import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, random_split, DataLoader, DistributedSampler

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
    emb_dim: int = 128,
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

    policy = GarenPolicy(emb_dim=emb_dim, device = device).to(device)
    # print number of parameters

    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy has {num_params} trainable parameters.")

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
        train_movement_loss = 0.0
        train_attack_loss = 0.0
        train_target_loss = 0.0
        train_ability_loss = 0.0
        count_steps = 0

        # train epoch
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} training", unit="batch"):
            states, expert_action = batch

            states = states.to(device)
            expert_action = expert_action.to(device)

            outputs = policy.forward(states) # (batch_size, action_dim)
            loss = policy.loss(outputs, expert_action)
            movement_loss = loss[0]
            attack_loss = loss[1]
            target_loss = loss[2]
            ability_loss = loss[3]
            total_loss = loss.sum()  # total loss is the sum of all individual losses

            # 4) Backprop + step
            optimizer.zero_grad()
            total_loss.backward()

            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)

            optimizer.step()

            train_epoch_loss += total_loss.item()
            train_movement_loss += movement_loss.item()
            train_attack_loss += attack_loss.item()
            train_target_loss += target_loss.item()
            train_ability_loss += ability_loss.item()
            count_steps += 1
        
        if checkpoint_dir is not None:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch+1}.pth")

        # val epoch
        policy.eval()
        val_epoch_loss = 0.0
        val_movement_loss = 0.0
        val_attack_loss = 0.0
        val_target_loss = 0.0
        val_ability_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} validation", unit="batch"):
                states, expert_action = batch
                states = states.to(device)
                expert_action = expert_action.to(device)
                outputs = policy.forward(states)
                loss = policy.loss(outputs, expert_action)
                movement_loss = loss[0]
                attack_loss = loss[1]
                target_loss = loss[2]
                ability_loss = loss[3]
                total_loss = loss.sum()

                val_epoch_loss += total_loss.item()
                val_movement_loss += movement_loss.item()
                val_attack_loss += attack_loss.item()
                val_target_loss += target_loss.item()
                val_ability_loss += ability_loss.item()
                
        avg_train_loss = train_epoch_loss / len(train_loader)
        avg_train_movement_loss = train_movement_loss / len(train_loader)
        avg_train_attack_loss = train_attack_loss / len(train_loader)
        avg_train_target_loss = train_target_loss / len(train_loader)
        avg_train_ability_loss = train_ability_loss / len(train_loader)


        val_epoch_loss /= len(val_loader)
        val_movement_loss /= len(val_loader)
        val_attack_loss /= len(val_loader)
        val_target_loss /= len(val_loader)
        val_ability_loss /= len(val_loader)

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
            
        print(f"Epoch {epoch} BC train loss = {avg_train_loss:.6f}")
        print(f"Epoch {epoch} BC train movement loss = {avg_train_movement_loss:.6f}")
        print(f"Epoch {epoch} BC train attack loss = {avg_train_attack_loss:.6f}")
        print(f"Epoch {epoch} BC train target loss = {avg_train_target_loss:.6f}")
        print(f"Epoch {epoch} BC train ability loss = {avg_train_ability_loss:.6f}")
        print(f"Epoch {epoch} validation loss: {val_epoch_loss:.6f}")
        print(f"Epoch {epoch} validation movement loss: {val_movement_loss:.6f}")
        print(f"Epoch {epoch} validation attack loss: {val_attack_loss:.6f}")
        print(f"Epoch {epoch} validation target loss: {val_target_loss:.6f}")
        print(f"Epoch {epoch} validation ability loss: {val_ability_loss:.6f}")

        wandb.log({
            "epoch_train_loss": avg_train_loss,
            "epoch_train_movement_loss": avg_train_movement_loss,
            "epoch_train_attack_loss": avg_train_attack_loss,
            "epoch_train_target_loss": avg_train_target_loss,
            "epoch_train_ability_loss": avg_train_ability_loss,
            "epoch_val_loss": val_epoch_loss,
            "epoch_val_movement_loss": val_movement_loss,
            "epoch_val_attack_loss": val_attack_loss,
            "epoch_val_target_loss": val_target_loss,
            "epoch_val_ability_loss": val_ability_loss,
            "best_val_loss": best_val_loss,
            "epoch": epoch
        }, step = epoch)
        
        # torch.save({
        #     "epoch": epoch,
        #     "model_state": policy.state_dict(),
        #     "optim_state": optimizer.state_dict()
        # }, ckpt_path)

        # print(f"Saved checkpoint: {ckpt_path}\n")

        if early_stop >= early_stop_epochs:
            print(f"Early stopping triggered after {early_stop_epochs} epochs without improvement.")
            break
           
    print("Training complete.")

def train_val_bc_ddp(
    data_path: str,
    emb_dim: int = 128,
    seq_len: int = 9,
    batch_size: int = 128,
    num_epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    checkpoint_dir: str = None,
    resume_from: str = None,
    early_stop_epochs: int = 10,
    wandb_project: str = "garen_bc_training",
    wandb_run_name: str = "default_run",
    seed: int = 42
):
    """
    Distributed BC training loop for GarenPolicy.
    Launch this with torch.distributed.run / torchrun.
    """
    # 1) Initialize process group & set device
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 2) Set seeds
    seed_everything(seed)

    # 3) Only rank 0 logs to wandb
    if rank == 0:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "data_path": data_path,
                "lr": lr,
                "seq_len": seq_len,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "device": str(device),
                "world_size": world_size
            }
        )

    # 4) Load dataset
    states, actions = torch.load(data_path)
    dataset = TensorDataset(states, actions)

    # 5) Train/val split (reproducible across ranks)
    N = len(dataset)
    train_size = int(0.8 * N)
    val_size = N - train_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    # 6) Distributed samplers & loaders
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False, seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
    )

    # 7) Model, DDP wrapper, optimizer
    policy = GarenPolicy(emb_dim=emb_dim, device=device).to(device)
    policy = DDP(policy, device_ids=[local_rank], output_device=local_rank)
    optimizer = AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

    # 8) Optional checkpoint resume (only model & optimizer)
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device)
        policy.module.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if rank == 0:
            print(f"Resumed from {resume_from}; starting at epoch {start_epoch}")
    else:
        start_epoch = 0

    # 9) Prepare checkpoint directory
    if checkpoint_dir and rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")
    early_stop_counter = 0

    # 10) Training loop
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        policy.train()

        total_train_loss = 0.0
        steps = 0

        for states_batch, actions_batch in tqdm(
            train_loader,
            desc=f"[Rank {rank}] Epoch {epoch+1} train",
            disable=(rank != 0),
        ):
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)

            outputs = policy(states_batch)  # forward
            losses = policy.module.loss(outputs, actions_batch)
            total_loss = losses.sum()

            optimizer.zero_grad()
            total_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip)
            optimizer.step()

            total_train_loss += total_loss.item()
            steps += 1

        avg_train_loss = total_train_loss / max(1, steps)

        # 11) Validation
        val_sampler.set_epoch(epoch)
        policy.eval()

        total_val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for states_batch, actions_batch in tqdm(
                val_loader,
                desc=f"[Rank {rank}] Epoch {epoch+1} val",
                disable=(rank != 0),
            ):
                states_batch = states_batch.to(device)
                actions_batch = actions_batch.to(device)

                outputs = policy(states_batch)
                losses = policy.module.loss(outputs, actions_batch)
                total_val_loss += losses.sum().item()
                val_steps += 1

        avg_val_loss = total_val_loss / max(1, val_steps)

        # 12) Check for best & early stop (rank 0 only)
        if rank == 0:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state": policy.module.state_dict(),
                    "optim_state": optimizer.state_dict()
                }, os.path.join(checkpoint_dir, "checkpoint_best.pth"))
                print(f"[Epoch {epoch+1}] New best val loss: {best_val_loss:.6f}")
            else:
                early_stop_counter += 1
                print(f"[Epoch {epoch+1}] No improvement ({avg_val_loss:.6f}); early-stop {early_stop_counter}/{early_stop_epochs}")

            print(f"  Train loss: {avg_train_loss:.6f} | Val loss: {avg_val_loss:.6f}")

            # 13) Log to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
            })

        # 14) Early stopping across runs
        if early_stop_counter >= early_stop_epochs:
            if rank == 0:
                print(f"Early stopping triggered at epoch {epoch+1}")
            break

    if rank == 0:
        print("Training complete.")

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Train a BC policy on Garen replay data.")
    parser.add_argument("--data_path", type=str, help="Path to .pt tensor dataset containing states and actions.")
    parser.add_argument("--seq_len", type=int, default=9, help="Length of input sequence for the policy.")
    parser.add_argument("--emb_dim", type=int, default=128, help="Dimension of the policy embedding.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer.")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="If > 0, clip gradient norm to this value.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run training on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--early_stop_epochs", type=int, default=50, help="Number of epochs without improvement before early stopping.")
    parser.add_argument("--wandb_project", type=str, default="garen_bc_training", help="WandB project name for logging.")
    parser.add_argument("--wandb_run_name", type=str, default="default_run", help="WandB run name for logging.")
    parser.add_argument("--distributed", action='store_true', help="Use distributed training with DDP.")
    args = parser.parse_args()
    
    if args.distributed:
        # Launch with torch.distributed.run or torchrun
        train_val_bc_ddp(
            data_path=args.data_path,
            emb_dim=args.emb_dim,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_from,
            early_stop_epochs=args.early_stop_epochs,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name
        )
    else:
        train_val_bc(
            data_path=args.data_path,
            emb_dim=args.emb_dim,
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