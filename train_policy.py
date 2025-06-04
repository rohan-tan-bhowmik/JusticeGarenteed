import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from LolGarenEnv import GarenReplayEnv
from policy import GarenBCPolicy
from tqdm import tqdm

import argparse
import wandb 

# ── Assumes you already have:
#    - GarenReplayEnv  (as defined above)
#    - GarenBCPolicy   (the LSTM‐backboned policy from earlier)
#    - ITEMS (imported from create_expert_trajectories)
#
#    and that each pickled trajectory is a list of:
#       [ {"state": state_dict, "action": action_dict}, … ].
#
#    Also assume your `state_dict` → obs_dict conversion is already embedded
#    in `GarenReplayEnv._dict_to_observation`, so you do not need to call a separate helper.

def train_bc_on_directory(
    data_dir: str,
    seq_len: int = 9,
    num_epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cuda", 
    checkpoint_dir: str = None, 
    resume_from: str = None,
    wandb_project: str = "garen_bc_training",
    wandb_run_name: str = "default_run", 
    logging_freq: int = 500
):
    """
    data_dir: directory containing many .pkl expert‐trajectory files.
    seq_len:  how many frames each training input uses (must match policy.seq_len).
    num_epochs: how many passes over all trajectories.
    """
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "data_dir": data_dir,
            "lr": lr,
            "seq_len": seq_len,
            "num_epochs": num_epochs,
            "lr": lr,
            "device": device
        }
    )
    # 1) Gather all .pkl files
    traj_files = [
        os.path.join(data_dir, fn)
        for fn in os.listdir(data_dir)
        if fn.endswith(".pkl")
    ]
    
    assert len(traj_files) > 0, f"No .pkl files found in {data_dir}"
    
    # 2) Instantiate the environment and a dummy obs to get obs_dim
    example_env = GarenReplayEnv(traj_files[0], seq_len=seq_len)
    # We can grab one obs_seq via reset() to see what "frame" looks like.
    # But note: reset() now returns a list of length=seq_len of obs_dicts.
    obs_seq_example = example_env.reset()  # a list of 9 obs_dicts
    # Each obs_dict has keys "continuous_f", "screen_detections", etc.
    # We need to know obs_dim to create our policy’s `mlp_reducer`. 
    # Our policy’s constructor already knows how to handle embedding, so we just trust it.

    # 3) Instantiate policy
    policy = GarenBCPolicy(device = device).to(device)
    policy.to(device)

    optimizer = AdamW(policy.parameters(), lr=lr)

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

    # 4) Main training loop
    for epoch in range(num_epochs):
        random.shuffle(traj_files)
        epoch_loss = 0.0
        count_steps = 0

        print(f"Epoch {epoch+1}/{num_epochs}  (trajectories: {len(traj_files)})")

        for traj_path in traj_files:
            # Create a fresh env for each trajectory so we start at idx=0
            env = GarenReplayEnv(traj_path, seq_len=seq_len)
            obs_seq = env.reset()  # list of length=seq_len of obs_dicts
            done = False

            # Step through the entire trajectory
            while not done:
                # 1) Forward pass through the policy
                outputs = policy(obs_seq)
                #   outputs["move_logits"]  shape (1,)
                #   outputs["attack_logit"] shape (1,)
                #   outputs["xy_pred"]      shape (1,2)
                #   outputs["abil_logits"]  shape (1,6)

                # 2) Advance the env; get next obs_seq and expert_action
                next_obs_seq, _, done, info = env.step(None)
                expert = info["expert_action"]
                # expert is a dict:
                #   {
                #     "move_dir": int∈[0..24],
                #     "target":   0 or 1,
                #     "x": float in [0,1] (or None if target=0),
                #     "y": float in [0,1] (or None),
                #     "abilities": list of 6 ints (0/1)
                #   }

                # 3) Compute BC losses
                loss = policy.loss(outputs, expert)

                # 4) Backprop + step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                count_steps += 1
                global_step += 1

                # 5.6) Per‐step logging to WandB
                if global_step % logging_freq == 0:
                    wandb.log({
                        "step_loss": loss_val,
                        "global_step": global_step,
                        "epoch": epoch + 1
                    }, step=global_step)

                # 5) Move to next step
                obs_seq = next_obs_seq

        avg_loss = epoch_loss / max(1, count_steps)
        print(f"Epoch {epoch+1} done, avg BC loss per frame = {avg_loss:.6f}")

        wandb.log({
            "epoch_avg_loss": avg_loss,
            "epoch": epoch + 1
        }, step=(epoch + 1))

        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state": policy.state_dict(),
            "optim_state": optimizer.state_dict()
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}\n")

    print("Training complete.")

def main():
    parser = argparse.ArgumentParser(description="Train a BC policy on Garen replay data.")
    parser.add_argument("--data_dir", type=str, help="Directory containing .pkl expert trajectory files.")
    parser.add_argument("--seq_len", type=int, default=9, help="Length of input sequence for the policy.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run training on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--wandb_project", type=str, default="garen_bc_training", help="WandB project name for logging.")
    parser.add_argument("--wandb_run_name", type=str, default="default_run", help="WandB run name for logging.")
    args = parser.parse_args()
    
    train_bc_on_directory(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device, 
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )

if __name__ == "__main__":
    main()