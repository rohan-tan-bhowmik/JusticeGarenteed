import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from LolGarenEnv import GarenReplayEnv
from policy import GarenBCPolicy

import argparse

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
    device: str = "cuda"
):
    """
    data_dir: directory containing many .pkl expert‐trajectory files.
    seq_len:  how many frames each training input uses (must match policy.seq_len).
    num_epochs: how many passes over all trajectories.
    """

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
    policy = GarenBCPolicy().to(device)
    policy.to(device)

    optimizer = AdamW(policy.parameters(), lr=lr)

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

                epoch_loss += loss.item()
                count_steps += 1

                # 5) Move to next step
                obs_seq = next_obs_seq

        avg_loss = epoch_loss / max(1, count_steps)
        print(f"Epoch {epoch+1} done, avg BC loss per frame = {avg_loss:.6f}")

    print("Training complete.")

def main():
    parser = argparse.ArgumentParser(description="Train a BC policy on Garen replay data.")
    parser.add_argument("data_dir", type=str, help="Directory containing .pkl expert trajectory files.")
    parser.add_argument("--seq_len", type=int, default=9, help="Length of input sequence for the policy.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run training on (e.g., 'cuda' or 'cpu').")
    
    args = parser.parse_args()
    
    train_bc_on_directory(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device
    )