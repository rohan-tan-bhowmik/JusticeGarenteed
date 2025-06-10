##PPO Policy###

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# replay environment from expert trajectories
from LolGarenEnv import GarenReplayEnv
from policy import GarenPolicy

# device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOAgent(nn.Module):
    def __init__(self):
        super().__init__()
        # use BC policy as feature extractor
        self.backbone = GarenPolicy(device=str(DEVICE))
        # value head for state values
        self.value_head = nn.Linear(self.backbone.emb_dim, 1)

    def forward(self, obs_seq):
        # obs_seq: [B, T, obs_dim]
        feats = self.backbone.mlp_reducer(self.backbone.embed(obs_seq))  # [B, T, emb]
        last = feats[:, -1, :]  # [B, emb]
        # action logits
        move   = self.backbone.move_head(last)
        attack = self.backbone.attack_head(last)
        xy     = torch.sigmoid(self.backbone.target_head(last))
        abil   = self.backbone.ability_head(last)
        logits = torch.cat([move, attack, xy, abil], dim=-1)
        # state value
        value = self.value_head(last).squeeze(-1)
        return logits, value

# ppo training, minimalist comments

def compute_gae(rewards, values, dones, gamma, lam):
    gae = 0
    returns = []
    values = values + [0]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])
    return returns


def train(args):
    env = GarenReplayEnv(args.traj_path, seq_len=args.seq_len)
    agent = PPOAgent().to(DEVICE)
    opt = optim.Adam(agent.parameters(), lr=args.lr)

    # storage
    buffer = {
        'obs': [], 'acts': [], 'logps': [],
        'rews': [], 'vals': [], 'dones': []
    }

    # init
    obs = env.reset()
    seq = [obs for _ in range(args.seq_len)]
    step = 0

    while step < args.total_steps:
        for _ in range(args.rollout_length):
            obs_tensor = torch.tensor(np.stack(seq), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits, val = agent(obs_tensor)
                dist = Categorical(logits=logits)
                act = dist.sample()[0]
                logp = dist.log_prob(act)[0]

            nxt, rew, done, _ = env.step(act.item())
            # store
            buffer['obs'].append(obs_tensor)
            buffer['acts'].append(act)
            buffer['logps'].append(logp)
            buffer['rews'].append(rew)
            buffer['vals'].append(val.item())
            buffer['dones'].append(done)

            step += 1
            seq.pop(0)
            seq.append(nxt)
            if done:
                nxt = env.reset()
                seq = [nxt for _ in range(args.seq_len)]

        # compute returns and advantages
        rets = compute_gae(buffer['rews'], buffer['vals'], buffer['dones'], args.gamma, args.lam)
        advs = [r - v for r, v in zip(rets, buffer['vals'])]

        # convert to tensors
        obs_batch = torch.cat(buffer['obs'])
        act_batch = torch.stack(buffer['acts'])
        old_logp = torch.stack(buffer['logps'])
        ret_batch = torch.tensor(rets, dtype=torch.float32, device=DEVICE)
        adv_batch = torch.tensor(advs, dtype=torch.float32, device=DEVICE)

        # ppo update
        for _ in range(args.ppo_epochs):
            logits, vals = agent(obs_batch)
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(act_batch)
            ratio = (new_logp - old_logp).exp()
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(vals, ret_batch)
            entropy = dist.entropy().mean()
            loss = policy_loss + args.vf_coeff * value_loss - args.ent_coeff * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.grad_norm)
            opt.step()

        # clear buffer
        for k in buffer: buffer[k].clear()

    print("training done")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--traj-path', type=str, required=True)
    p.add_argument('--seq-len', type=int, default=9)
    p.add_argument('--total-steps', type=int, default=200000)
    p.add_argument('--rollout-length', type=int, default=2048)
    p.add_argument('--ppo-epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--clip-eps', type=float, default=0.2)
    p.add_argument('--vf-coeff', type=float, default=0.5)
    p.add_argument('--ent-coeff', type=float, default=0.01)
    p.add_argument('--grad-norm', type=float, default=0.5)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lam', type=float, default=0.95)
    args = p.parse_args()
    train(args)