import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from create_expert_trajectories import \
    MAX_NUM_CHAMPION_DETECTIONS, MAX_NUM_MINION_DETECTIONS,  \
    MAX_NUM_TOWER_DETECTIONS, NUM_CONTINUOUS_F, MAX_NUM_DETECTIONS

NUM_ITEMS = 287
NUM_PLAYERS = 10
NUM_ITEMS_PER_CHAMPION = 7

from classes import CONDENSED_CHAMPIONS_TO_I, CONDENSED_CHAMPIONS
NUM_CHAMPION_CLASSES = len(CONDENSED_CHAMPIONS_TO_I)
NUM_MINION_CLASSES = 8
NUM_TOWER_CLASSES = 6

class GarenPolicy(nn.Module):
    def __init__(self, 
                 max_champion_detections: int = MAX_NUM_CHAMPION_DETECTIONS,
                 max_minion_detections: int = MAX_NUM_MINION_DETECTIONS,
                 max_tower_detections: int = MAX_NUM_TOWER_DETECTIONS, 
                 num_items: int = NUM_ITEMS,
                 num_players: int = NUM_PLAYERS,
                 emb_dim: int = 128,
                 seq_len: int = 9,
                 num_cont: int = NUM_CONTINUOUS_F,
                 inter_emb_dim: int = 512,
                 device: str = "cuda"
                 ):
        super().__init__()
        
        self.device = device
        self.emb_dim = emb_dim

        # ── Store various counts ────────────────────────────────────────
        self.num_champions = max_champion_detections
        self.num_minions   = max_minion_detections
        self.num_towers    = max_tower_detections
        self.num_items     = num_items + 1               # reserve 1 for “no item” / unknown
        self.num_cont      = num_cont
        self.num_items_in_game = NUM_ITEMS_PER_CHAMPION * num_players
        self.seq_len = seq_len
        self.champ_vector_dim = NUM_CHAMPION_CLASSES * (4 + self.emb_dim) # 50 * (emb_dim + x, y, hp, team_color)
        self.minion_vector_dim = NUM_MINION_CLASSES * (2 + self.emb_dim // 2) # 8 * (emb_dim // 2 + x, y)
        self.tower_vector_dim = NUM_TOWER_CLASSES * (2 + self.emb_dim)   # 6 * (emb_dim // 2 + x, y)

        # ── Compute fixed obs_dim (once) ─────────────────────────────────
        #  obs_dim = [ num_cont ]
        #            + [ screen_pooled (emb_dim) ]
        #            + [ mini_pooled   (emb_dim) ]
        #            + [ items_flat   (num_items_in_game * (emb_dim//2)) ]
        self.obs_dim = (
            self.num_cont
            + self.emb_dim  # champion detections
            + self.emb_dim // 2 # minions
            + self.emb_dim // 2 # towers
            + self.emb_dim // 2 # items
        )

        self.MLPChamp = nn.Sequential(
            nn.Linear(self.emb_dim + 4, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        )

        self.MLPMinion = nn.Sequential(
            nn.Linear(self.emb_dim // 2 + 4, self.emb_dim // 2),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 2, self.emb_dim // 2)
        )

        self.MLPTower = nn.Sequential(
            nn.Linear(self.emb_dim // 2 + 4, self.emb_dim // 2),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 2 , self.emb_dim // 2)
        )

        self.MLPItem = nn.Sequential(
            nn.Linear(self.emb_dim // 2, self.emb_dim // 2),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 2, self.emb_dim // 2)
        )

        # ── Embedding tables for champion, minion, tower and 
        self.champ_embedding = nn.Embedding(
            num_embeddings=NUM_CHAMPION_CLASSES + 1,  # +1 for "no champion" / unknown
            embedding_dim=self.emb_dim
        )

        self.minion_embedding = nn.Embedding(
            num_embeddings=NUM_MINION_CLASSES + 1,  # +1 for "no minion" / unknown
            embedding_dim=self.emb_dim // 2
        )

        self.tower_embedding = nn.Embedding(
            num_embeddings=NUM_TOWER_CLASSES + 1,  # +1 for "no tower" / unknown
            embedding_dim=self.emb_dim // 2
        )

        self.item_embedding = nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.emb_dim // 2
        )

        # ── Reducer: from obs_dim → emb_dim (through inter_emb_dim) ───────
        self.frame_embed_dim = inter_emb_dim

        self.mlp_reducer = nn.Sequential(
            nn.Linear(self.obs_dim, self.frame_embed_dim),
            nn.ReLU(),
            nn.Linear(self.frame_embed_dim, self.emb_dim),
            nn.ReLU()
        )

        # ── LSTM over time sequence of length `seq_len` ─────────────────
        self.lstm = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.emb_dim,
            num_layers=1,
            batch_first=True
        )

        # ── Action heads (after LSTM) ────────────────────────────────────
        self.move_head    = nn.Linear(self.emb_dim, 25)  # 25 discrete move directions
        self.attack_head  = nn.Linear(self.emb_dim, 1)   # binary attack flag
        self.target_head  = nn.Linear(self.emb_dim, 2)   # (x, y) continuous target
        self.ability_head = nn.Linear(self.emb_dim, 6)   # 6 binary ability flags

    def embed(self,
              x: torch.Tensor,
              ) -> torch.Tensor:
        """
        :param x: torch.Tensor shape (B, T, obs_dim)
        Each tensor in each sequence is of shape (obs_dim), structured like so:
        [num_continuous_f  | champion_detections padded | minion_detections padded | tower_detections padded | items_flattened]
        
        Returns:
          embedding vector of shape (B, emb_vector_dim)
        """
        cont_f = x[:, :, :NUM_CONTINUOUS_F]  # (B, T, num_continuous_f)
        detections = x[:, :, NUM_CONTINUOUS_F:NUM_CONTINUOUS_F + MAX_NUM_DETECTIONS * 5]  # (B, T, 5 * MAX_NUM_DETECTIONS)

        champ_detections = detections[:, :, :MAX_NUM_CHAMPION_DETECTIONS * 5]  # (B, T, 5 * MAX_NUM_CHAMPION_DETECTIONS)
        minion_detections = detections[:, :, MAX_NUM_CHAMPION_DETECTIONS * 5:5 * (MAX_NUM_CHAMPION_DETECTIONS + MAX_NUM_MINION_DETECTIONS)]  # (B, T, 5 * MAX_NUM_MINION_DETECTIONS)
        tower_detections = detections[:, :, 5 * (MAX_NUM_CHAMPION_DETECTIONS + MAX_NUM_MINION_DETECTIONS):
                                        5 * (MAX_NUM_CHAMPION_DETECTIONS + MAX_NUM_MINION_DETECTIONS + MAX_NUM_TOWER_DETECTIONS)]  # (B, T, 5 * MAX_NUM_TOWER_DETECTIONS)

        B, T, _ = detections.shape
        champ_detections = champ_detections.reshape(B, T, MAX_NUM_CHAMPION_DETECTIONS, 5)  # (B, T, N_champions, 5)
        ids_c = champ_detections[:, :, :, 0].long().unsqueeze(-1)  # (B, T, N_champions, 1)
        champ_other = champ_detections[:, :, :, 1:]  # (B, T, N_champions, 4)

        minion_detections = minion_detections.reshape(B, T, MAX_NUM_MINION_DETECTIONS, 5)  # (B, T, N_minions, 5)
        ids_m = minion_detections[:, :, :, 0].long().unsqueeze(-1)  # (B, T, N_minions, 1)
        minion_other = minion_detections[:, :, :, 1:]  # (B, T, N_minions, 4)

        tower_detections = tower_detections.reshape(B, T, MAX_NUM_TOWER_DETECTIONS, 5)  # (B, T, N_towers, 5)
        ids_t = tower_detections[:, :, :, 0].long().unsqueeze(-1)  # (B, T, N_towers, 1)
        tower_other = tower_detections[:, :, :, 1:]  # (B, T, N_towers, 4)

        # 3) shift IDs down to [0..num_classes_per_type) so they index into each table
        #    we clamp negatives back to zero but they won’t be used (masked out anyway)
        ids_c = ids_c.clamp(min=0, max=NUM_CHAMPION_CLASSES-1)
        ids_m = (ids_m - NUM_CHAMPION_CLASSES).clamp(min=0, max=NUM_MINION_CLASSES-1)
        ids_t = (ids_t - NUM_CHAMPION_CLASSES - NUM_MINION_CLASSES).clamp(min=0, max=NUM_TOWER_CLASSES-1)

        # 4) Embed each type of detection
        emb_champ = self.champ_embedding(ids_c).squeeze(-2)  # (B, T, N_champions, emb_dim)
        emb_minion = self.minion_embedding(ids_m).squeeze(-2)  # (B, T, N_minions, emb_dim // 2)
        emb_tower = self.tower_embedding(ids_t).squeeze(-2)  # (B, T, N_detections, emb_dim // 2)

        champ_features = torch.cat([emb_champ, champ_other], dim=-1)  # (B, T, N_champions, emb_dim + 4)
        minion_features = torch.cat([emb_minion, minion_other], dim=-1)  # (B, T, N_minions, emb_dim // 2 + 4)
        tower_features = torch.cat([emb_tower, tower_other], dim=-1)  # (B, T, N_towers, emb_dim // 2 + 4)

        # 5) Apply MLPs to each type of detection
        emb_champ = self.MLPChamp(champ_features)  # (B, T, N_champions, emb_dim)
        emb_minion = self.MLPMinion(minion_features)  # (B, T, N_minions, emb_dim // 2)
        emb_tower = self.MLPTower(tower_features)  # (B, T, N_towers, emb_dim // 2)

        # 6) Maxpool over num_detections dimension
        pooled_champ = emb_champ.max(dim=2).values  # (B, T, emb_dim)
        pooled_minion = emb_minion.max(dim=2).values  # (B, T, emb_dim // 2)
        pooled_tower = emb_tower.max(dim=2).values  # (B, T, emb_dim // 2)
        
        # 7) Embed items
        items = x[:, :, NUM_CONTINUOUS_F + MAX_NUM_DETECTIONS * 5:]  # (B, T, num_items_in_game)
        items_emb = self.item_embedding(items.long())  # (B, T, num_items_in_game, emb_dim // 2)
        items_emb = self.MLPItem(items_emb)  # (B, T, num_items_in_game, emb_dim // 2)
        # Maxpool over items dimension
        items_emb = items_emb.max(dim=2).values  # (B, T, emb_dim // 2)

        # 8) Concatenate all embeddings
        x_obs = torch.cat([
            cont_f,
            pooled_champ,
            pooled_minion,
            pooled_tower,
            items_emb
        ], dim=-1)  # (B, T, obs_dim)

        return x_obs  # (B, T, obs_dim)

    def forward(self, obs_sequence: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the policy model.

        :param obs_sequence: torch.Tensor of shape (B, seq_len, obs_dim) 

        Returns:
            (B, action_dim) torch.Tensor with action logits.
        """
        
        # 1) Embed the sequence of observations
        seq_tensor = self.embed(obs_sequence)  # (B, seq_len, obs_dim)

        # 2) Reduce obs_dim to emb_dim
        seq_tensor = self.mlp_reducer(seq_tensor)  # (B, seq_len, emb_dim)

        # 3) Pass through LSTM → (B, seq_len, emb_dim)
        lstm_out, (h_n, c_n) = self.lstm(seq_tensor)

        # 4) Take last timestep’s feature → (B, emb_dim)
        last_feature = lstm_out[:, -1, :]

        # 5) Heads
        move_logits   = self.move_head(last_feature)                # (B, 25)
        attack_logit  = self.attack_head(last_feature)  # (B, 1)
        xy_pred       = torch.sigmoid(self.target_head(last_feature))  # (B, 2)
        abil_logits   = self.ability_head(last_feature)                # (B, 6)


        return torch.cat([
            move_logits,    # → (B,25)
            attack_logit,   # → (B,1)
            xy_pred,        # → (B,2)
            abil_logits     # → (B,6)
        ], dim=1)          # ← concat on last dim
        
    def loss(self, outputs: dict, expert: dict) -> torch.Tensor:
        """
        Compute BC loss given model outputs and expert actions
        :param outputs:  (B, 34) tensor of [move_logits(25), attack_logit(1), xy_pred(2), abil_logits(6)]
        :param expert: (B, 10) tensor of [move_dir(0-24), target(0/1), x(0-1), y(0-1), abilities(6-binary)]
        :return: loss value as a single tensor
        """

        # 1) Extract logits from outputs
        move_logits = outputs[:, :25]  # (B, 25)
        attack_logits = outputs[:, 25:26]  # (B, 1)
        xy_pred = outputs[:, 26:28]  # (B, 2)
        abil_logits = outputs[:, 28:]  # (B, 6)

        # 2) Extract expert actions
        move_dir = expert[:, 0]  # (B,)
        target = expert[:, 1]  # (B,)
        xy_target = expert[:, 2:4]  # (B,)
        abilities = expert[:, 4:]  # (B, 6)

        # 3) Compute losses
        # weight movement more
        loss_move = 2 * F.cross_entropy(move_logits, move_dir.long(), reduction='mean')  # (B,)

        loss_attack = F.binary_cross_entropy_with_logits(attack_logits, target.float().unsqueeze(-1), reduction='mean')
        per_sample_mse = F.mse_loss(xy_pred, xy_target, reduction='none')  # (B,2)
        per_sample_mse = per_sample_mse.mean(dim=-1)                       # (B,)

        # mask out non-attack samples
        mask = (target == 1).float()                                      # (B,)
        valid_count = mask.sum().clamp(min=1)                             # scalar

        loss_xy = (per_sample_mse * mask).sum() / valid_count             # scalar

        loss_abil = F.binary_cross_entropy_with_logits(abil_logits, abilities.float(), reduction='mean')

        return loss_move + loss_attack + loss_xy + loss_abil  # scalar tensor
    
    def predict(self, state: list) -> dict:
        with torch.no_grad():
            logits = self.forward(state)
            move_dir = logits[:, :25].argmax(dim=1)  # (B,)
            attack = (logits[:, 25:26] > 0.5).float()
            xy_target = torch.sigmoid(logits[:, 26:28])
            abilities = (logits[:, 28:] > 0.5).float()
            return {
                "move_dir": move_dir,
                "attack": attack,
                "xy_target": xy_target,
                "abilities": abilities
            }
