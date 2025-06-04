import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NUM_ITEMS = 287
NUM_PLAYERS = 10
NUM_ITEMS_PER_CHAMPION = 7

class GarenBCPolicy(nn.Module):
    def __init__(self, 
                 num_champions: int = 50,
                 num_minions: int = 8,
                 num_towers: int = 6, 
                 num_players: int = 10, 
                 num_items: int = NUM_ITEMS,
                 emb_dim: int = 128,
                 seq_len: int = 9,
                 num_cont: int = 88,
                 inter_emb_dim: int = 512,
                 device: str = "cuda"
                 ):
        super().__init__()
        
        self.device = device
        self.emb_dim = emb_dim

        # ── Store various counts ────────────────────────────────────────
        self.num_champions = num_champions
        self.num_minions   = num_minions
        self.num_towers    = num_towers
        self.num_items     = num_items + 1               # reserve 1 for “no item” / unknown
        self.num_cont      = num_cont
        self.num_items_in_game = NUM_ITEMS_PER_CHAMPION * num_players
        self.seq_len = seq_len
        self.total_char_ids = num_champions + num_minions + num_towers + 1  # +1 for “Unknown”

        # ── Compute fixed obs_dim (once) ─────────────────────────────────
        #  obs_dim = [ num_cont ]
        #            + [ screen_pooled (emb_dim) ]
        #            + [ mini_pooled   (emb_dim) ]
        #            + [ items_flat   (num_items_in_game * (emb_dim//2)) ]
        self.obs_dim = (
            self.num_cont
            + self.emb_dim   # screen pooled
            + self.emb_dim   # minimap pooled
            + self.num_items_in_game * (self.emb_dim // 2)
        )

        # ── MLPs for per‐detection embeddings ────────────────────────────
        self.MLPScreen = nn.Sequential(
            nn.Linear(self.emb_dim + 4, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        )
        self.MLPMinimap = nn.Sequential(
            nn.Linear(self.emb_dim + 2, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        )

        # ── Embedding tables ─────────────────────────────────────────────
        self.char_embedding = nn.Embedding(
            num_embeddings=self.total_char_ids,
            embedding_dim=self.emb_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=self.emb_dim // 2
        )

        # ── Learned “empty” tokens for screen/minimap when no detections ──
        self.no_screen = nn.Parameter(torch.zeros(self.emb_dim), requires_grad=True)
        self.no_mini   = nn.Parameter(torch.zeros(self.emb_dim), requires_grad=True)

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
              continuous_feats: np.ndarray,
              screen_detections: torch.Tensor,
              minimap_detections: torch.Tensor,
              items: np.ndarray) -> torch.Tensor:
        """
        Build a fixed‐size observation embedding from:
          - continuous_feats:   np.ndarray shape (num_cont,)
          - screen_detections:  torch.Tensor shape (N_s, 5)   [id, x, y, hp, team]
          - minimap_detections: torch.Tensor shape (N_m, 3)   [id, x, y]
          - items:              np.ndarray    shape (num_items_in_game,)
        
        Returns:
          obs_vec: torch.Tensor shape (1, obs_dim)
        """
        device = self.char_embedding.weight.device
        emb_dim = self.emb_dim

        # ── 1) Embed + pool “screen” detections ───────────────────────────
        screen_list = []
        for i in range(screen_detections.shape[0]):
            char_id = int(screen_detections[i, 0])
            x, y, hp, team = (
                screen_detections[i, 1].item(),
                screen_detections[i, 2].item(),
                screen_detections[i, 3].item(),
                screen_detections[i, 4].item(),
            )
            # (a) char embedding
            char_emb = self.char_embedding(torch.tensor(char_id, device=device, dtype=torch.long))  # (emb_dim,)
            # (b) numeric features
            feat_vec = torch.tensor([x, y, hp, team], device=device, dtype=torch.float32)  # (4,)
            # (c) pass through MLPScreen → (emb_dim,)
            inp = torch.cat([char_emb, feat_vec], dim=-1).unsqueeze(0)  # (1, emb_dim+4)
            out = self.MLPScreen(inp).squeeze(0)                          # (emb_dim,)
            screen_list.append(out)

        if len(screen_list) > 0:
            screen_pooled = torch.mean(torch.stack(screen_list, dim=0), dim=0)  # (emb_dim,)
        else:
            screen_pooled = self.no_screen  # (emb_dim,)

        # ── 2) Embed + pool “minimap” detections ──────────────────────────
        mini_list = []
        for i in range(minimap_detections.shape[0]):
            char_id = int(minimap_detections[i, 0])
            x, y = (
                minimap_detections[i, 1].item(),
                minimap_detections[i, 2].item(),
            )
            char_emb = self.char_embedding(torch.tensor(char_id, device=device, dtype=torch.long))  # (emb_dim,)
            feat_vec = torch.tensor([x, y], device=device, dtype=torch.float32)  # (2,)
            inp = torch.cat([char_emb, feat_vec], dim=-1).unsqueeze(0)  # (1, emb_dim+2)
            out = self.MLPMinimap(inp).squeeze(0)                       # (emb_dim,)
            mini_list.append(out)

        if len(mini_list) > 0:
            mini_pooled = torch.mean(torch.stack(mini_list, dim=0), dim=0)  # (emb_dim,)
        else:
            mini_pooled = self.no_mini  # (emb_dim,)

        # ── 3) Embed “items” exactly as before ─────────────────────────────
        item_feats = torch.zeros((self.num_items_in_game, emb_dim // 2), device=device)
        for i, item_id in enumerate(items):
            idx = torch.tensor(item_id, dtype=torch.long, device=device)
            item_feats[i, :] = self.item_embedding(idx)  # (emb_dim//2,)
        item_flat = item_feats.view(1, -1)  # (1, num_items_in_game * (emb_dim//2))

        # ── 4) Continuous features ────────────────────────────────────────
        cont = torch.from_numpy(continuous_feats).unsqueeze(0).to(device)  # (1, num_cont)

        # ── 5) Concatenate into final obs_vec ─────────────────────────────
        #    [ cont (1×num_cont)
        #      | screen_pooled (1×emb_dim)
        #      | mini_pooled   (1×emb_dim)
        #      | item_flat     (1×(num_items_in_game*emb_dim//2)) ]
        screen_pooled = screen_pooled.unsqueeze(0)  # (1, emb_dim)
        mini_pooled   = mini_pooled.unsqueeze(0)    # (1, emb_dim)
        obs_vec = torch.cat([cont, screen_pooled, mini_pooled, item_flat], dim=1)  # (1, obs_dim)

        # Sanity check: ensure obs_vec width matches self.obs_dim
        assert obs_vec.shape[1] == self.obs_dim, (
            f"embed() returned width {obs_vec.shape[1]} but expected {self.obs_dim}"
        )

        return obs_vec  # (1, obs_dim)

    def forward(self, obs_sequence: list) -> dict:
        """
        obs_sequence: list of length=seq_len, each element is a dict:
          {
            "continuous_f":      np.ndarray shape (num_cont,),
            "screen_detections":  torch.Tensor shape (N_s, 5),
            "minimap_detections": torch.Tensor shape (N_m, 3),
            "items":              np.ndarray shape (num_items_in_game,)
          }
        
        Returns a dict of:
          {
            "move_logits":  tensor shape (1,),
            "attack_logit": tensor shape (1,),
            "xy_pred":      tensor shape (1,2),
            "abil_logits":  tensor shape (1,6),
          }
        """
        device = next(self.parameters()).device
        per_frame_embeddings = []

        # 1) For each frame in the sequence, embed → reduce → get (emb_dim,)
        for t in range(self.seq_len):
            frame_dict = obs_sequence[t]
            cont_feats   = frame_dict["continuous_f"]
            screen_dets  = frame_dict["screen_detections"]
            minimap_dets = frame_dict["minimap_detections"]
            items        = frame_dict["items"]

            # a) Build obs_vec (1, obs_dim)
            x_obs = self.embed(cont_feats, screen_dets, minimap_dets, items)

            # b) Reduce to (1, emb_dim) via mlp_reducer
            x_emb = self.mlp_reducer(x_obs)

            # c) Drop batch‐dim to get (emb_dim,) and append
            per_frame_embeddings.append(x_emb.squeeze(0))

        # 2) Stack into (seq_len, emb_dim), then unsqueeze to (1, seq_len, emb_dim)
        seq_tensor = torch.stack(per_frame_embeddings, dim=0).unsqueeze(0).to(device)

        # 3) Pass through LSTM → (1, seq_len, emb_dim)
        lstm_out, (h_n, c_n) = self.lstm(seq_tensor)

        # 4) Take last timestep’s feature → (1, emb_dim)
        last_feature = lstm_out[:, -1, :]

        # 5) Heads
        move_logits   = self.move_head(last_feature).squeeze(-1)       # (1,)
        attack_logit  = self.attack_head(last_feature).squeeze(-1)     # (1,)
        xy_pred       = torch.sigmoid(self.target_head(last_feature))  # (1,2)
        abil_logits   = self.ability_head(last_feature)                # (1,6)

        return {
            "move_logits":  move_logits,
            "attack_logit": attack_logit,
            "xy_pred":      xy_pred,
            "abil_logits":  abil_logits
        }

    def loss(self, outputs: dict, expert: dict) -> torch.Tensor:
        """
        Compute BC loss given model outputs and a single expert action.
        expert = {
          "move_dir": int in [0..24],
          "target":   0 or 1,
          "x": float or None,
          "y": float or None,
          "abilities": list of 6 ints (0/1)
        }
        """
        loss = torch.tensor(0.0, device=self.device)

        # – move_dir (cross‐entropy over 25 classes)
        move_t = torch.tensor([expert["move_dir"]], dtype=torch.long, device=self.device)  # (1,)
        move_logits = outputs["move_logits"]  # (1,)
        loss_move = F.cross_entropy(move_logits, move_t)
        loss = loss + 2 * loss_move

        # – attack_flag (BCEWithLogits)
        atk_t = torch.tensor([expert["target"]], dtype=torch.float32, device=self.device)  # (1,)
        atk_logit = outputs["attack_logit"]  # (1,)
        loss_atk = F.binary_cross_entropy_with_logits(atk_logit, atk_t)
        loss = loss + loss_atk

        # – target x,y only if attack==1
        if expert["target"] == 1:
            ex_xy = torch.tensor([[expert["x"], expert["y"]]], dtype=torch.float32, device=self.device)  # (1,2)
            loss_xy = F.mse_loss(outputs["xy_pred"], ex_xy)
        else:
            loss_xy = torch.tensor(0.0, device=self.device)
        loss = loss + loss_xy

        # – abilities (6‐binary BCEWithLogits)
        ex_abil = torch.tensor([expert["abilities"]], dtype=torch.float32, device=self.device)  # (1,6)
        loss_abil = F.binary_cross_entropy_with_logits(outputs["abil_logits"], ex_abil)
        loss = loss + loss_abil

        return loss
    
    def predict(self, state: list) -> dict:
        with torch.no_grad():
            return self.forward(state)
