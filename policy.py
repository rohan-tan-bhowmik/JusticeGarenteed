import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ITEMS = 287
NUM_PLAYERS = 10
NUM_ITEMS_PER_CHAMPION = 7

class GarenBCPolicy(nn.Module):
    def __init__(self, 
                num_champions = 50,
                num_minions = 8,
                num_towers = 6, 
                num_players = 10, 
                num_items = NUM_ITEMS,
                emb_dim = 128,
                seq_len = 9,
                num_cont = 88,
                inter_emb_dim = 512,
                device = "cuda"
                ):
        super().__init__()
        
        self.device = device

        self.emb_dim = emb_dim
        self.num_champions = num_champions
        self.num_minions = num_minions
        self.num_towers = num_towers
        self.num_items = num_items + 1 
        self.num_cont = num_cont
        self.num_items_in_game = NUM_ITEMS_PER_CHAMPION * num_players

        self.seq_len = seq_len # number of steps to look back on to predict the next action
        self.total_char_ids = num_champions + num_minions + num_towers + 1  # +1 for "Unknown" class
        self.obs_dim = self.num_cont + self.total_char_ids * emb_dim * 2 + self.num_items_in_game * (emb_dim // 2)
        
        self.MLPScreen = nn.Sequential(
            nn.Linear(emb_dim + 4, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.MLPMinimap = nn.Sequential(
            nn.Linear(emb_dim + 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.char_embedding = nn.Embedding(num_embeddings = self.total_char_ids,
                                      embedding_dim = emb_dim)
        
        self.item_embedding = nn.Embedding(num_embeddings = num_items,
                                      embedding_dim = emb_dim//2)

        

        self.frame_embed_dim = inter_emb_dim
        self.mlp_reducer = nn.Sequential(
            nn.Linear(self.obs_dim, self.frame_embed_dim),
            nn.ReLU(),
            nn.Linear(self.frame_embed_dim, emb_dim),
            nn.ReLU()
        )

        # LSTM that reads a sequence of `emb_dim`‐vectors 
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim,
            num_layers=1,
            batch_first=True
        )

        self.move_head = nn.Linear(emb_dim, 25) # Output: degree to turn
        self.attack_head = nn.Linear(emb_dim, 1) # Output: whether to attack or not
        self.target_head = nn.Linear(emb_dim, 2) # Output: target x, target y
        self.ability_head = nn.Linear(emb_dim, 6) # Output: ability qwerdf

    def embed(self, continuous_feats, screen_detections, minimap_detections, items):
        """
        Embed detections into fixed-size vectors
        :param screen_detections: [batch_size, num_screen_detections, (id, x, y)]
        """

        device = self.char_embedding.weight.device
        T = self.total_char_ids
        screen_feats = torch.zeros((T, 4), device=device)

        # Embed screen detections
        for i in range(len(screen_detections)):
            char_id = int(screen_detections[i, 0])
            x = screen_detections[i, 1]
            y = screen_detections[i, 2]
            hp = screen_detections[i, 3]
            team = screen_detections[i, 4]
            screen_feats[char_id, :] = torch.tensor([x, y, hp, team], device=device)
        
        # Embed minimap detections
        minimap_feats = torch.zeros((T, 2), device=device)
        for i in range(len(minimap_detections)):
            char_id = int(minimap_detections[i, 0])
            x = minimap_detections[i, 1]
            y = minimap_detections[i, 2]
            minimap_feats[char_id, :] = torch.tensor([x, y], device=device)

        # Embed items

        item_feats = torch.zeros((self.num_items_in_game, self.emb_dim // 2), device=device)

        for i, item_id in enumerate(items):
            idx = torch.tensor(item_id, dtype=torch.long, device=device)
            emb = self.item_embedding(idx)            # shape: (emb_dim//2,)
            item_feats[i, :] = emb

        item_flat = item_feats.view(1, -1)  # (1, T * emb_dim // 2)

        all_ids = torch.arange(0, T, device=device, dtype=torch.long)  # shape (  66,)
        base_embeds = self.char_embedding(all_ids)  # → (66, emb_dim)
        screen_inp = torch.cat([base_embeds, screen_feats], dim=1) # shape (66, emb_dim + 4)
        screen_out = self.MLPScreen(screen_inp)                     # (66, emb_dim)
        # screen_flat = screen_out.view(1, -1)                        # (1, 66*emb_dim)

        mini_inp  = torch.cat([base_embeds, minimap_feats], dim=1) # (66, emb_dim+2)
        mini_out  = self.MLPMinimap(mini_inp)                      # (66, emb_dim)
        # mini_flat = mini_out.view(1, -1)                           # (1, 66*emb_dim)
        # concatenate the screen and minimap embeddings
        detection_emb = torch.cat([screen_out, mini_out], dim=1)  # (66, emb_dim * 2)
        detection_flat = detection_emb.view(1, -1)  # (1, 66 * emb_dim * 2)

        cont = torch.from_numpy(continuous_feats).unsqueeze(0).to(device)  # (1, N_cont)
        
        # ── G) Concatenate: [screen_flat | mini_flat | cont | item_flat ] → shape (1, total_dim)
        obs_vec = torch.cat([cont, detection_flat, item_flat], dim = 1)  # (1, total_dim)
        self.obs_dim = obs_vec.shape[1]
        return obs_vec
    
    def forward(self, obs_sequence):
        """
        :param: obs_sequence: Python list of length = seq_len.
        Each obs_sequence[t] is itself a dict with keys:
        - "continuous_f"       → np.ndarray of shape (N_cont,)
        - "screen_detections"   → np.ndarray of shape (N_s_t, 5)
        - "minimap_detections"  → np.ndarray of shape (N_m_t, 3)
        - "items"               → np.ndarray of shape (num_items_in_game,)
        """

        device = next(self.parameters()).device
        per_frame_embeddings = []

        # 1) For each frame t, extract its dict and embed→reduce
        for t in range(self.seq_len):
            frame_dict = obs_sequence[t]
            cont_feats        = frame_dict["continuous_f"]
            screen_dets       = frame_dict["screen_detections"]
            minimap_dets      = frame_dict["minimap_detections"]
            items             = frame_dict["items"]

            # a) Embed the raw state → (1, obs_dim)
            x_obs = self.embed(cont_feats, screen_dets, minimap_dets, items)  # (1, obs_dim)

            # b) Reduce to emb_dim via MLP
            x_emb = self.mlp_reducer(x_obs)  # (1, emb_dim)

            # c) Drop the leading batch‐dim so we have (emb_dim,)
            per_frame_embeddings.append(x_emb.squeeze(0))  # shape: (emb_dim,)

        # 2) Stack into (seq_len, emb_dim), then add batch‐dim → (1, seq_len, emb_dim)
        seq_tensor = torch.stack(per_frame_embeddings, dim=0).unsqueeze(0).to(device)

        # 3) Run through LSTM → (1, seq_len, emb_dim)
        lstm_out, (h_n, c_n) = self.lstm(seq_tensor)

        # 4) Take last time‐step’s output → (1, emb_dim)
        last_feature = lstm_out[:, -1, :]

        # 5) Action heads
        move_logits   = self.move_head(last_feature).squeeze(-1)    # → (1,)
        attack_logit  = self.attack_head(last_feature).squeeze(-1) # → (1,)
        xy_pred       = torch.sigmoid(self.target_head(last_feature))  # → (1, 2)
        abil_logits   = self.ability_head(last_feature)             # → (1, 6)

        return {
            "move_logits":  move_logits,
            "attack_logit": attack_logit,
            "xy_pred":      xy_pred,
            "abil_logits":  abil_logits
        }

    def loss(self, outputs, expert):
        loss = torch.tensor(0.0, device=self.device)

        # – move_dir (cross‐entropy over 25 classes)
        move_t = torch.tensor(
            [expert["move_dir"]], dtype=torch.long, device=self.device
        )  # shape (1,)
        move_logits = outputs["move_logits"] # ensure shape (1,25) if needed
        loss_move = F.cross_entropy(move_logits, move_t)
        loss = loss + 2 * loss_move

        # – attack_flag (BCEWithLogits)
        atk_t = torch.tensor([expert["target"]], dtype=torch.float32, device=self.device)
        atk_logit = outputs["attack_logit"]  
        loss_atk = F.binary_cross_entropy_with_logits(atk_logit, atk_t)
        loss = loss + loss_atk

        # – target x,y only if attack==1
        if expert["target"] == 1:
            ex_xy = torch.tensor(
                [[expert["x"], expert["y"]]], dtype=torch.float32, device=self.device
            )  # shape (1,2)
            loss_xy = F.mse_loss(outputs["xy_pred"], ex_xy)
        else:
            loss_xy = torch.tensor(0.0, device=self.device)

        loss = loss + loss_xy

        # – abilities (6-binary BCEWithLogits)
        ex_abil = torch.tensor(
            [expert["abilities"]], dtype=torch.float32, device=self.device
        )  # shape (1,6)
        loss_abil = F.binary_cross_entropy_with_logits(
            outputs["abil_logits"], ex_abil
        )
        loss = loss + loss_abil
        return loss
    
    def predict(self, state):
        with torch.no_grad():
            return self.forward(state)
        
