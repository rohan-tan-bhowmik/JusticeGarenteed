import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ITEMS = 287
NUM_PLAYERS = 10
NUM_ITEMS_PER_CHAMPION = 7

class GarenBCPolicy(nn.Module):
    def __init__(self, 
                num_champions = 51,
                num_minions = 8,
                num_towers = 6, 
                num_players = 10, 
                num_items = NUM_ITEMS,
                emb_dim = 128,
                ):
        super().__init__()
        
        
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

        self.emb_dim = emb_dim
        self.num_champions = num_champions
        self.num_minions = num_minions
        self.num_towers = num_towers
        self.num_items = num_items + 1 

        self.num_items_in_game = NUM_ITEMS_PER_CHAMPION * num_players

        self.total_char_ids = num_champions + num_minions + num_towers + 1  # +1 for "Unknown" class

        self.char_embedding = nn.Embedding(num_embeddings = self.total_char_ids,
                                      embedding_dim = emb_dim)
        
        self.item_embedding = nn.Embedding(num_embeddings = num_items,
                                      embedding_dim = emb_dim//2)
        
        # self.backbone = nn.Sequential(
        #     nn.Linear(total_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU()
        #     # … or an LSTM, etc.
        # )

        self.move_head = nn.Linear(emb_dim, 1) # Output: degree to turn
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
        screen_flat = screen_out.view(1, -1)                        # (1, 66*emb_dim)

        mini_inp  = torch.cat([base_embeds, minimap_feats], dim=1) # (66, emb_dim+2)
        mini_out  = self.MLPMinimap(mini_inp)                      # (66, emb_dim)
        mini_flat = mini_out.view(1, -1)                           # (1, 66*emb_dim)

        cont = torch.from_numpy(continuous_feats).unsqueeze(0).to(device)  # (1, N_cont)

        # ── G) Concatenate: [screen_flat | mini_flat | cont | item_flat ] → shape (1, total_dim)
        obs_vec = torch.cat([cont, screen_flat, mini_flat, item_flat], dim = 1)  # (1, total_dim)

        return obs_vec
    
    def forward(self, obs_dict):
        cont_feats = obs_dict.get('continuous_f')
        screen_detections = obs_dict.get('screen_detections')
        minimap_detections = obs_dict.get('minimap_detections')
        items = obs_dict.get('items')

        x = self.embed(cont_feats, screen_detections, minimap_detections, items)
        
        return x

    def predict(self, state):
        with torch.no_grad():
            return self.forward(state)
        
