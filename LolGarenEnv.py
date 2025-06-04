import gymnasium as gym
from collections import deque
import numpy as np
import time
import math
from collections import deque
import csv
import json
import pickle

from create_expert_trajectories import ITEMS

class GarenReplayEnv(gym.Env):
    """
    Gym-style Env that replays a list of (state, action) pairs.
    Each call to step() advances the pointer and returns the stored state and action.
    """
    metadata = {"render.modes": []}

    def __init__(self, trajectory_path, seq_len = 9):
        super().__init__()
        # Load the list of (state, action) pairs from pickle
        with open(trajectory_path, "rb") as f:
            self.trajectory = pickle.load(f)

        # Each element in trajectory is a dict with 'state' and 'action' keys
        self.n_steps = len(self.trajectory)
        self._idx = 0
        self.seq_len = seq_len
        self._buffer = deque(maxlen=seq_len)

        # Convert the first state to an observation to define observation_space
        first_state = self.trajectory[0]['state']
        obs = self._dict_to_observation(first_state)
        obs_dim = obs.shape[0]  # length of flattened observation vector
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Define action_space just to match signature; agent actions are ignored in replay
        self.action_space = gym.spaces.Dict({
            "move_dir": gym.spaces.Discrete(25),  # 0-23 headings + 24 idle
            "target": gym.spaces.Discrete(2),     # 0=no attack, 1=attack
            "x": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "y": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "abilities": gym.spaces.MultiBinary(6)  # six abilities flags
        })

    def reset(self):
        self._idx = 0
        # Clear & refill the buffer with the first obs repeated (or zeros)
        self._buffer.clear()
        first_state = self.trajectory[0]["state"]
        first_obs = self._dict_to_observation(first_state)
        for _ in range(self.seq_len):
            # pad with the first frame (or you could pad with zeros)
            self._buffer.append(first_obs.copy())
        # Return a (seq_len, obs_dim) array
        return np.stack(self._buffer, axis=0)

    def step(self, agent_action):
        # Get the expert action for the current index
        record = self.trajectory[self._idx]
        expert_action = record["action"]

        # Advance index
        self._idx += 1
        done = False
        if self._idx >= self.n_steps:
            done = True
            # When done, just return a zero‐sequence 
            zero_seq = np.zeros((self.seq_len, self.obs_dim), dtype=np.float32)
            return zero_seq, 0.0, True, {"expert_action": expert_action}

        # Build the next “single frame” obs
        next_state = self.trajectory[self._idx]["state"]
        next_obs = self._dict_to_observation(next_state)

        # Push into buffer, pop oldest automatically
        self._buffer.append(next_obs)

        # Stack into (seq_len, obs_dim) and return
        seq_obs = np.stack(self._buffer, axis=0)
        return seq_obs, 0.0, done, {"expert_action": expert_action}

    def _dict_to_observation(self, step_dict):
        """
        Convert the feature dict into a flat numpy array of floats.
        Convert numeric strings to floats; extract only chosen features here.

        Returns: flat array of features, minimap detections, champion detections, and minion detections.
        """

        frame = step_dict["frame"]
        x, y = step_dict.get("garen_pos", 0) # garen's position on the screen
        mini_x, mini_y = step_dict.get("minimap_x_ratio", 0.0), step_dict.get("minimap_y_ratio", 0) # garen's position on minimap
        move_dir = step_dict.get("move_dir", 0)
        xp_bar = step_dict.get("xp-bar", 0.0)
        health_bar = step_dict.get("health-bar")
        attack_dmg = step_dict.get("attack-damage", 0.0)
        armor = step_dict.get("armor", 0.0)
        magic_resist = step_dict.get("magic-resist", 0.0)
        move_speed = step_dict.get("move-speed", 0.0)
        q_cd = step_dict.get("q-cd", 0.0)
        w_cd = step_dict.get("w-cd", 0.0)
        e_cd = step_dict.get("e-cd", 0.0)
        r_cd = step_dict.get("r-cd", 0.0)
        d_cd = step_dict.get("d-cd", 0.0)
        f_cd = step_dict.get("f-cd", 0.0)
        lanes = ["b1, b2, b3, b4, b5, r1, r2, r3, r4, r5"]
        objective_names = ["towers", "grubs", "heralds-barons", "dragons", "kills"]
        kdas, cs, health_levels, levels, objectives, items = [], [], [], [], [], []
        for lane in lanes:
            kda = step_dict[f"{lane}-kda"]
            kdas.extend(kda)
            cs.append(step_dict.get(f"{lane}-cs", 0.0))
            health_levels.append(step_dict.get(f"{lane}-health", 0.0))
            levels.append(step_dict.get(f"{lane}-level", 0.0))
        for objective in objective_names:
            objectives.append(step_dict.get(f"b-{objective}", 0.0))
            objectives.append(step_dict.get(f"r-{objective}", 0.0))

        for item in ITEMS:
            items.append(step_dict.get(item, len(ITEMS)))  # Use length of ITEMS as placeholder for no item

        # concatenate all features into a flat array
        state = np.array([
            frame, x, y, mini_x, mini_y, move_dir,
            xp_bar, health_bar, attack_dmg, armor, magic_resist,
            move_speed, q_cd, w_cd, e_cd, r_cd, d_cd, f_cd
        ] + kdas + cs + health_levels + levels + objectives)

        # minimap detections
        minimap_detections = np.array(step_dict.get("minimap", [])) # shape (N, class + x + y) where N is the number of detections
        # on-screen detections
        champion_detections = np.array(step_dict.get("champions", [])) # shape (N, class + x + y + hp + color) where N is the number of detections
        minion_detections = np.array(step_dict.get("minions", [])) # shape (N, class + x + y + hp + color) where N is the number of detections
        tower_detections = np.array(step_dict.get("towers", [])) # shape (N, class + x + y + hp + color) where N is the number of detections

        # concatenate all on-screen detections
        screen_detections = np.concatenate((champion_detections, minion_detections, tower_detections), axis=0)
        
        return {
            "continuous_f": state.astype(np.float32),
            "minimap_detections": minimap_detections.astype(np.float32),
            "screen_detections": screen_detections.astype(np.float32), 
            "items": np.array(items, dtype=np.uint8)
        }
