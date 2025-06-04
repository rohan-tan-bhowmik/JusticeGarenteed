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
        self.n_steps = len(self.trajectory['state'])
        self._idx = 0
        self.seq_len = seq_len
        self._buffer = deque(maxlen=seq_len)

        # Convert the first state to an observation to define observation_space
        first_state = self.trajectory['state'][0]  # Get the first state
        obs = self._dict_to_observation(first_state)

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
        self._buffer.clear()

        first_state = self.trajectory["state"][0]
        first_obs_dict = self._dict_to_observation(first_state)

        # Fill the buffer with the same first_obs_dict repeated
        for _ in range(self.seq_len):
            # We store a copy so that later we don't accidentally modify the same dict
            self._buffer.append(first_obs_dict.copy())

        # Return a Python list of length seq_len
        return list(self._buffer)

    def step(self, agent_action):
        """
        agent_action is ignored. We return:
          - obs_seq: a list of the last `seq_len` obs_dicts
          - reward:  0.0
          - done:    True if we have exhausted all frames
          - info:    {"expert_action": expert_action}
        """
        expert_action = self.trajectory["action"][self._idx]

        # Advance index
        self._idx += 1
        done = False
        if self._idx >= self.n_steps:
            done = True
            # When done, just return a “zero‐padded” list of obs_dicts
            # (Here we fill with one copy of the last valid obs, though you could also fill with zeros.)
            last_obs = self._dict_to_observation(self.trajectory["state"][-1])
            zero_list = [last_obs.copy() for _ in range(self.seq_len)]
            return zero_list, 0.0, True, {"expert_action": expert_action}

        # Build the next single‐frame obs_dict
        next_state = self.trajectory["state"][self._idx]
        next_obs = self._dict_to_observation(next_state)

        # Push into buffer, pop oldest automatically
        self._buffer.append(next_obs)

        # Return a **list** of length=seq_len
        seq_obs = list(self._buffer)
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
        lanes = ["b1", "b2", "b3", "b4", "b5", "r1", "r2", "r3", "r4", "r5"]
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
