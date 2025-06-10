import gymnasium as gym
from collections import deque
import numpy as np
import time
import math
from collections import deque
import csv
import json
import pickle

from create_expert_trajectories import *

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
        obs = dict_to_observation(first_state)

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
        first_obs_dict = self.dict_to_observation(first_state)

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
            last_obs = self.dict_to_observation(self.trajectory["state"][-1])
            zero_list = [last_obs.copy() for _ in range(self.seq_len)]
            return zero_list, 0.0, True, {"expert_action": expert_action}

        # Build the next single‐frame obs_dict
        next_state = self.trajectory["state"][self._idx]
        next_obs = self.dict_to_observation(next_state)

        # Push into buffer, pop oldest automatically
        self._buffer.append(next_obs)

        # Return a **list** of length=seq_len
        seq_obs = list(self._buffer)
        return seq_obs, 0.0, done, {"expert_action": expert_action}
    
    