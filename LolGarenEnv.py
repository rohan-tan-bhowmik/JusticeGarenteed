import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import math
from collections import deque
from policy import GarenBCPolicy
import csv
import json
from classes import CONDENSED_CHAMPIONS_TO_I, HEALTHBAR_CLASSES
import pickle

ITEMS = set(['b1-1', 'b1-2', 'b1-3', 'b1-4', 'b1-5', 'b1-6', 'b1-7',
 'r1-1', 'r1-2', 'r1-3', 'r1-4', 'r1-5', 'r1-6', 'r1-7',
 'b2-1', 'b2-2', 'b2-3', 'b2-4', 'b2-5', 'b2-6', 'b2-7',
 'r2-1', 'r2-2', 'r2-3', 'r2-4', 'r2-5', 'r2-6', 'r2-7',
 'b3-1', 'b3-2', 'b3-3', 'b3-4', 'b3-5', 'b3-6', 'b3-7',
 'r3-1', 'r3-2', 'r3-3', 'r3-4', 'r3-5', 'r3-6', 'r3-7',
 'b4-1', 'b4-2', 'b4-3', 'b4-4', 'b4-5', 'b4-6', 'b4-7',
 'r4-1', 'r4-2', 'r4-3', 'r4-4', 'r4-5', 'r4-6', 'r4-7',
 'b5-1', 'b5-2', 'b5-3', 'b5-4', 'b5-5', 'b5-6', 'b5-7',
 'r5-1', 'r5-2', 'r5-3', 'r5-4', 'r5-5', 'r5-6', 'r5-7'])

# Lane‐bounds (normalized) for clipping movement
LANE_X_MIN, LANE_X_MAX = 0.15, 0.85
LANE_Y_MIN, LANE_Y_MAX = 0.18, 0.55
ITEM_DICT_PATH = "item_dict.json"

# Precompute unit vectors for 24 directions (every 15 degrees)
N_DIRS = 24
ANGLE = 15  # degrees
STEP_R = 0.05  # normalized step size
MOVEMENT_MAPPING = {angle: i for i, angle in enumerate(range(0, 360, ANGLE))}
print(MOVEMENT_MAPPING)

champtoi = {k: v.lower() for v, k in CONDENSED_CHAMPIONS_TO_I.items()} 
itochamp = {v: k for k, v in champtoi.items()}

def convert_row_dict(row_dict, item_dict):
    """
    Process the row dictionary from the CSV file to convert into numerical values for states/actions
    """
    state_dict = row_dict.copy()
    del state_dict["move_dir"]
    del state_dict["target"]

    action_dict = {}
    for key, item in row_dict.items():
        if item == '' or item == '0':
            if key in ITEMS:
                state_dict[key] = len(item_dict) # Use the length of item_dict as a placeholder for no item
            else:
                state_dict[key] = 0
        elif key == "frame":
            state_dict[key] = int(item)
        elif key == "champion":
            state_dict[key] = champtoi.get(item, -1)
        elif 'kda' in key:
            kda = item.split('/')
            state_dict[key] = kda
        elif key in ITEMS:
            item_name = item[:item.find('_item')]
            if item_name in item_dict:
                # Convert item to vector representation
                state_dict[key] = item_dict[item_name]["id"]
            else:
                raise ValueError(f"Item {item_name} not found in item_dict.")
        elif key == "minimap":
            detections = []
            for minimap_detection in item.split('|'):
                if minimap_detection:
                    parts = minimap_detection.split(':')
                    class_id = champtoi.get(parts[0], -1)  # Get class ID from name
                    coords = parts[1].split(',')
                    x = float(coords[0])
                    y = float(coords[1])
                    detections.append((class_id, x, y))
            state_dict[key] = detections
        elif key == "move_dir":
            if item == 'STOPPED':
                action_dict[key] = N_DIRS + 1  # Use N_DIRS + 1 to represent stopped state
            else:
                item = int(item)
                action_dict[key] = MOVEMENT_MAPPING.get(item)
        elif key == "champions":
            detections = []
            for champion_detection in item.split('|'):
                if champion_detection:
                    champion_name, x, y, _, color = champion_detection.split(',')
                    class_id = champtoi.get(champion_name)
                    detections.append((class_id, float(x), float(y), color))
            state_dict[key] = detections
        elif key == "minions":
            detections = []
            for minion_detection in item.split('|'):
                if minion_detection:
                    minion_name, x, y, _, color = minion_detection.split(',')
                    class_id = champtoi.get(minion_name)
                    detections.append((class_id, float(x), float(y), color))
            state_dict[key] = detections
        elif key == "target":
            if item == 'NONE':
                action_dict[key] = 0 # binary flag for no target
                action_dict["x"] = None
                action_dict["y"] = None
            else:
                action_dict[key] = 1
                x, y = item.split(',')[1:]
                action_dict["x"] = float(x)
                action_dict["y"] = float(y)

    return state_dict, action_dict

def parse_csvs(ocr_csv, movement_csv, ocr_sampling_rate=9, movement_sampling_rate=3):
    """
    Returns a list of dicts, each mapping a movement-frame to a merged dict of
    movement info + OCR info (sampled at ocr_sampling_rate).
    """
    # 1) Read all OCR rows into a list of “{frame: {…fields…}}”
    with open(ITEM_DICT_PATH, 'r') as f:
        item_dict = json.load(f)

    ocr_data = []
    with open(ocr_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = row["frame"]
            # Copy everything except “frame” into a new dict
            info = {k: v for k, v in row.items() if k != "frame"}
            ocr_data.append({frame: info})

    all_states = []
    all_actions = []
    with open(movement_csv, newline="") as f:
        reader = csv.DictReader(f)

        # Pull off the very first movement-row to get start_frame
        try:
            first_row = next(reader)
        except StopIteration:
            return []  # empty file

        start_frame = int(first_row["frame"])
        offset = start_frame % ocr_sampling_rate

        # Prepare to merge OCR in lockstep
        ocr_idx = 0
        ocr_dict = ocr_data[ocr_idx]  # this is like {"123": {…ocr fields…}}

        # Process the first movement-row, then loop over the rest
        for i, row_dict in enumerate([first_row] + list(reader)):
            frame = int(row_dict["frame"])
            # Merge row_dict + ocr_ dict into a fresh dict
            # Note: ocr_dict is {"<ocr_frame>": {..fields..}}, 
            # so we need its inner dict, not the key.
            ocr_frame_str, ocr_fields = next(iter(ocr_dict.items()))

            merged = row_dict.copy()
            merged.update(ocr_fields)
            merged["frame"] = frame
            state_dict, action_dict = convert_row_dict(merged, item_dict)
            ability_actions = [0, 0, 0, 0, 0, 0]
            all_states.append(state_dict)
            if i > 0:
                prev_state = all_states[i - 1]
                for i, cd_key in enumerate(['q-cd', 'w-cd',	'e-cd', 'r-cd',	'd-cd', 'f-cd']):
                    if merged[cd_key] != 0 and prev_state[cd_key] == 0: # ability was used (thus, cooldown is nonzero at current step)
                        ability_actions[i] = 1

            action_dict["abilities"] = ability_actions
            all_actions.append(action_dict)            

            # Update OCR index every time we cross an OCR sampling boundary
            if (ocr_idx % ocr_sampling_rate == offset
                    and ocr_idx < len(ocr_data) - 1):
                ocr_idx += 1
                ocr_dict = ocr_data[ocr_idx] 

    return {"state": all_states, "action": all_actions}

class GarenReplayEnv(gym.Env):
    """
    Gym-style Env that replays a list of (state, action) pairs.
    Each call to step() advances the pointer and returns the stored state and action.
    """
    metadata = {"render.modes": []}

    def __init__(self, trajectory_path):
        super().__init__()
        # Load the list of (state, action) pairs from pickle
        with open(trajectory_path, "rb") as f:
            self.trajectory = pickle.load(f)

        # Each element in trajectory is a dict with 'state' and 'action' keys
        self.n_steps = len(self.trajectory)
        self._idx = 0

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
        first_state = self.trajectory[self._idx]['state']
        obs = self._dict_to_observation(first_state)
        return obs

    def step(self, agent_action):
        """
        Ignore agent_action; return the next stored state and expert action.
        """
        record = self.trajectory[self._idx]
        expert_action = record['action']

        # Advance index
        self._idx += 1
        done = False
        if self._idx >= self.n_steps:
            done = True
            # Return zeroed obs when done
            last_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return last_obs, 0.0, True, {"expert_action": expert_action}

        next_state = self.trajectory[self._idx]['state']
        obs = self._dict_to_observation(next_state)
        return obs, 0.0, done, {"expert_action": expert_action}

    def _dict_to_observation(self, step_dict):
        """
        Convert the feature dict into a flat numpy array of floats.
        Convert numeric strings to floats; extract only chosen features here.
        """
        # Example: extract a few continuous features for demonstration
        x = float(step_dict["minimap_x_ratio"])
        y = float(step_dict["minimap_y_ratio"])
        hp = float(step_dict["health-bar"])
        attack_speed = float(step_dict["attack-speed"])
        # You would extend this to include all desired fields and embeddings

        return np.array([x, y, hp, attack_speed], dtype=np.float32)

def main():
    pass

if __name__ == "__main__":
    main()