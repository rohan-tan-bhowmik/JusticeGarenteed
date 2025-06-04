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
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Precompute unit vectors for 24 directions (every 15 degrees)
N_DIRS = 24
ANGLE = 15  # degrees
STEP_R = 0.05  # normalized step size
MOVEMENT_MAPPING = {angle: i for i, angle in enumerate(range(0, 360, ANGLE))}
print(MOVEMENT_MAPPING)

champtoi = {k.lower(): v for k, v in CONDENSED_CHAMPIONS_TO_I.items()} 
itochamp = {v: k for k, v in champtoi.items()}

def convert_row_dict(row_dict, item_dict):
    """
    Process the row dictionary from the CSV file to convert into numerical values for states/actions
    """
    state_dict = row_dict.copy()
    del state_dict["move_dir"]
    del state_dict["target"]
    state_dict["garen_pos"] = (0.5, 0.5)
    action_dict = {}
    
    for key, item in row_dict.items():
        if item == '' or item == '0':
            if key in ITEMS:
                state_dict[key] = len(item_dict) # Use the length of item_dict as a placeholder for no item
            elif key not in 'minions towers minimap champions':
                state_dict[key] = 0
            else:
                state_dict[key] = np.zeros((0, 5), dtype=np.float32)
        elif key == "frame":
            state_dict[key] = int(item)
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
            rows = []  # Python list of [class_id, x, y]
            for minimap_detection in item.split('|'):
                if not minimap_detection:
                    continue
                parts = minimap_detection.split(':')
                class_id = champtoi.get(parts[0], len(champtoi))  # or -1 if name not found
                coords = parts[1].split(',')
                x = float(coords[0])
                y = float(coords[1])
                rows.append([class_id, x, y])

            # Convert list → numpy array of shape (N, 3), or (0, 3) if empty
            detections = np.array(rows, dtype=np.float32)
            state_dict[key] = detections

        elif key == "move_dir":
            if item == 'STOPPED':
                action_dict[key] = N_DIRS + 1  # sentinel for “stopped”
            else:
                action_dict[key] = MOVEMENT_MAPPING.get(int(item), N_DIRS + 1)

        elif key == "champions":
            rows = []  # Python list of [class_id, x_ratio, y_ratio, health, color]
            detected_champions = set()
            for champion_detection in item.split('|'):
                if not champion_detection:
                    continue
                champion_name, x, y, health, color = champion_detection.split(',')
                color_flag = 1 if color == 'Blue' else 0

                if champion_name.lower() == 'garen':
                    state_dict["garen_pos"] = (float(x), float(y))

                detected_champions.add(champion_name.lower())
                class_id = champtoi.get(champion_name.lower(), len(champtoi))  # or len(champtoi) if name not found
                rows.append([
                    class_id,
                    round(float(x) / SCREEN_WIDTH, 3),
                    round(float(y) / SCREEN_HEIGHT, 3),
                    float(health),
                    color_flag
                ])

            
            detections = np.array(rows, dtype=np.float32)
            state_dict[key] = detections

        elif key == "minions" or key == "towers":
            rows = []  # Python list of [class_id, x_ratio, y_ratio, health, color]
            for detection in item.split('|'):
                if not detection:
                    continue
                name, x, y, health, color = detection.split(',')
                color_flag = 1 if color == 'Blue' else 0
                class_id = champtoi.get(name.lower(), len(champtoi))
                rows.append([
                    class_id,
                    round(float(x) / SCREEN_WIDTH, 3),
                    round(float(y) / SCREEN_HEIGHT, 3),
                    float(health),
                    color_flag
                ])
            
            detections = np.array(rows, dtype=np.float32)
            
            state_dict[key] = detections
            print(f"{key} = {state_dict[key]}")

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

def main():
    pass

if __name__ == "__main__":
    main()