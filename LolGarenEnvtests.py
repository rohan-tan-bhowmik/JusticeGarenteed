from LolGarenEnv import * 
from policy import GarenBCPolicy
from create_expert_trajectories import *

def test_parse_csvs():
    ocr_csv = "test_ocr.csv"
    movement_csv = "test_movement.csv"
    data = parse_csvs(ocr_csv, movement_csv)
    for state, action in zip(data["state"], data["action"]):
        print(f"state: {state}\n")
        print("action", action)
        raise ValueError("Test completed, check the printed state and action.")

def dict_to_observation( step_dict):
        """
        This is a non-class version just for testing
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
        champion_detections = np.array(step_dict.get("champions", np.zeros((0, 5)))) # shape (N, class + x + y + hp + color) where N is the number of detections
        minion_detections = np.array(step_dict.get("minions", np.zeros((0, 5)))) # shape (N, class + x + y + hp + color) where N is the number of detections
        tower_detections = np.array(step_dict.get("towers", np.zeros((0, 5)))) # shape (N, class + x + y + hp + color) where N is the number of detections
        screen_detections = np.concatenate((champion_detections, minion_detections, tower_detections), axis=0)
        print(state.shape)
        return {
            "continuous_f": state.astype(np.float32),
            "minimap_detections": minimap_detections.astype(np.float32),
            "screen_detections": screen_detections.astype(np.float32), 
            "items": np.array(items, dtype=np.uint8)
        }

def testPolicyEmb():
    ocr_csv = "trajectories/full/game-1-full.csv"
    movement_csv = "trajectories/stats/game-1-stats.csv"
    data = parse_csvs(ocr_csv, movement_csv)
    policy = GarenBCPolicy()
    for state, action in zip(data["state"], data["action"]):
        state_dict = dict_to_observation(state)
        emb = policy.embed(state_dict["continuous_f"], state_dict["screen_detections"], state_dict["minimap_detections"], state_dict["items"])
        print(emb.shape)
        raise ValueError("Test completed, check the printed embedding shape.")

def testPolicyForward():
    ocr_csv = "test_ocr.csv"
    movement_csv = "test_movement.csv"
    data = parse_csvs(ocr_csv, movement_csv)
    policy = GarenBCPolicy()
    obs_sequence = [dict_to_observation(state) for state in data["state"]]
    print(len(obs_sequence))
    outputs = policy.forward(obs_sequence)
    print(outputs)
    raise ValueError("Test completed, check the printed embedding shape.")

def testPolicyLoss():
    ocr_csv = "test_ocr.csv"
    movement_csv = "test_movement.csv"
    device = "cpu"
    data = parse_csvs(ocr_csv, movement_csv)
    policy = GarenBCPolicy(device = device)
    obs_sequence = [dict_to_observation(state) for state in data["state"]]
    outputs = policy.forward(obs_sequence)
    loss = policy.loss(outputs, data["action"][-1])
    print(f"Loss: {loss.item()}")
    raise ValueError("Test completed, check the printed loss value.")

if __name__ == "__main__":
    # test_parse_csvs()
    testPolicyEmb()
    # testPolicyForward()
    # testPolicyLoss()
