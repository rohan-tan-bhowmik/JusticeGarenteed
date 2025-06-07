from LolGarenEnv import * 
from policy import GarenBCPolicy
from create_expert_trajectories import *
import json

def test_parse_csvs():
    video_name = "game-1-1"
    ocr_csv = "trajectories/stats/game-1-1-stats.csv"
    movement_csv = "trajectories/full/game-1-1-full.csv"
    replay_json = "replay_info.json"
    with open(replay_json, 'r') as f:
        replay_info = json.load(f)
    video_info = replay_info.get(video_name, {})

    team_info = {"blue": set(video_info.get("blue_team")),
                "red": set(video_info.get("red_team"))}
    data = parse_csvs_arr(ocr_csv, movement_csv, team_info=team_info)
    for state, action in zip(data["state"], data["action"]):
        print(f"state: {state}\n")
        print("action", action)
        # raise ValueError("Test completed, check the printed state and action.")
    print(len(data["state"]), len(data["action"]))

def testPolicyEmb():
    ocr_csv = "trajectories/full/game-1-1-full.csv"
    movement_csv = "trajectories/stats/game-1-1-stats.csv"
    data = parse_csvs(ocr_csv, movement_csv)
    policy = GarenBCPolicy()
    for state, action in zip(data["state"], data["action"]):
        state_dict = dict_to_observation(state)
        print(f"state: {state_dict}\n")
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
    test_parse_csvs()
    # testPolicyEmb()
    # testPolicyForward()
    # testPolicyLoss()
