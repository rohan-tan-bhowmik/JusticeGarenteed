from LolGarenEnv import * 
from policy import GarenBCPolicy
from create_expert_trajectories import *
import json
import torch
import matplotlib.pyplot as plt

def test_parse_csvs_arr():
    video_name = "game-1-1"
    ocr_csv = "trajectories/stats/game-1-1-stats.csv"
    movement_csv = "trajectories/full/game-1-1-full.csv"
    replay_json = "replay_info.json"
    with open(replay_json, 'r') as f:
        replay_info = json.load(f)
    video_info = replay_info.get(video_name, {})

    team_info = {"blue": set(video_info.get("blue_team")),
                "red": set(video_info.get("red_team"))}
    all_states, all_actions = parse_csvs_arr(ocr_csv, movement_csv, team_info=team_info)
    
    print(all_states.shape, all_actions.shape)

def test_dict_to_arr():
    video_name = "game-1-1"
    ocr_csv = "trajectories/stats/game-1-1-stats.csv"
    movement_csv = "trajectories/full/game-1-1-full.csv"
    replay_json = "replay_info.json"
    with open(replay_json, 'r') as f:
        replay_info = json.load(f)
    video_info = replay_info.get(video_name, {})
    team_info = {"blue": set(video_info.get("blue_team")),
            "red": set(video_info.get("red_team"))}
    data = parse_csvs_arr(ocr_csv, movement_csv, team_info = team_info )
    state = data["state"][182]
    action = data["action"][182]
    state_arr = dict_to_obs_arr(state)
    state_dict = dict_to_observation(state)
    print(f"State dict: {state_dict}\n")
    print(f"State array: {state_arr}\n")
    # view the detections to double check
    detections = state_arr[NUM_CONTINUOUS_F:NUM_CONTINUOUS_F + MAX_NUM_DETECTIONS * 5]
    print(detections.reshape(MAX_NUM_DETECTIONS, 5))
    # view items to double check
    print(state_arr[NUM_CONTINUOUS_F + MAX_NUM_DETECTIONS * 5:])

    raise ValueError("Test completed, check the printed state array.")

def test_save_trajectories_bc():
    ocr_dir= "trajectories/stats"
    movement_dir = "trajectories/full"
    replay_json = "replay_info.json"

    save_trajectories_bc(ocr_dir, movement_dir, output_dir = "trajectories/bc", replay_info_path = replay_json)

def testPolicyEmb():
    video_name = "game-1-1"
    ocr_csv = "trajectories/full/game-1-1-full.csv"
    movement_csv = "trajectories/stats/game-1-1-stats.csv"
    replay_json = "replay_info.json"
    with open(replay_json, 'r') as f:
        replay_info = json.load(f)
    video_info = replay_info.get(video_name, {})
    team_info = {"blue": set(video_info.get("blue_team")),
            "red": set(video_info.get("red_team"))}
    
    data = parse_csv_pair(ocr_csv, movement_csv, team_info = team_info)
    policy = GarenBCPolicy()
    for state, action in zip(data["state"], data["action"]):
        state_dict = dict_to_observation(state)
        print(f"state: {state_dict}\n")
        emb = policy.embed(state_dict["continuous_f"], state_dict["screen_detections"], state_dict["minimap_detections"], state_dict["items"])
        print(emb.shape)
        raise ValueError("Test completed, check the printed embedding shape.")

def testPolicyForward():
    trajectories_path = "trajectories/bc/trajectories.pt"
    # Load the trajectories from the file
    data = torch.load(trajectories_path)
    states = data[0]
    print(states.shape)
    policy = GarenBCPolicy()
    outputs = policy.forward(states)
    print(outputs.shape)
    raise ValueError("Test completed, check the printed embedding shape.")

def testPolicyLoss():
    trajectories_path = "trajectories/bc/trajectories.pt"
    # Load the trajectories from the file
    states, actions = torch.load(trajectories_path)

    # raise ValueError("Test completed, check the printed state mean value.")
    policy = GarenBCPolicy()
    print(states.shape, actions.shape)
    outputs = policy.forward(states)
    loss = policy.loss(outputs, actions)
    print(f"Loss: {loss.item()}")
    raise ValueError("Test completed, check the printed loss value.")

if __name__ == "__main__":
    # test_parse_csvs_arr()
    # test_dict_to_arr()
    # test_save_trajectories_bc()
    # testPolicyEmb()
    # testPolicyForward()
    testPolicyLoss()
