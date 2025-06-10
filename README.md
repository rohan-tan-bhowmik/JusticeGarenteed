# JusticeGarenteed

A vision‐only pipeline for training and evaluating a League of Legends “Garen” agent using synthetic scene generation, RF-DETR object detection, expert trajectory extraction, behavior cloning, and PPO fine-tuning.

## High-Level Repository Structure

```
.
├── experiments/               # Logs, metrics, saved models
├── healthbars/                # Healthbar generation & chroma-key utilities
├── trajectories/              # Processed expert trajectories
├── utils/                     # Helper functions (dataset generation, I/O, clustering)
├── LolGarenEnv.py             # Custom Gym-style environment
├── LolGarenEnvtests.py        # Environment unit tests
├── classes.py                 # Detection class labels & enums
├── create_expert_trajectories.py  # Convert replay extractions to trajectories
├── item_dict.json             # In-game item mappings
├── policy.py                  # Policy network definition
├── train_bc_policy.py         # Behavior cloning training script
├── ppo_policy.py              # PPO fine-tuning script
├── train_rfdetr.py            # RF-DETR training script
```
