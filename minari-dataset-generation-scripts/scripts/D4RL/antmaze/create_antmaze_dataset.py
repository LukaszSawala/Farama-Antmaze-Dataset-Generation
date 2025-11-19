"""
This script generates AntMaze datasets.

Usage:

python create_antmaze_dataset.py

See --help for full list of options.
"""

import sys
import os
import gymnasium as gym
import time
import gymnasium.spaces
import gymnasium_robotics
import minari
from minari import DataCollector, StepDataCallback
from gymnasium.utils import RecordConstructorArgs
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import argparse

from stable_baselines3 import SAC
from controller import WaypointController

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../checks")))
from check_maze_dataset import run_maze_checks, calculate_maze_stats

R = "r"
G = "g"


class AntMazeStepDataCallback(StepDataCallback):
    """Add environment state information to 'infos' and mark trajectory boundaries."""

    ALL_KEYS = [
        "x_position", "y_position", "distance_from_origin",
        "reward_contact", "reward_ctrl", "reward_forward", "reward_survive",
        "success", "x_velocity", "y_velocity",
    ]

    def __call__(self, env, obs, info, action=None, rew=None, terminated=None, truncated=None):
        step_data = super().__call__(env, obs, info, action, rew, terminated, truncated)

        # Check for 'info' (singular)
        if "info" in step_data and isinstance(step_data["info"], dict):
            local_info = step_data["info"]

            # Add our custom keys
            for k in self.ALL_KEYS:
                if k not in local_info:
                    local_info[k] = np.nan

            local_info["qpos"] = np.concatenate([obs["achieved_goal"], obs["observation"][:13]])
            local_info["qvel"] = obs["observation"][13:27]
            local_info["contact_forces"] = obs["observation"][27:] 
            local_info["goal"] = obs["desired_goal"]

            if local_info.get("success", False):
                step_data["truncations"] = True

            step_data["info"] = local_info

        return step_data


class AntMazeV5CompatWrapper(RecordConstructorArgs, gym.Wrapper):
    INFO_KEYS = [
        "x_position", "y_position", "distance_from_origin",
        "reward_contact", "reward_ctrl", "reward_forward", "reward_survive",
        "success", "x_velocity", "y_velocity",
    ]

    def __init__(self, env):
        RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

        assert isinstance(env.observation_space, gymnasium.spaces.Dict), \
            "Wrapper requires a Dict observation space"

        self.original_space = env.observation_space
        self._last_dict_obs = None

        flat_low = np.concatenate([
            self.original_space['observation'].low,
            self.original_space['achieved_goal'].low,
            self.original_space['desired_goal'].low,
        ])
        flat_high = np.concatenate([
            self.original_space['observation'].high,
            self.original_space['achieved_goal'].high,
            self.original_space['desired_goal'].high,
        ])
        self.flat_observation_space = gymnasium.spaces.Box(
            low=flat_low,
            high=flat_high,
            shape=flat_low.shape,
            dtype=self.original_space['observation'].dtype,
        )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._last_dict_obs = deepcopy(obs)
        return obs, self._process_info(info)   

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._last_dict_obs = deepcopy(obs)
        return obs, reward, terminated, truncated, self._process_info(info)

    def _process_info(self, info):
        for k in self.INFO_KEYS:
            if k not in info:
                info[k] = np.nan
        return info

    def get_dict_obs(self):
        return deepcopy(self._last_dict_obs)

    def flatten_obs(self, obs=None):
        if obs is None:
            obs = self._last_dict_obs
        if isinstance(obs, dict):
            return np.concatenate([
                obs['observation'],
                obs['achieved_goal'],
                obs['desired_goal']
            ])
        return obs

    def get_serializable_obs(self):
        obs = self._last_dict_obs
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}


def wrap_maze_obs(obs, waypoint_xy):
    """
    Converts goals to numpy arrays for arithmetic without modifying the controller's maze object.
    """
    waypoint_xy = np.array(waypoint_xy)
    achieved_goal = np.array(obs["achieved_goal"])
    goal_direction = waypoint_xy - achieved_goal
    return np.concatenate([obs["observation"], goal_direction])


def init_dataset(collector_env, dataset_id, eval_env_spec, expert_policy, args, description=None):
    try:
        dataset = minari.load_dataset(dataset_id)
        print(f"Appending to existing dataset: {dataset_id}")
        collector_env.add_to_dataset(dataset)
        return dataset
    except (ValueError, FileNotFoundError):
        print(f"Creating new dataset: {dataset_id}")

        # --- TIMER START ---
        print("\nStarting metadata generation (policy evaluation)... This may take several minutes.")
        start_time = time.time()

        dataset = collector_env.create_dataset(
            dataset_id=dataset_id,
            eval_env=eval_env_spec,
            expert_policy=expert_policy,
            algorithm_name=f"{args.maze_solver}+SAC",
            code_permalink="https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation",
            author=args.author,
            author_email=args.author_email,
            description=description,
            num_episodes_average_score=100, # default, not recommended to change as it is used to ensure metadata for a dataset are valid
        )

        # --- TIMER END ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Metadata generation finished in {elapsed_time:.2f} seconds.\n")

        return dataset

EVAL_ENV_MAPS = {
    "umaze": [[1, 1, 1, 1, 1],
              [1, 0, 0, R, 1],
              [1, 0, 1, 1, 1],
              [1, 0, 0, G, 1],
              [1, 1, 1, 1, 1]],
    "medium": [[1, 1, 1, 1, 1, 1, 1, 1],
               [1, R, 0, 1, 1, 0, 0, 1],
               [1, 0, 0, 1, 0, 0, 0, 1],
               [1, 1, 0, 0, 0, 1, 1, 1],
               [1, 0, 0, 1, 0, 0, 0, 1],
               [1, 0, 1, 0, 0, 1, 0, 1],
               [1, 0, 0, 0, 1, 0, G, 1],
               [1, 1, 1, 1, 1, 1, 1, 1]],
    "large": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, R, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
              [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
              [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
              [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
              [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
              [1, 0, 0, 1, 0, 0, 0, 1, 0, G, 0, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
}


DATASET_ID_TO_ENV_ID = {
    "antmaze-open-diverserandom-v1": "AntMaze_Open_Diverse_GR-v5",
    "antmaze-open-diverserandom-dense-v1": "AntMaze_Open_Diverse_GRDense-v5",
    "antmaze-umaze-v1": "AntMaze_UMaze-v5",
    "antmaze-umaze-dense-v1": "AntMaze_UMazeDense-v5",
    "antmaze-medium-diverse-dense-v1": "AntMaze_Medium_Diverse_GDense-v5",
    "antmaze-medium-diverserandom-dense-v1": "AntMaze_Medium_Diverse_GRDense-v5",
    "antmaze-medium-diverserandom-dense-resettarget-v1": "AntMaze_Medium_Diverse_GRDense-v5",
    "antmaze-large-diverse-v1": "AntMaze_Large_Diverse_G-v5",
    "antmaze-large-diverserandom-v1": "AntMaze_Large_Diverse_GR-v5",
    "antmaze-large-diverse-dense-v1": "AntMaze_Large_Diverse_GDense-v5",
    "antmaze-large-diverserandom-dense-v1": "AntMaze_Large_Diverse_GRDense-v5",
    "antmaze-large-diverserandom-dense-resettarget-v1": "AntMaze_Large_Diverse_GRDense-v5",
    "antmaze-snake-large-dense-v1": "AntMaze_LargeDense-v5",
}


if __name__ == "__main__":
    program_start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze-solver", type=str, default="QIteration")
    parser.add_argument("--policy-file", type=str, default="GoalAnt-53.5m-1.93k")
    parser.add_argument("--total-timesteps", type=int, default=50000)
    parser.add_argument("--action-noise", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint_interval", type=int, default=10000)
    parser.add_argument("--author", type=str, default="Lukasz Sawala")
    parser.add_argument("--author-email", type=str, default="lukaszsawala2003@gmail.com")
    parser.add_argument("--upload-dataset", type=bool, default=False)
    parser.add_argument("--path_to_private_key", type=str, default=None)
    args = parser.parse_args()

    for dataset_id, env_id in DATASET_ID_TO_ENV_ID.items():
        print("\n")
        print("="*40)
        print(f"STARTING DATA COLLECTION FOR {env_id}")
        print("="*40)
        print("\n")

        split_dataset_id = dataset_id.split("-")
        reset_target = "resettarget" in split_dataset_id

        description_str = f"A Minari dataset for AntMaze, generated from the '{env_id}' environment. Dataset ID: {dataset_id}."

        if "snake" in split_dataset_id:
            snake_map = [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, "C", 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, "C", 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
            env = gym.make(env_id, continuing_task=True, reset_target=False, maze_map=snake_map)
        else:
            env = gym.make(env_id, continuing_task=True, reset_target=reset_target)

        original_obs_space = env.observation_space
        obs_dim = original_obs_space['observation'].shape[0]
        goal_dim = original_obs_space['achieved_goal'].shape[0]

        env = AntMazeV5CompatWrapper(env)
        print("Successfully created environment:", env_id)

        collector_env = DataCollector(env, step_data_callback=AntMazeStepDataCallback, record_infos=True)
        np.random.seed(args.seed)

        model = SAC.load(args.policy_file)

        def action_callback(obs, waypoint_xy):
            return model.predict(wrap_maze_obs(obs, waypoint_xy))[0]

        true_env = env
        while hasattr(true_env, "env"):
            true_env = true_env.env
        waypoint_controller = WaypointController(true_env.maze, action_callback)

        obs_dict, info = collector_env.reset(seed=args.seed)
        obs_flat = env.flatten_obs(obs_dict)

        dataset = None
        print(f"\nCreating {dataset_id}:")
        for step in tqdm(range(args.total_timesteps)):
            # Pass dict obs to controller
            action = waypoint_controller.compute_action(obs_dict)
            action += args.action_noise * np.random.randn(*action.shape)
            action = np.clip(action, -1.0, 1.0)

            # Step the environment
            obs_dict, reward, terminated, truncated, info = collector_env.step(action)
            obs_flat = env.flatten_obs(obs_dict)

            # Checkpointing
            if (step + 1) % args.checkpoint_interval == 0:
                truncated = True
                if dataset is None:
                    # Use unwrapped env spec
                    base_env = env.unwrapped
                    eval_env_spec = deepcopy(base_env.spec)

                    # Tell the spec to run headless evaluation
                    eval_env_spec.kwargs["render_mode"] = None
                    # enforce end of the while loop
                    eval_env_spec.max_episode_steps = 1000
                    dataset = init_dataset(
                        collector_env,
                        dataset_id,
                        eval_env_spec,
                        waypoint_controller.compute_action,
                        args,
                        description=description_str
                    )
                else:
                    collector_env.add_to_dataset(dataset)

            # Handle trajectory truncation
            if truncated:
                args.seed += 1
                obs_dict, info = collector_env.reset(seed=args.seed)
                obs_flat = env.flatten_obs(obs_dict)


        print(f"Checking {dataset_id}:")
        assert run_maze_checks(dataset)
        
        # Calculate the stats one more time
        success_rate, avg_velocity = calculate_maze_stats(dataset)
        
        # Create the metadata dictionary
        final_metadata = {
            "dataset_success_rate": success_rate,
            "dataset_avg_velocity": avg_velocity
        }

        # Update the dataset file on disk
        dataset.storage.update_metadata(final_metadata)
        print("Successfully added success_rate and avg_velocity to metadata.")

        if args.upload_dataset:
            minari.upload_dataset(dataset_id, args.path_to_private_key)

    program_end_time = time.time()
    total_elapsed = program_end_time - program_start_time
    print(f"TOTAL EXECUTION TIME: {total_elapsed:.2f} seconds.\n")