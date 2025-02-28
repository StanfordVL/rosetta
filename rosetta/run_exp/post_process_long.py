import os
import json
import shutil
import os
import gymnasium as gym
import numpy as np
import json
from stable_baselines3 import MAPLE
import rosetta.maniskill.long_env
import rosetta.maniskill.short_env 
from rosetta.maniskill.wrappers.skill_wrapper import SkillGymWrapper
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from rosetta.maniskill.wrappers.skill_wrapper import SkillGymWrapper
from mani_skill.envs.sapien_env import BaseEnv
from rosetta.maniskill.pipelines.reward_manipulator import RewardManipulator
from rosetta.prompts.utils import stitch_mp4_files
from stable_baselines3 import MAPLE
from typing import Callable
import glob
import fire

def make_env(env_id: str, record_dir=None, max_episode_steps=6, max_steps_per_video=6, reward_json_path=None, stage=0) -> Callable:
    def _init() -> gym.Env:
        env_kwargs = dict(obs_mode="state", control_mode="pd_ee_delta_pose", render_mode="rgb_array", sim_backend="cpu", stage=stage)
        env = gym.make(env_id, num_envs=1, enable_shadow=True, **env_kwargs)
        if reward_json_path is not None:
            reward_manipulator = RewardManipulator(env=env, json_path=reward_json_path)
            reward_manipulator.change_function(['evaluate','skill_reward'])
        env = CPUGymWrapper(env)
        env = SkillGymWrapper(env,
                    skill_indices=env.task_skill_indices,
                    record_dir=record_dir,
                    max_episode_steps=max_episode_steps,
                    max_steps_per_video=max_steps_per_video,)
        return env
    return _init

def post_process_long(src_path: str) -> None:
    """
    Post-process function to handle model loading, rollouts and video generation.
    
    Args:
        src_path (str): Path to the source directory containing exp_config.json
    """
    try:
        # Load exp_config.json and extract stages
        with open(os.path.join(src_path, 'exp_config.json'), 'r') as f:
            exp_config = json.load(f)
        annotator_id = exp_config.get('annotator_id')
        env_id = exp_config.get('env_id')
        uid_reward = exp_config.get('uid_reward')
        video_name = f"{annotator_id}-{env_id}-{uid_reward}"
        stages = int(exp_config.get('stages'))
        
        if not all([env_id, stages]):
            raise ValueError("Missing required fields (env_id or stages) in exp_config.json")
        
        # Find and copy the final model
        final_stage = stages - 1
        final_model_path = os.path.join(src_path, 'exp/eval/best_model/best_model.zip')
        
        if not os.path.exists(final_model_path):
            raise FileNotFoundError(f"Final model not found at: {final_model_path}")
        
        # Copy model to src_path
        target_model_path = os.path.join(src_path, 'best_model.zip')
        shutil.copy2(final_model_path, target_model_path)
        
        # Find the best rollout
        demo_dir = os.path.join(src_path, "demo_dir")
        rollout_infos = sorted(glob.glob(os.path.join(demo_dir, "episode_*", "rollout_info.json")))
        final_stage_achieved = []
        for fp in rollout_infos:
            with open(fp, "r") as f:
                rollout_info = json.load(f)
            final_stage_achieved.append(rollout_info["final_stage"])
        final_stage_achieved
        best_rollout_idx = np.argmax(final_stage_achieved)

        # Copy the best rollout to a new directory
        shutil.copytree(rollout_infos[best_rollout_idx].split("rollout_info.json")[0], os.path.join(demo_dir, "best_demo"))

        # Generate video from the best rollout
        new_video_path = os.path.join(src_path, f'{video_name}.mp4')
        stitch_mp4_files(os.path.join(src_path, "demo_dir", "best_demo", "video") , new_video_path)
        
        print(f"Post-processing completed successfully:")
        print(f"1. Copied final model to: {target_model_path}")
        print(f"2. Generated video (from demo_dir) at: {new_video_path}")
        
    except Exception as e:
        print(f"Error during post-processing: {str(e)}")
        raise

if __name__ == "__main__":
    fire.Fire(post_process_long)