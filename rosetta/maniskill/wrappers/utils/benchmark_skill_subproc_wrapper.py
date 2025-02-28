import os
import sys
import argparse
import gymnasium as gym
import torch
import mani_skill.envs
import numpy as np
import json
from tqdm.notebook import tqdm
from dataclasses import dataclass
import tyro
from stable_baselines3 import MAPLE
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EveryNTimesteps
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from feedback_to_reward.maniskill.customized_tasks import *
from feedback_to_reward.maniskill.wrappers.skill_wrapper import SkillGymWrapper
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from feedback_to_reward.maniskill.wrappers.maniskill_sb3_wrapper import ContinousTaskEnv
from feedback_to_reward.maniskill.wrappers.skill_wrapper import SkillGymWrapper
from feedback_to_reward.maniskill.primitive_skills.primitive_skills_pose import _quat2euler
from feedback_to_reward.maniskill.utils.utils import get_task_env_source,get_avail_save_path
from mani_skill.envs.sapien_env import BaseEnv
from feedback_to_reward.maniskill.pipelines.reward_manipulator import RewardManipulator
from stable_baselines3.common.noise import NormalActionNoise
import time
from stable_baselines3.maple.policies import MAPLEPolicy
@dataclass
class MAPLEConfig:
    env_id: str = "Align3CubeCurLearning"
    timesteps: int = int(1e8)
    learning_rate: float = 3e-3
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 32
    gsteps: int = 5
    save_path: str = './maple_skill_runs'
    use_aff: bool = False
    save_checkpoint: bool = True
    max_episode_steps: int = 5
    max_steps_per_video: int = 5
    num_envs: int = 32
    eval_step: int = 100000
    reward_json_path: str = None
    add_noise: bool = False
    noise_mean: float = 0.0
    noise_sigma: float = 0.1
    net_arch_depth: int = 2
    net_arch_width: int = 256

def make_env(env_id: str,record_dir=None,max_episode_steps=5,max_steps_per_video=5,reward_json_path=None):
    def _init() -> gym.Env:
        import feedback_to_reward.maniskill.curriculum_learning.customized_tasks
        env_kwargs = dict(obs_mode="state", control_mode="pd_ee_delta_pose", render_mode="rgb_array", sim_backend="cpu")
        env = gym.make(env_id, num_envs=1, enable_shadow=True,**env_kwargs)
        if reward_json_path is not None:
            reward_manipulator = RewardManipulator(env=env, json_path=reward_json_path)
            reward_manipulator.change_function(['evaluate','skill_reward'])
        env=CPUGymWrapper(env)
        env=SkillGymWrapper(env,
                    skill_indices=env.task_skill_indices,
                    record_dir=record_dir,
                 max_episode_steps=max_episode_steps,
                 max_steps_per_video=max_steps_per_video,)
        return env
    return _init

def main():
    args=MAPLEConfig()
    env = SubprocVecEnv([make_env(args.env_id, 
                                    max_episode_steps=args.max_episode_steps,reward_json_path=args.reward_json_path) for i in range(args.num_envs)])
    n_eval_episodes=2
    episode_steps=10
    import time
    start_time=time.time()
    for episode in range(n_eval_episodes):
        obs = env.reset()
        total_reward = 0
        step=0
        for step in range(episode_steps):
            obs, reward, done, info = env.step(np.stack([env.action_space.sample() for i in range(args.num_envs)]))  # Take a step in the environment
            step+=1
            print(f"Step {step}: done: {done}")
            total_reward += reward
    total_time=time.time()-start_time

    import json
    data_dict={'time':total_time}
    with open("sub_proc.json", 'w') as f:
        json.dump(data_dict, f, indent=4)
        
if __name__ == "__main__":
    main()
        