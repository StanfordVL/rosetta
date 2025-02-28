from collections import defaultdict
import inspect
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
from stable_baselines3 import MAPLE
import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# F2R specific imports 
import feedback_to_reward.maniskill.long_env
import feedback_to_reward.maniskill.short_env 
from feedback_to_reward.maniskill.pipelines.reward_manipulator import RewardManipulator
from feedback_to_reward.maniskill.utils.utils import get_task_env_source
from feedback_to_reward.maniskill.wrappers.record_wrapper import RecordWrapper
from feedback_to_reward.maniskill.short_horizon_learning.maniskill_ppo import Agent
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from feedback_to_reward.maniskill.wrappers.skill_wrapper import SkillGymWrapper


@dataclass
class Args:
    out_dir: Optional[str] = None
    """the output directory to save the results"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    task_type: Literal["short", "long"] = "short"
    """type of task (short or long horizon)"""
    num_demo: int = 1
    """the number of demonstrations to collect"""
    sample_freq: int = 1
    """the frequency of recording the trajectory"""
    seed: int = 0
    """random seed"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    max_episode_steps: int = 6
    """for long horizon pipeline, the maximum number of steps per episode"""
    reward_json_path: Optional[str] = None
    """path to the reward json file"""

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_model(env, net_arch, seed):
    model = MAPLE(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=net_arch),
        action_dim_s=len(env.unwrapped.task_skill_indices.keys()),
        seed=seed
    )
    return model

def make_env(env_id: str,record_dir=None,max_episode_steps=5,max_steps_per_video=1,reward_json_path=None,stage=0):
    def _init() -> gym.Env:
        env_kwargs = dict(obs_mode="state", control_mode="pd_ee_delta_pose", render_mode="rgb_array", sim_backend="cpu",stage=stage)
        env = gym.make(env_id, num_envs=1, enable_shadow=True,**env_kwargs)
        if reward_json_path is not None:
            reward_manipulator = RewardManipulator(env=env, json_path=reward_json_path)
            reward_manipulator.change_function(['evaluate','skill_reward'])
        env=CPUGymWrapper(env)
        env=SkillGymWrapper(env,
                    skill_indices=env.unwrapped.task_skill_indices,
                    record_dir=record_dir,
                 max_episode_steps=max_episode_steps,
                 max_steps_per_video=max_steps_per_video,)
        return env
    return _init

def record_demo_long_horizon(env_id, model_checkpoint, max_episode_steps, out_dir, num_demo = 1, rand_seed = 42, device="cuda", reward_json_path=None):
    os.makedirs(out_dir, exist_ok=True)
    seed_all(rand_seed)

    env = make_env(env_id, record_dir=out_dir, max_episode_steps=max_episode_steps, max_steps_per_video=1, stage=0, reward_json_path=reward_json_path)()

    model = MAPLE.load(model_checkpoint, env=env)

    obs, info = env.reset()
    rollout_dict = {}
    for i in range(num_demo):
        print("Running demo: ", i)
        done = False
        length = 0
        for _ in range(max_episode_steps):
            length += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        rollout_dict = {
            "is_success": bool(info["is_success"].any()),
            "length": length,
            "final_stage": env.cur_stage,
        }
        with open(os.path.join(out_dir, f"episode_{i}", f"rollout_info.json"), "w") as f:
            json.dump(rollout_dict, f)
        obs, info = env.reset()
    print("Evaluation is done. The results are saved in: ", out_dir)

def record_demo_short_horizon(env_id, model_checkpoint, out_dir, sample_freq = 5, num_demo = 1, rand_seed = 42, device="cuda", reward_json_path=None):
    os.makedirs(out_dir, exist_ok=True)
    seed_all(rand_seed)

    env_kwargs = dict(obs_mode="state", control_mode="pd_joint_delta_pos", render_mode="rgb_array", sim_backend="gpu")
    eval_envs = gym.make(env_id, num_envs=1, **env_kwargs)

    if reward_json_path is not None:
        rm = RewardManipulator(env_id=args.env_id, json_path=args.reward_json_path, sim_backend="gpu")
        rm.change_function(["evaluate", "compute_dense_reward"])

    max_episode_steps = gym_utils.find_max_episode_steps_value(eval_envs)
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    eval_envs = RecordWrapper(eval_envs, record_dir=out_dir, sample_freq=sample_freq)
    eval_envs = ManiSkillVectorEnv(eval_envs, 1, ignore_terminations=True, record_metrics=True)
    print(f"Environment: {env_id} has been loaded")

    agent = Agent(eval_envs).to(device)
    agent.load_state_dict(torch.load(model_checkpoint, weights_only=False))
    print(f"Agent has been loaded from {model_checkpoint}")

    print("Evaluating the agent...")
    agent.eval()
    eval_obs, _ = eval_envs.reset()
    for demo_id in range(num_demo):
        has_success = False
        accumulated_reward = 0
        print("Running demo: ", demo_id)
        for step in range(max_episode_steps):
            with torch.no_grad():
                eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                accumulated_reward += eval_rew.item()
            if eval_terminations or eval_truncations:
                print(f"Took {step} Steps")
                break
            if eval_infos["success"]:
                has_success = True
        
        rollout_dict = {
            "is_success": has_success,
            "accumulated_reward": accumulated_reward,
        }
        with open(os.path.join(out_dir, f"episode_{demo_id}", f"rollout_info.json"), "w") as f:
            json.dump(rollout_dict, f)
        print("Rollout Concluded...")
        
    print("Evaluation is done. The results are saved in: ", out_dir)
        

if __name__ == "__main__":
    args = tyro.cli(Args)

    assert args.out_dir is not None, "Please specify the output directory using --out_dir"
    assert args.checkpoint is not None, "Please specify the checkpoint path using --checkpoint"
    assert args.env_id is not None, "Please specify the environment id using --env_id"

    if args.task_type == "short":
        record_demo_short_horizon(args.env_id, args.checkpoint, args.out_dir, args.sample_freq, args.num_demo, args.seed, "cuda" if args.cuda else "cpu")
    elif args.task_type == "long":
        record_demo_long_horizon(args.env_id, args.checkpoint, args.max_episode_steps, args.out_dir, args.num_demo, args.seed, "cuda" if args.cuda else "cpu", args.reward_json_path)
    else:
        raise ValueError("task_type must be either 'short' or 'long'")