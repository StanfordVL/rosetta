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
from stable_baselines3 import MAPLEStage, MAPLE
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EveryNTimesteps
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from rosetta.maniskill.wrappers.skill_wrapper import SkillGymWrapper
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from rosetta.maniskill.wrappers.skill_wrapper import SkillGymWrapper
from rosetta.maniskill.utils.utils import get_task_env_source, get_avail_save_path, get_modified_source
from mani_skill.envs.sapien_env import BaseEnv
from rosetta.maniskill.pipelines.reward_manipulator import RewardManipulator
from stable_baselines3.common.noise import NormalActionNoise
import time
from stable_baselines3.maple.policies import MAPLEPolicy
from rosetta.sb3.callbacks.reward_callback import RewardEvalCallback
import rosetta.maniskill.long_env
import timeout_decorator
from rosetta.sb3.callbacks.early_stop_callback import EarlyStoppingEvalCallback
from copy import deepcopy

@dataclass
class MAPLEConfig:
    # env config
    env_id: str = "Align3CubeCurLearning"
    timesteps: int = int(2e8)
    
    # algorithm config
    learning_rate: float = 3e-3
    gamma: float = 0.2
    tau: float = 0.5
    batch_size: int = 1024
    gsteps: int = 8
    use_aff: bool = False
    add_noise: bool = False
    noise_mean: float = 0.0
    noise_sigma: float = 0.1
    net_arch_depth: int = 2
    net_arch_width: int = 512
    learning_starts: int = 0 # low-level steps
    
    # training config
    save_path: str = './test'
    auto_save_path: bool = False
    save_checkpoint: bool = False
    max_episode_steps: int = 6
    max_steps_per_video: int = 1
    num_envs: int = 32
    eval_step: int = int(1e5)
    reward_json_path: str = None
    stage: int=0
    load_checkpoint_path: str = None
    time_limit: int = 144000
    early_stop_threshold: float = 0.9

def make_env(env_id: str,record_dir=None,max_episode_steps=5,max_steps_per_video=5,reward_json_path=None,stage=0):
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

def get_env(args):
    # create one eval environment
    eval_env = SubprocVecEnv([make_env(args.env_id, record_dir=os.path.join(args.save_path, 'videos'),
                                       max_episode_steps=args.max_episode_steps, max_steps_per_video=args.max_steps_per_video,
                                       reward_json_path=args.reward_json_path, stage=args.stage) for i in range(1)])
    print(eval_env.observation_space, eval_env.action_space)
    
    reward_env = SubprocVecEnv([make_env(args.env_id,
                                       max_episode_steps=args.max_episode_steps,
                                       reward_json_path=args.reward_json_path, stage=args.stage) for i in range(1)])
    eval_env.seed(0)
    eval_env.reset()
    
    
    # create num_envs training environments
    env = SubprocVecEnv([make_env(args.env_id, 
                                  max_episode_steps=args.max_episode_steps,
                                  reward_json_path=args.reward_json_path,
                                  stage=args.stage) for i in range(args.num_envs)])
    env.seed(0)
    env.reset()
    return env, reward_env, eval_env

def get_callback(args, env, reward_env, eval_env):
    eval_callback = EarlyStoppingEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_path, 'eval', 'best_model'),
        log_path=os.path.join(args.save_path, 'eval', 'log'),
        n_eval_episodes=10,
        eval_freq=1,
        deterministic=True,
        render=False,
        early_stop_threshold=args.early_stop_threshold
    )
    
    eval_callback = EveryNTimesteps(n_steps=args.eval_step, callback=eval_callback)
    callback = [eval_callback]

    reward_event = RewardEvalCallback(
        reward_env,
        eval_freq=1,
        n_eval_episodes=10,
    )
    reward_callback = EveryNTimesteps(n_steps=args.eval_step, callback=reward_event)
    callback.append(reward_callback)
    if args.save_checkpoint:
        checkpoint_event = CheckpointCallback(
            save_freq=1,
            save_path=os.path.join(args.save_path, 'checkpoints'),
            name_prefix='maple',
            save_vecnormalize=True,
        )
        checkpoint_callback = EveryNTimesteps(n_steps=args.eval_step, callback=checkpoint_event)
        callback.append(checkpoint_callback)
    return callback

def main():
    args = tyro.cli(MAPLEConfig)
    #args.save_path = os.path.join(args.save_path, args.env_id)
    # Generate an index for save_path to prevent overwriting
    if args.auto_save_path:
        args.save_path=os.path.join(args.save_path, args.env_id)
        args.save_path = get_avail_save_path(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    
    #Change reward function if provided
    if args.reward_json_path!=None:
        os.system(f"cp {args.reward_json_path} {args.save_path}")
            
    print(args)
    # folder created
    with open(os.path.join(args.save_path, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    
    env=gym.make(args.env_id, num_envs=1,sim_backend="cpu",
                 obs_mode="state_dict", control_mode="pd_ee_delta_pose", render_mode="rgb_array",)
    if args.reward_json_path is not None:
        reward_manipulator = RewardManipulator(env=env, json_path=args.reward_json_path)
        reward_manipulator.change_function(['evaluate','skill_reward'])
        source_code=get_task_env_source(env, BaseEnv)
        with open(args.reward_json_path, "r") as f:
            reward_json = json.load(f)
        modified_source_code = get_modified_source(source_code, reward_json)    
        with open(os.path.join(args.save_path,"env_source_modified.py"), "w") as f:
            f.write(modified_source_code)

    task_skill_indices=env.unwrapped.task_skill_indices
    source_code=get_task_env_source(env, BaseEnv)
    with open(os.path.join(args.save_path,"env_source.py"), "w") as f:
        f.write(source_code)


    info = env.evaluate()
    # find the largest key in the info dictionary with format "stageX_success"
    num_stages = max([int(key.split('_')[0][5:]) for key in info.keys() if (('stage' in key) and ('success' in key))]) + 1
    args.num_stage = num_stages

    print(f"Number of stages: {num_stages}")
    env.close()

    net_arch = [args.net_arch_width] * args.net_arch_depth
    
    env, reward_env, eval_env = get_env(args)
    callback = get_callback(args, env, reward_env, eval_env)
    # MAPLEStage.load(model_checkpoint, env=env)
    
    model_sac = MAPLEStage(
        "MlpPolicy",
        env,
        num_stages = num_stages,
        policy_kwargs=dict(net_arch=net_arch),
        action_dim_s=len(task_skill_indices.keys()),
        learning_rate=args.learning_rate,
        buffer_size=1000000,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        gradient_steps=args.gsteps,
        seed=1,
        tensorboard_log=os.path.join(args.save_path, 'logs'),
    )

    if args.load_checkpoint_path:
        model_trained_sac = MAPLE.load(args.load_checkpoint_path, env=env)
        model_sac.policy = deepcopy(model_trained_sac.policy)


    @timeout_decorator.timeout(args.time_limit)
    def run():
        print("Training Starts")
        model_sac.learn(total_timesteps=args.timesteps, callback=callback, log_interval=4)
        
    run()
    env.close()
    reward_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()


