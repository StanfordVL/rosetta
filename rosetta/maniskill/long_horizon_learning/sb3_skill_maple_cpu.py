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
from rosetta.maniskill.wrappers.skill_wrapper import SkillGymWrapper
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from rosetta.maniskill.wrappers.skill_wrapper import SkillGymWrapper
from rosetta.maniskill.utils.utils import get_task_env_source,get_avail_save_path,update_config_from_dict,load_default_config
from mani_skill.envs.sapien_env import BaseEnv
from rosetta.maniskill.pipelines.reward_manipulator import RewardManipulator
from stable_baselines3.common.noise import NormalActionNoise
import time
from stable_baselines3.maple.policies import MAPLEPolicy
from rosetta.sb3.callbacks.reward_callback import RewardEvalCallback
import timeout_decorator
from rosetta.sb3.callbacks.early_stop_callback import EarlyStoppingEvalCallback
@dataclass
class MAPLEConfig:
    # env config
    env_id: str = "Align3CubeCurLearning"
    timesteps: int = int(2e8)
    
    # algorithm config
    learning_rate: float = 3e-3
    gamma: float = 0.8
    tau: float = 0.005
    batch_size: int = 256
    gsteps: int = 8
    use_aff: bool = False
    add_noise: bool = False
    noise_mean: float = 0.0
    noise_sigma: float = 0.1
    net_arch_depth: int = 4
    net_arch_width: int = 256
    learning_starts: int = 0 # 50000 # low-level steps
    
    # training config
    default_config: str = None
    save_path: str = './test'
    auto_save_path: bool = False
    save_checkpoint: bool = False
    max_episode_steps: int = 6
    max_steps_per_video: int =6
    num_envs: int = 32
    eval_step: int = int(3e5)
    reward_json_path: str = None
    stage: int=0
    load_checkpoint_path: str = None
    time_limit: int = 144000
    early_stop_threshold: float = 0.9

def make_env(env_id: str,record_dir=None,max_episode_steps=5,max_steps_per_video=5,reward_json_path=None,stage=0):
    if record_dir is not None:
        os.makedirs(record_dir, exist_ok=True)
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

def main():
    args = tyro.cli(MAPLEConfig)
    
    if args.default_config is not None:
        try:
            default_config = load_default_config(args.default_config)
            args = update_config_from_dict(args, default_config)
            print(f"Loaded default configuration from: {args.default_config}")
        except Exception as e:
            print(f"Error loading default config: {str(e)}")
            return 
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
    with open(os.path.join(args.save_path,"args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    env=gym.make(args.env_id, num_envs=1,sim_backend="cpu",
                 obs_mode="state_dict", control_mode="pd_ee_delta_pose", render_mode="rgb_array",)
    task_skill_indices=env.unwrapped.task_skill_indices
    source_code=get_task_env_source(env, BaseEnv)
    with open(os.path.join(args.save_path,"env_source.py"), "w") as f:
        f.write(source_code)
    env.close()
    # create one eval environment
    eval_env = SubprocVecEnv([make_env(args.env_id, record_dir=os.path.join(args.save_path, 'eval'),
                                       max_episode_steps=args.max_episode_steps,max_steps_per_video=args.max_steps_per_video,
                                       reward_json_path=args.reward_json_path,stage=args.stage) for i in range(1)])
    reward_env = SubprocVecEnv([make_env(args.env_id,
                                       max_episode_steps=args.max_episode_steps,
                                       reward_json_path=args.reward_json_path,stage=args.stage) for i in range(1)])
    eval_env.seed(0)
    eval_env.reset()
    
    
    # create num_envs training environments
    env = SubprocVecEnv([make_env(args.env_id, 
                                  max_episode_steps=args.max_episode_steps,
                                  reward_json_path=args.reward_json_path,
                                  stage=args.stage) for i in range(args.num_envs)])
    env.seed(0)
    env.reset()
    
    eval_callback = EarlyStoppingEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_path,'eval', 'best_model'),
        log_path=os.path.join(args.save_path,'eval', 'log'),
        n_eval_episodes=5,
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
        n_eval_episodes=5,
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
        
    net_arch = [args.net_arch_width] * args.net_arch_depth
    if args.add_noise:
        action_noise = NormalActionNoise(mean=args.noise_mean* np.ones(env.action_space.shape), sigma=args.noise_sigma * np.ones(env.action_space.shape))
        model_sac = MAPLE(
            "MlpPolicy",
            env,
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
            action_noise=action_noise,
        )
    else:
        model_sac = MAPLE(
            "MlpPolicy",
            env,
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
        
    if args.load_checkpoint_path is not None:
        print(f"Loading model from checkpoint: {args.load_checkpoint_path}")
        
        # Load the saved model (to access weights)
        saved_model = MAPLE.load(args.load_checkpoint_path, env=env)
        
        # Extract the policy weights from the loaded model
        policy = saved_model.policy
        model_sac.policy.load_state_dict(policy.state_dict())
    
    print("Training model")
    @timeout_decorator.timeout(args.time_limit)
    def run():
        model_sac.learn(total_timesteps=args.timesteps, callback=callback, log_interval=4)
        
    run()
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
