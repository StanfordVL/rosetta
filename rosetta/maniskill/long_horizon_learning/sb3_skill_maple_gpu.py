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
from stable_baselines3.maple.maple import MAPLE as MAPLE_YES_FAIL
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EveryNTimesteps
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from feedback_to_reward.maniskill.customized_tasks import *
from feedback_to_reward.maniskill.wrappers.skill_wrapper import SkillGymWrapper
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from feedback_to_reward.maniskill.wrappers.skill_wrapper import SkillGymWrapper
from feedback_to_reward.maniskill.utils.utils import get_task_env_source, get_avail_save_path
from mani_skill.envs.sapien_env import BaseEnv
from feedback_to_reward.maniskill.pipelines.reward_manipulator import RewardManipulator
from stable_baselines3.common.noise import NormalActionNoise
import time
from stable_baselines3.maple.policies import MAPLEPolicy
from feedback_to_reward.sb3.callbacks.reward_callback import RewardEvalCallback
from feedback_to_reward.sb3.callbacks.init_callback import EvalCallbackAtStart
import feedback_to_reward.maniskill.curriculum_learning.customized_tasks
from feedback_to_reward.maniskill.wrappers.sb3_vec_skill_wrapper import SB3VecSkillWrapper
from stable_baselines3.maple.maple_no_fail import MAPLE_NO_FAIL
import time
import timeout_decorator
from feedback_to_reward.sb3.callbacks.early_stop_callback import EarlyStoppingEvalCallback

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
    net_arch_depth: int = 2
    net_arch_width: int = 256
    learning_starts: int = 100000 # low-level steps
    
    # training config
    save_path: str = './test'
    auto_save_path: bool = False
    save_checkpoint: bool = False
    max_episode_steps: int = 6
    max_steps_per_video: int =6
    num_envs: int = 32
    eval_step: int = int(2e5)
    num_eval_envs: int = 10
    reward_json_path: str = None
    stage: int=0
    load_checkpoint_path: str = None
    time_limit: int = 144000
    early_stop_threshold: float = 0.9
    stuck_penalty: bool = True
    reject_fail: bool = True


def main():
    args = tyro.cli(MAPLEConfig)
    # Generate an index for save_path to prevent overwriting
    if args.auto_save_path:
        args.save_path=os.path.join(args.save_path, args.env_id)
        args.save_path = get_avail_save_path(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    
    # Copy reward function if provided
    if args.reward_json_path is not None:
        os.system(f"cp {args.reward_json_path} {args.save_path}")
            
    print(args)
    # Save arguments
    with open(os.path.join(args.save_path, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    
    eval_env = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        sim_backend="gpu",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        stage=args.stage
    )
    
    if args.reward_json_path is not None:
        reward_manipulator = RewardManipulator(env=eval_env, json_path=args.reward_json_path)
        reward_manipulator.change_function(['evaluate', 'skill_reward'])
    
    train_env = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        sim_backend="gpu",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        stage=args.stage
    )
    source_code = get_task_env_source(train_env, BaseEnv)
    with open(os.path.join(args.save_path, "env_source.py"), "w") as f:
        f.write(source_code)
    train_env = SB3VecSkillWrapper(
        env=train_env,
        skill_indices=train_env.base_env.task_skill_indices,
        stuck_penalty=args.stuck_penalty,
        max_episode_length=args.max_episode_steps
    )
    reward_env = SB3VecSkillWrapper(
        env=eval_env,
        skill_indices=eval_env.base_env.task_skill_indices,
        stuck_penalty=args.stuck_penalty,
        max_episode_length=args.max_episode_steps
    )
    eval_env = SB3VecSkillWrapper(
        env=eval_env,
        skill_indices=eval_env.base_env.task_skill_indices,
        record_dir=None,
        video_length=args.max_steps_per_video,
        video_triggers=1,
        stuck_penalty=args.stuck_penalty,
        max_episode_length=args.max_episode_steps
    )
    
    train_env.reset()
    eval_env.reset()
    
    # Early stopping callback based on evaluation success rate
    eval_callback = EarlyStoppingEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_path, 'eval', 'best_model'),
        log_path=os.path.join(args.save_path, 'eval', 'log'),
        n_eval_episodes=5,
        eval_freq=1,
        deterministic=True,
        render=False,
        early_stop_threshold=args.early_stop_threshold
    )
    eval_callback = EveryNTimesteps(n_steps=args.eval_step, callback=eval_callback)
    callback = [eval_callback]

    # Optionally add reward evaluation callback
    reward_event = RewardEvalCallback(
        reward_env,
        eval_freq=1,
        n_eval_episodes=5,
    )
    reward_callback = EveryNTimesteps(n_steps=args.eval_step, callback=reward_event)
    callback.append(reward_callback)

    # Optionally add checkpoint callback
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
    MAPLE = MAPLE_NO_FAIL if args.reject_fail else MAPLE_YES_FAIL
    
    if args.add_noise:
        action_noise = NormalActionNoise(
            mean=args.noise_mean * np.ones(train_env.action_space.shape),
            sigma=args.noise_sigma * np.ones(train_env.action_space.shape)
        )
        model_sac = MAPLE(
            "MlpPolicy",
            train_env,
            policy_kwargs=dict(net_arch=net_arch),
            action_dim_s=len(train_env.base_env.task_skill_indices.keys()),
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
            train_env,
            policy_kwargs=dict(net_arch=net_arch),
            action_dim_s=len(train_env.base_env.task_skill_indices.keys()),
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
        saved_model = MAPLE.load(args.load_checkpoint_path, env=train_env)
        # Extract the policy weights from the loaded model
        policy = saved_model.policy
        model_sac.policy.load_state_dict(policy.state_dict())
    
    @timeout_decorator.timeout(args.time_limit)
    def run():
        model_sac.learn(total_timesteps=args.timesteps, callback=callback, log_interval=4)
        
    run()
    train_env.close()
    eval_env.close()
    

if __name__ == "__main__":
    main()
