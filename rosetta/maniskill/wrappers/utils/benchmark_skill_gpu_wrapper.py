import os
import gymnasium as gym
from stable_baselines3 import PPO  # You can use your trained model
from stable_baselines3.common.monitor import Monitor
from feedback_to_reward.maniskill.wrappers.skill_gpu_sb3_wrapper import SkillGPUSB3Wrapper
import feedback_to_reward.maniskill.curriculum_learning.customized_tasks


def traverse_env_num(env_id,num_envs,n_eval_episodes,episode_steps):
    import numpy as np
    env_kwargs = dict(obs_mode="state", control_mode="pd_ee_delta_pose", render_mode="rgb_array", sim_backend="gpu")
    env = gym.make(env_id, num_envs=num_envs, enable_shadow=True,**env_kwargs)
    env=SkillGPUSB3Wrapper(env,
            skill_indices={
                0:'pick',
                1:"place"
            },
        )
    for episode in range(n_eval_episodes):
        obs = env.reset()
        total_reward = 0
        step=0
        for step in range(episode_steps):
            obs, reward, done, info = env.step(np.stack([env.action_space.sample() for i in range(num_envs)]))  # Take a step in the environment
            step+=1
            print(f"Step {step}: done: {done}")
            total_reward += reward
    rst= {
        'step_time':sum(env.total_step_time)/len(env.total_step_time),
        'simulation_time':sum(env.total_simulation_time)/len(env.total_simulation_time),
        'low_level_step_per_skill':len(env.total_simulation_time)/(episode_steps*n_eval_episodes)
    }
    env.close()
    return rst
    
import json
def main():
    data_dict={}
    for i in [32,64,128,256,512,1024]:
        data_dict[i]=traverse_env_num("Align3CubeCurLearning",i,2,10)
        with open('benchmark_skill_gpu_wrapper.json', 'w') as f:
            json.dump(data_dict, f, indent=4)
            
if __name__ == "__main__":
    main()