import gymnasium as gym
import sapien.physx as physx
import numpy as np
from mani_skill.utils import common
import sys
from mani_skill.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
    tile_images,
)
from copy import deepcopy
import os
from mani_skill.envs.sapien_env import BaseEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv as SB3VecEnv
import time
from typing import Any, List, Optional, Type, Union
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnv as SB3VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from feedback_to_reward.maniskill.primitive_skills.primitive_skills_gpu import PrimitiveSkillDelta
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
import time

def select_index_from_dict(data: Union[dict, np.ndarray], i: int):
    if isinstance(data, np.ndarray):
        return data[i]
    elif isinstance(data, dict):
        out = dict()
        for k, v in data.items():
            out[k] = select_index_from_dict(v, i)
        return out
    else:
        return data[i]

def convert_info(infos,search_key="success"):
    # Step 1: Convert all tensors in the dictionary to NumPy arrays
    numpy_dict = {}
    for key, value in infos.items():
        if isinstance(value, torch.Tensor):
            numpy_dict[key] = value.cpu().numpy()
        else:
            numpy_dict[key] = value  # If any non-tensor values, just copy them
    
    # Step 2: Convert the dictionary of NumPy arrays to a list of dicts
    num_entries = numpy_dict[search_key].shape[0]  # Assuming all values have the same batch size
    result_list = []
    
    for i in range(num_entries):
        entry_dict = {}
        for key, value in numpy_dict.items():
            entry_dict[key] = value[i]  # Take the i-th entry for each key
        result_list.append(entry_dict)
    
    return result_list

class SB3VecSkillWrapper(SB3VecEnv):
    def __init__(self, env: BaseEnv,
                 skill_indices,
                 is_params_scaled=True,
                 stuck_penalty=True,
                 max_episode_length=10,
                 verbose=0,
                 record_dir=None,
                 video_fps=30,
                 video_length=10,
                 video_triggers=1,
                 **kwargs):
        self._env = env
        self._last_seed = None
        self.skill_indices = skill_indices
        self.num_envs = self._env.num_envs
        self.is_params_scaled = is_params_scaled
        if env.skill_config is None:
            skill_config={}
        else:
            skill_config=env.skill_config
        kwargs.update(skill_config)
        self.primitive_skills = [PrimitiveSkillDelta(skill_indices=skill_indices, **kwargs) for _ in range(self.num_envs)]
        self.stuck_penalty = stuck_penalty
        self.max_episode_length = max_episode_length
        low = -np.ones(self.primitive_skills[0].n_skills + self.primitive_skills[0].max_num_params)
        high = np.ones(self.primitive_skills[0].n_skills + self.primitive_skills[0].max_num_params)
        super().__init__(
            env.num_envs, env.single_observation_space, gym.spaces.Box(low=low, high=high, dtype=np.float32)
        )
        self.episode_lengths=np.zeros(self.num_envs)
        self.episode_returns=np.zeros(self.num_envs)
        self.verbose=verbose
        self.total_action_time=[]
        self.total_step_time = []
        self.total_simulation_time= []
        
        self.record_dir = record_dir
        self.video_fps = video_fps 
        self.image_buffer = [[] for _ in range(self.num_envs)]
        self._video_id = 0
        self._video_steps = 0
        self._total_steps = 0
        self.video_length=video_length
        self.video_triggers=video_triggers
        self.start_recording = False
        
    def flush_video(self):
        output_dir=os.path.join(self.record_dir, f"video_{self._video_id}")
        os.makedirs(output_dir, exist_ok=True)
        for i in range(self.num_envs):
            images_to_video(images=self.image_buffer[i],
                            output_dir=output_dir,
                            video_name=f"video_{self._video_id}_{i}",
                            fps=self.video_fps)
        self.image_buffer = [[] for _ in range(self.num_envs)]
        self._video_id+=1
        self.start_recording=False
        self._video_steps = 0
        
        
    @property
    def base_env(self) -> BaseEnv:
        return self._env.unwrapped
    
    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        self._last_seed = seed
        
    def reset(self) -> VecEnvObs:
        obs = self._env.reset(seed=self._last_seed)[0]
        self._last_seed = None  # use seed from call to seed() once
        
        for i in range(self.num_envs):
            self.primitive_skills[i].reset()
        self.prev_info=[None]*self.num_envs
        cur_info=self._env.evaluate()
        self.cur_info=convert_info(cur_info)
        self.episode_lengths=np.zeros(self.num_envs)
        self.episode_returns=np.zeros(self.num_envs)
        return obs.cpu().numpy()  # currently SB3 only support numpy arrays
    
    def _scale_params(self, actions: np.ndarray) -> np.ndarray:
        """
        Scales normalized parameters from [-1, 1] to actual workspace ranges for a batch of actions.
        The first dimension of the 'actions' array is the batch size.
        """
        actions = actions.copy()
        num_skills = self.primitive_skills[0].n_skills
        
        workspace_x = self.base_env.workspace_x  
        workspace_y = self.base_env.workspace_y
        workspace_z = self.base_env.workspace_z

        actions[:, num_skills] = ((actions[:, num_skills] + 1) / 2) * (workspace_x[1] - workspace_x[0]) + workspace_x[0]
        actions[:, num_skills + 1] = ((actions[:, num_skills + 1] + 1) / 2) * (workspace_y[1] - workspace_y[0]) + workspace_y[0]
        actions[:, num_skills + 2] = ((actions[:, num_skills + 2] + 1) / 2) * (workspace_z[1] - workspace_z[0]) + workspace_z[0]

        return actions
    
    def step_async(self, actions: np.ndarray) -> None:
        self.cur_actions = actions
        
    def step_wait(self) -> VecEnvStepReturn:
        if self._total_steps % self.video_triggers == 0 and self.record_dir is not None:
            self.start_recording = True
        step_time = time.time()
        dones = np.array([False]*self.num_envs)
        actions=self.cur_actions
        self.prev_info=self.cur_info
        if self.is_params_scaled:
            actions = self._scale_params(actions)
        skill_dones = np.array([False]*self.num_envs)
        skill_stucks = np.array([False]*self.num_envs)
        truncations = np.array([False]*self.num_envs)
        terminations = np.array([False]*self.num_envs)
        skill_finisheds=skill_dones | skill_stucks | truncations | terminations
        num_timesteps=0
        while not all(skill_finisheds):
            num_timesteps+=1
            osc_actions=[]
            action_time=time.time()
            if self.start_recording:
                render_images=self.render()
            for i in range(self.num_envs):
                if self.start_recording:
                    self.image_buffer[i].append(render_images[i])
                if skill_finisheds[i]:
                    osc_actions.append(self.primitive_skills[i].get_empty_action().get("action"))
                else:
                    eef_state={
                        "robot0_eef_pos":common.to_numpy(self.base_env.agent.tcp.pose.p)[i],
                        "robot0_eef_quat":common.to_numpy(self.base_env.unwrapped.agent.tcp.pose.q)[i],
                    }
                    #print('action_len:',len(actions),"env_num:",self.num_envs)
                    rst=self.primitive_skills[i].get_action(actions[i],eef_state)
                    osc_action=rst.get("action")
                    skill_done=rst.get("skill_done")
                    skill_success=rst.get("skill_success")
                    skill_stuck=rst.get("skill_stuck",False)
                    if skill_stuck:
                        print("Skill Stuck")
                    skill_dones[i]=skill_done or skill_dones[i]
                    skill_stucks[i]=skill_stuck or skill_stucks[i]
                    osc_actions.append(osc_action)
            # calculate action time in seconds
            if self.verbose:
                self.total_action_time.append(time.time()-action_time)
            #print(osc_actions)
            osc_actions = np.array(osc_actions)
            osc_actions=common.to_tensor(osc_actions)
            simulation_time=time.time()
            vec_obs, _ , terminations_cur, truncations_cur, infos = self._env.step(
                osc_actions
            )
            if self.verbose:
                self.total_simulation_time.append(time.time()-simulation_time)
            terminations_cur = terminations_cur.cpu().numpy()
            terminations = terminations_cur | terminations
            #truncations_cur = truncations_cur.cpu().numpy()
            #truncations = truncations_cur | truncations
            skill_finisheds=skill_dones | skill_stucks | truncations | terminations | skill_finisheds
        
        vec_obs = vec_obs.cpu().numpy()
        new_infos = convert_info(infos)
        
        #0 update episode step
        self.episode_lengths+=1
        
        #1 get reward
        for i in range(self.num_envs):
            new_infos[i]["reward_components"]={}
            rew=self.base_env.skill_reward(prev_info=self.prev_info[i], cur_info=new_infos[i], action=self.cur_actions[i])
            new_infos[i]["is_fail"]=self.base_env.task_fail(info=new_infos[i])
            if isinstance(rew, dict):
                for k,v in rew.items():
                    new_infos[i]["reward_components"][k]=v
            else:
                new_infos[i]["reward_components"]["reward"]=rew
                
            if skill_stucks[i]:
                new_infos[i]["is_fail"]=True
                
            if new_infos[i].get("is_fail",False):
                if isinstance(self.stuck_penalty, bool) and self.stuck_penalty:
                    new_infos[i]["reward_components"]["stuck"]=self.episode_lengths[i]-self.max_episode_length # penalize the whole trajectory
                elif isinstance(self.stuck_penalty, (int, float)):
                    new_infos[i]["reward_components"]["stuck"]=self.stuck_penalty
        
        rewards = np.array([np.sum(list(info["reward_components"].values())) for info in new_infos])
            
        
        #2. update episode step and reward
        self.episode_returns+=rewards
        
        #3. update truncation
        for i in range(self.num_envs):
            if self.episode_lengths[i]>=self.max_episode_length:
                truncations[i]=True
        
        
        #3. update info
        for i in range(self.num_envs):
            new_infos[i]["episode"]={
                "r":self.episode_returns[i],
                "l":self.episode_lengths[i]
            }
            new_infos[i]["TimeLimit.truncated"]=truncations[i] and not terminations[i]
            new_infos[i]["is_success"]=new_infos[i].get("success",False)
            new_infos[i]["num_timesteps"]=num_timesteps
        
        #dones=truncations | terminations | skill_stucks 
        dones=truncations | terminations
        for i in range(self.num_envs):
            if new_infos[i]["is_fail"]:
                dones[i]=True
        
        #3. reset env if done
        for i, done in enumerate(dones):
            if done:
                new_infos[i]["terminal_observation"] = select_index_from_dict(
                    vec_obs, i
                )
                
        if dones.any():
            reset_indices = np.where(dones)[0]
            new_obs = self._env.reset(options=dict(env_idx=reset_indices))[0]
            vec_obs[reset_indices] = new_obs[reset_indices].cpu().numpy()
            self.episode_returns[reset_indices] = 0
            self.episode_lengths[reset_indices] = 0
            self.cur_info=convert_info(self._env.evaluate())
            for i in reset_indices:
                self.primitive_skills[i].reset()
                self.prev_info[i]=None
        else:
            self.cur_info=new_infos
        if self.verbose:
            self.total_step_time.append(time.time()-step_time)
            
        self._total_steps+=1
        if self.start_recording:
            self._video_steps+=1
        if self._video_steps>=self.video_length and self.record_dir is not None:
            self.flush_video()
        return vec_obs, rewards, dones, new_infos
            
    

    def close(self) -> None:
        return self._env.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        indices=self._get_indices(indices)
        return [getattr(self._env, attr_name) for i in indices]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        setattr(self._env, attr_name, value)

    def render(self):
        return common.to_numpy(self.base_env.render())
    
    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        return self._env.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        return [False] * self.num_envs