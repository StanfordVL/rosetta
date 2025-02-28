import gymnasium as gym
import sapien.physx as physx
#from feedback_to_reward.maniskill.utils.primitive_skills_cpu import PrimitiveSkillDelta
from feedback_to_reward.maniskill.primitive_skills.primitive_skills_cpu import PrimitiveSkillDelta
import numpy as np
from mani_skill.utils import common
import sys
from mani_skill.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
    tile_images,
)
from mani_skill.utils import common, gym_utils
from copy import deepcopy
import os
from .skill_wrapper import SkillGymWrapper
import numpy as np
from mani_skill.utils import common

class SkillGymWrapperBufferOnly(SkillGymWrapper):
    def __init__(self, env: gym.Env,
                skill_indices,
                use_check_point=True,
                record_dir=None,
                max_episode_steps=None,
                max_steps_per_video=None,
                video_fps: int = 30,
                max_buffer_size=100,
                **kwargs):
        super().__init__(env,skill_indices,record_dir,max_episode_steps,max_steps_per_video,video_fps,**kwargs)
        self.use_check_point=use_check_point
        self.state_dict_buffer=None
        if self.use_check_point:
            self.state_dict_buffer=[[] for _ in range(len(env.evaluate_check_points().keys())-1)]
            self.max_buffer_size=max_buffer_size

    def step(self, action):
        obs, reward, terminated, truncated, info=super().step(action)
        if self.use_check_point:
            check_points_status=common.unbatch(common.to_numpy(self.env.unwrapped.evaluate_check_points()))
            for i in range(len(check_points_status)-2,-1,-1):
                if check_points_status[i]:
                    state_dict=self.env.unwrapped.get_state_dict()
                    if len(self.state_dict_buffer[i])<self.max_buffer_size:
                        self.state_dict_buffer[i].append(state_dict)
                    else:
                        self.state_dict_buffer[i][np.random.randint(self.max_buffer_size)]=state_dict
                    break
        return obs, reward, terminated, truncated, info

    def reset(self,seed=None, options=None):
        obs,info=super().reset(seed=seed, options=options)
        if self.state_dict_buffer is not None:
            not_empty_buffer=[-1]
            for i in range(len(self.state_dict_buffer)):
                if len(self.state_dict_buffer[i])>0:
                    not_empty_buffer.append(i)
            buffer_idx=np.random.choice(not_empty_buffer)
            
            if buffer_idx==-1:
                 return obs,info
            else:
                idx=np.random.randint(len(self.state_dict_buffer[buffer_idx]))
                self.env.unwrapped.set_state_dict(self.state_dict_buffer[buffer_idx][idx])
                obs=self.env.unwrapped.get_obs()
                info=self.env.unwrapped.evaluate()
                return common.unbatch(common.to_numpy(obs),common.to_numpy(info))
        else:
            return obs,info
        

