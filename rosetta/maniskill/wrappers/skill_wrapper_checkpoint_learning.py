import gymnasium as gym
import sapien.physx as physx
#from rosetta.maniskill.utils.primitive_skills_cpu import PrimitiveSkillDelta
from rosetta.maniskill.primitive_skills.primitive_skills_cpu import PrimitiveSkillDelta
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

class SkillGymWrapperCheckpointLearning(SkillGymWrapper):
    def __init__(self, env: gym.Env,
                skill_indices,
                check_point=None,
                state_dict_buffer=None,
                add_simulator_info=False,
                record_dir=None,
                max_episode_steps=None,
                max_steps_per_video=None,
                video_fps: int = 30,
                **kwargs):
        super().__init__(env,skill_indices,record_dir,max_episode_steps,max_steps_per_video,video_fps,**kwargs)
        self.check_point=check_point
        self.state_dict_buffer=state_dict_buffer
        self.add_simulator_info=add_simulator_info

    def step(self, action):
        obs, reward, terminated, truncated, info=super().step(action,check_point=self.check_point)
        if self.check_point is not None:
            check_point_success=common.unbatch(common.to_numpy(self.env.unwrapped.evaluate_check_points()))[self.check_point]
            if check_point_success:
                terminated=True
            info['is_success']=check_point_success
            info['success']=check_point_success
        if self.add_simulator_info:
            info['simulator_info']=self.env.unwrapped.get_state_dict()
        return obs, reward, terminated, truncated, info

    def reset(self,seed=None, options=None):
        obs,info=super().reset(seed=seed, options=options)
        if self.state_dict_buffer is not None:
            state_dict=np.random.choice(self.state_dict_buffer)
            self.env.unwrapped.set_state_dict(state_dict)
            obs=self.env.unwrapped.get_obs()
            info=self.env.unwrapped.evaluate()
            return common.unbatch(common.to_numpy(obs),common.to_numpy(info))
        else:
            return obs,info
        

