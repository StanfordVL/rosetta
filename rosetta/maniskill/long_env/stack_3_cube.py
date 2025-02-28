from typing import Any, Dict, Union

import numpy as np
import torch

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import common
import time
from collections import defaultdict

@register_env("Stack3Cube", max_episode_steps=3e3)
class Stack3CubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]
    skill_config=None

    def __init__(self, stage=0,*args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.stage=stage
        self.cur_stage = 0
        self.cube_size = 0.04

        self.workspace_x=[-0.10, 0.15]
        self.workspace_y=[-0.15, 0.15]
        self.workspace_z=[0.01, 0.15]
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.reward_components = ["success","afford"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def task_skill_indices(self):
        return {
        0 : "pick",
        1 : "place",
        2 : "push",
    }

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.8, 0.8, 0.6], [0.14, +0.08, 0.12])
        return CameraConfig(
            "render_camera", pose=pose, width=2048, height=2048, fov=0.63, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.red_cube = actors.build_cube(
            self.scene, half_size=self.cube_size/2, color=[1, 0, 0, 1], name="red_cube"
        )
        self.green_cube = actors.build_cube(
            self.scene, half_size=self.cube_size/2, color=[0, 1, 0, 1], name="green_cube"
        )
        self.purple_cube = actors.build_cube(
            self.scene, half_size=self.cube_size/2, color=[1, 0, 1, 1], name="purple_cube"
        )
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            region =[[self.workspace_x[0],self.workspace_y[0]],[self.workspace_x[1],self.workspace_y[1]]]
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.02
            
            red_cube_xy = sampler.sample(radius, 100)
            green_cube_xy = sampler.sample(radius, 100, verbose=False)
            purple_cube_xy = sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = red_cube_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.red_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = green_cube_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.green_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = purple_cube_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.purple_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            self.object_list = {"red_cube": self.red_cube, 
                                "green_cube": self.green_cube,
                                "purple_cube": self.purple_cube}
    
    def _get_obs_info(self):
        info = {}
        for name in self.object_list:
            info[f"is_{name}_grasped"] = self.agent.is_grasping(self.object_list[name])[0]
            info[f"{name}_pos"] = self.object_list[name].pose.p[0]
        
        info["stage"] = self.cur_stage
        info["gripper_pos"] = self.agent.tcp.pose.p[0]
        return info

    def evaluate(self):
        info= self._get_obs_info()
        
        def stage0_success(info):
            return info["is_red_cube_grasped"]
        
        def stage1_success(info):
            red_not_grasped = ~info["is_red_cube_grasped"]
            red_on_green = (torch.linalg.norm(info["red_cube_pos"][:2] - info["green_cube_pos"][:2]) < self.cube_size/2) and (info["red_cube_pos"][2] > (info["green_cube_pos"][2] + self.cube_size/2))
            return (red_on_green and red_not_grasped)
        
        def stage2_success(info):
            return info["is_purple_cube_grasped"]
        
        def stage3_success(info):
            purple_not_grasped = ~info["is_purple_cube_grasped"]
            purple_on_red = (torch.linalg.norm(info["purple_cube_pos"][:2] - info["red_cube_pos"][:2]) < self.cube_size/2) and (info["purple_cube_pos"][2] > (info["red_cube_pos"][2] + self.cube_size/2))
            return purple_on_red and purple_not_grasped
        
        info["stage0_success"] = stage0_success(info)
        info["stage1_success"] = stage1_success(info)
        info["stage2_success"] = stage2_success(info)
        info["stage3_success"] = stage3_success(info)
        
        info["success"] = torch.tensor(False)
        if self.cur_stage==3:
            info["success"] = info["stage3_success"]
        
        return info

    def get_obs(self, info: Dict = None):
        if info is None:
            info = self.get_info()
        obs = []
        for name in self.object_list:
            obs += info[f"{name}_pos"].flatten().tolist()

        for name in self.object_list:
             obs += info[f"is_{name}_grasped"].flatten().tolist()
        
        obs += [self.cur_stage]
        return torch.tensor([obs], device = self.device, dtype = torch.float32)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros_like(info["success"],dtype=torch.float32,device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 80.0
    
    def skill_reward(self, prev_info, cur_info, action,**kwargs):
        #start=time.time()
        """
        prev_state: dict
        cur_state: dict
        action: np.array (4,)
            
        please note:
        prev_info, cur_info, action are all unbatched
        BUT any api call from env is batched, please unbatch them first before using!
        """
        reward_components = dict((k, 0.0) for k in self.reward_components)
        current_selected_action = np.argmax(action[:len(self.task_skill_indices.keys())])

        if current_selected_action in [0, 1]:
            current_selected_pos_1 = action[len(self.task_skill_indices.keys()):len(self.task_skill_indices.keys())+3]
            current_selected_pos_2 = None
        else:
            current_selected_pos_1 = action[len(self.task_skill_indices.keys()):len(self.task_skill_indices.keys())+3]
            current_selected_pos_2 = action[len(self.task_skill_indices.keys())+3:len(self.task_skill_indices.keys())+6]

        def stage_0_reward():
            target_action = 0
            target_pos_1 = prev_info["red_cube_pos"].copy()
            target_pos_2 = None
            if current_selected_action==target_action:
                reward = - np.tanh(np.linalg.norm(current_selected_pos_1-target_pos_1))
                reward_components["afford"] = (1 + reward) * 5
            if cur_info["stage0_success"]:
                reward_components["success"] = 10
            return reward_components
        
        def stage_1_reward():
            target_action = 1
            target_pos_1 = prev_info["green_cube_pos"] + 0.04
            target_pos_2 = None
            if current_selected_action==target_action:
                reward = -np.tanh(np.linalg.norm(current_selected_pos_1-target_pos_1))
                reward_components["afford"] = (1 + reward) * 5
            if cur_info["stage1_success"]:
                reward_components["success"] = 10
            return reward_components
            
            
        def stage_2_reward():
            target_action = 0
            target_pos_1 = prev_info["purple_cube_pos"]
            target_pos_2 = None
            if current_selected_action == target_action:
                reward = - np.tanh(np.linalg.norm(current_selected_pos_1 - target_pos_1))
                reward_components["afford"] = (1 + reward) * 5
            if cur_info["stage2_success"]:
                reward_components["success"] = 10
            return reward_components

        def stage_3_reward():
            target_action = 1
            target_pos_1 = prev_info["red_cube_pos"] + 0.04
            target_pos_2 = None
            if current_selected_action == target_action:
                reward = - np.tanh(np.linalg.norm(current_selected_pos_1 - target_pos_1))
                reward_components["afford"] = (1 + reward) * 5
            if cur_info["stage3_success"]:
                reward_components["success"] = 10
            return reward_components

        if self.cur_stage==0:
            reward = stage_0_reward()
        elif self.cur_stage==1:
            reward = stage_1_reward()
        elif self.cur_stage==2:
            reward = stage_2_reward()
        elif self.cur_stage==3:
            reward = stage_3_reward()
        
        cur_before = self.cur_stage
        # move to next stage if success
        if (self.cur_stage == 0) and cur_info["stage0_success"]:
            self.cur_stage = 1
        elif (self.cur_stage == 1) and cur_info["stage1_success"]:
            self.cur_stage = 2
        elif (self.cur_stage == 2) and cur_info["stage2_success"]:
            self.cur_stage = 3
        elif (self.cur_stage == 3) and cur_info["stage3_success"]:
            self.cur_stage = 4
        
        print("stage:",cur_before, "->", self.cur_stage)
        return reward



    def reset(self, **kwargs):
        self.cur_stage = 0
        return super().reset(**kwargs)