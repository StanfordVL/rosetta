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
import sapien
import time
from transforms3d.euler import euler2quat
from collections import defaultdict

@register_env("ObjectsAndBins", max_episode_steps=2e3)
class ObjectsAndBinsEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]
    skill_config=None
    baseball_id = "055_baseball"
    tennis_ball_id = "056_tennis_ball"

    apple_id = "013_apple"
    orange_id = "017_orange"

    bin_base_half_len = 0.10 # half side length of the bin's bottom block
    bin_base_half_height = 0.005 # half height of the bin's bottom block
    bin_wall_half_thickness = 0.005 # half thickness of the bin's walls
    bin_wall_half_height = 0.01
    bin_wall_half_length = 2 * bin_wall_half_thickness + bin_base_half_len
    # bin_wall_half_size = [bin_wall_half_thickness, 2*bin_wall_half_thickness+bin_base_half_len, 2*bin_wall_half_thickness] # The wall block size of the bin: the list represents the half length of the block along the [x, y, z] axis respectively.
    bin_wall_half_size = [bin_wall_half_thickness, bin_base_half_len, 2*bin_wall_half_height] # The wall block size of the bin: the list represents the half length of the block along the [x, y, z] axis respectively.
    

    def __init__(self, stage=0,*args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.stage=stage
        self.cur_stage=0
        self.workspace_x=[-0.3, 0.1] # close to robot, further from robot
        self.workspace_y=[-0.3, 0.3] # left of robot, right of robot
        self.workspace_z=[0.01, 0.25]
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
        pose = sapien_utils.look_at(eye = [0.5, 0.2, 0.5], target = [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 2048, 2048, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        builder1 = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{self.baseball_id}",
        )
        self.baseball= builder1.build_dynamic(name=f"baseball")
        
        builder2 = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{self.tennis_ball_id}",
        )
        self.tennis_ball= builder2.build_dynamic(name=f"tennis_ball")

        builder3 = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{self.apple_id}",
        )
        self.apple= builder3.build_dynamic(name=f"apple")

        builder4 = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{self.orange_id}",
        )
        self.orange= builder4.build_dynamic(name=f"orange")

        self.blue_bin = self._build_bin('blue_bin', [0.5, 0.7, 1, 1])
        self.white_bin = self._build_bin('white_bin', [1, 1, 1, 1])

        self.object_list = {"baseball": self.baseball,
                            "tennis_ball": self.tennis_ball,
                            "apple": self.apple,
                            "orange": self.orange,
                            "blue_bin": self.blue_bin,
                            "white_bin": self.white_bin}

        
    def _build_bin(self, name, base_color=[0, 0, 0, 1]):
        builder = self.scene.create_actor_builder()
        
        # init the locations of the basic blocks
        dx = self.bin_base_half_len - self.bin_base_half_height 
        dy = self.bin_base_half_len - self.bin_base_half_height 
        dz = self.bin_wall_half_size[2] + self.bin_base_half_height

        
        # build the bin bottom and edge blocks
        poses = [
            sapien.Pose([0, 0, 0]),     # base at pose
            sapien.Pose([-dx, 0, dz]),  # back wall on top of base
            sapien.Pose([dx, 0, dz]),   # front wall on top of base
            sapien.Pose([0, -dy, dz]),  # side wall on top of base
            sapien.Pose([0, dy, dz]),   # other side wall on top of base
        ]
        half_sizes = [
            [self.bin_base_half_len, self.bin_base_half_len, self.bin_base_half_height],
            self.bin_wall_half_size,
            self.bin_wall_half_size,
            [self.bin_wall_half_size[1], self.bin_wall_half_size[0], self.bin_wall_half_size[2]],
            [self.bin_wall_half_size[1], self.bin_wall_half_size[0], self.bin_wall_half_size[2]],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size,
                material=sapien.render.RenderMaterial(
                    base_color=base_color,
                )
            )

      	# build the kinematic bin
        return builder.build_kinematic(name=name)
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Place cubes at random positions
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            region = [[-0.2, 0], [0.05, 0.2]] # object on the right side
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.02

            baseball_xy = sampler.sample(radius, 100)
            tennis_ball_xy = sampler.sample(radius, 100, verbose=False)
            apple_xy = sampler.sample(radius, 100, verbose=False)
            orange_xy = sampler.sample(radius, 100, verbose=False)

            # Set initial positions for objects
            xyz[:, :2] = baseball_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.baseball.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = tennis_ball_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.tennis_ball.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = apple_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.apple.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = orange_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.orange.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # init bin1 in the last 1/2 zone along the x-axis (so that it doesn't collide with the sphere)
            # and in the first 1/2 zone along the y-axis (so it doesn't collide with bin2)
            q = [1, 0, 0, 0]
            pos1 = torch.zeros((b, 3))
            pos1[:, 0] = -0.25
            pos1[:, 1] = -0.2
            pos1[:, 2] = self.bin_base_half_height  # on the table
            bin_pose1 = Pose.create_from_pq(p=pos1, q=q)
            self.blue_bin.set_pose(bin_pose1)

            # init bin2 in the last 1/2 zone along the x-axis and make sure it does not collide with bin1
            pos2 = torch.zeros((b, 3))
            pos2[:, 0] = 0   # slightly offset from bin1 along the x-axis 
            pos2[:, 1] = -0.2   # fixed distance from bin1 along the y-axis 
            pos2[:, 2] = self.bin_base_half_height  # on the table
            bin_pose2 = Pose.create_from_pq(p=pos2, q=q)
            self.white_bin.set_pose(bin_pose2)


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
    
    def object_in_bin(self, object_name, bin_name):
        object_pos = self.object_list[object_name].pose.p[0]
        bin_pos = self.object_list[bin_name].pose.p[0]
        object_not_grasped = ~self.agent.is_grasping(self.object_list[object_name])[0]
        return (np.abs(object_pos[0] - bin_pos[0]) < self.bin_base_half_len) and (np.abs(object_pos[1] - bin_pos[1]) < self.bin_base_half_len) \
             and (object_pos[2] < 0.12) and object_not_grasped
    

    def _get_obs_info(self):
        info= {}
        for name in self.object_list:
            info[f"is_{name}_grasped"] = self.agent.is_grasping(self.object_list[name])[0]
            info[f"{name}_pos"] = self.object_list[name].pose.p[0]
        
        info["stage"] = self.cur_stage
        info["gripper_pos"] = self.agent.tcp.pose.p[0]
        return info


    def evaluate(self):
        info = self._get_obs_info()
        
        def stage0_success(info):
            return info["is_baseball_grasped"]
        
        def stage1_success(info):
            baseball_in_bin = self.object_in_bin("baseball", "white_bin")
            return ~info["is_baseball_grasped"] and baseball_in_bin

        def stage2_success(info):
            return info["is_tennis_ball_grasped"] and stage1_success(info)
        
        def stage3_success(info):
            tennis_ball_in_bin = self.object_in_bin("tennis_ball", "white_bin")
            return ~info["is_tennis_ball_grasped"] and tennis_ball_in_bin and stage1_success(info)

        info["stage0_success"] = stage0_success(info)
        info["stage1_success"] = stage1_success(info)
        info["stage2_success"] = stage2_success(info)
        info["stage3_success"] = stage3_success(info)

        info["success"] = torch.tensor(False)

        if self.cur_stage==3:
            info["success"] = torch.tensor(info["stage3_success"])

        return info

    def skill_reward(self, prev_info, cur_info, action, **kwargs):
        reward_components = defaultdict(float)
        predicted_action = np.argmax(action[:len(self.task_skill_indices.keys())])
        predicted_pos = action[len(self.task_skill_indices.keys()):len(self.task_skill_indices.keys())+3]

        def stage_0_reward():
            target_action = 0  # pick
            target_pos = cur_info["baseball_pos"]
            
            if predicted_action == target_action:
                distance = np.linalg.norm(predicted_pos - target_pos)
                reward = -np.tanh(distance)
                reward_components["afford"] = (1 + reward) * 5
            if cur_info["stage0_success"]:
                reward_components["success"] = 10.0
            return reward_components

        def stage_1_reward():
            target_action = 1
            target_pos = np.array([0.15, -0.3,  0.2])
            if predicted_action == target_action:
                distance = np.linalg.norm(predicted_pos - target_pos)
                reward = -np.tanh(distance)
                reward_components["afford"] = (1 + reward) * 5
            if cur_info["stage1_success"]:
                reward_components["success"] = 10.0
            return reward_components
        
        def stage_2_reward():
            target_action = 0
            target_pos = cur_info["tennis_ball_pos"]
            if predicted_action == target_action:
                distance = np.linalg.norm(predicted_pos - target_pos)
                reward = -np.tanh(distance)
                reward_components["afford"] = (1 + reward) * 5
            if cur_info["stage2_success"]:
                reward_components["success"] = 10.0
            return reward_components
        
        def stage_3_reward():
            target_action = 1
            target_pos = np.array([0.05, -0.3,  0.2])
            if predicted_action == target_action:
                distance = np.linalg.norm(predicted_pos - target_pos)
                reward = -np.tanh(distance)
                reward_components["afford"] = (1 + reward) * 5
            if cur_info["stage3_success"]:
                reward_components["success"] = 10.0
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
    