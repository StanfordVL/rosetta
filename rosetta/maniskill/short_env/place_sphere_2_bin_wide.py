from typing import Any, Dict, Union

import numpy as np
import torch
import torch.random
import sapien
from transforms3d.euler import euler2quat
from mani_skill.envs.utils import randomization

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

import matplotlib.pyplot as plt
import gymnasium as gym
@register_env("PlaceSphere2BinWide", max_episode_steps=200)
class PlaceSphere2BinWideEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    # set some commonly used values
    radius = 0.02 # radius of the sphere
    bin_base_half_len = 0.08 # half side length of the bin's bottom block
    bin_base_half_height = 0.005 # half height of the bin's bottom block
    bin_wall_half_thickness = 0.005 # half thickness of the bin's walls
    bin_wall_half_height = 0.005 
    bin_wall_half_length = 2 * bin_wall_half_thickness + bin_base_half_len
    # bin_wall_half_size = [bin_wall_half_thickness, 2*bin_wall_half_thickness+bin_base_half_len, 2*bin_wall_half_thickness] # The wall block size of the bin: the list represents the half length of the block along the [x, y, z] axis respectively.
    bin_wall_half_size = [bin_wall_half_thickness, bin_base_half_len, 2*bin_wall_half_height] # The wall block size of the bin: the list represents the half length of the block along the [x, y, z] axis respectively.
        
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.2], target=[-0.1, 0, 0])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose_headon = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        pose_birdseye = sapien_utils.look_at([1.0, 0.0, 1.0], [0.0, 0.0, 0.0])
        return [
            CameraConfig(
                "render_camera_headon", 
                pose=pose_headon, 
                width=512, 
                height=512, 
                fov=1, 
                near=0.01, 
                far=100
            )
        ]
        
    def _build_bin(self, name):
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
            builder.add_box_visual(pose, half_size)

      	# build the kinematic bin
        return builder.build_kinematic(name=name)

    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # load the sphere
        self.obj = actors.build_sphere(
            self.scene,
            radius=self.radius,
            color=np.array([12, 42, 160, 255]) / 255,
            name="sphere",
            body_type="dynamic",
        )
        
        # load the bin
        self.bin1 = self._build_bin('bin1')
        self.bin2 = self._build_bin('bin2')

        self.object_list = {"sphere": self.obj, 
                            "bin_1": self.bin1,
                            "bin_2": self.bin2}

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # init the sphere in the first 1/4 zone along the x-axis (so that it doesn't collide the bin)
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b, 1)) * 0.05 - 0.1)[..., 0] # first 1/4 zone of x ([-0.1, -0.05])
            xyz[..., 1] = (torch.rand((b, 1)) * 0.2 - 0.1)[..., 0] # spanning all possible ys
            xyz[..., 2] = self.radius # on the table
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            # init bin1 in the last 1/2 zone along the x-axis (so that it doesn't collide with the sphere)
            # and in the first 1/2 zone along the y-axis (so it doesn't collide with bin2)
            pos1 = torch.zeros((b, 3))
            pos1[:, 0] = torch.rand((b, 1))[..., 0] * 0.05 + 0.05  # ensuring it's within [0.05, 0.1]
            # pos1[:, 1] = torch.rand((b, 1))[..., 0] * 0.2 - 0.1  # spanning all possible ys
            pos1[:, 1] = torch.rand((b, 1))[..., 0] * 0.05         # ensuring it's within [0, 0.05]
            pos1[:, 2] = self.bin_base_half_height  # on the table
            bin_pose1 = Pose.create_from_pq(p=pos1, q=q)
            self.bin1.set_pose(bin_pose1)

            # init bin2 in the last 1/2 zone along the x-axis and make sure it does not collide with bin1
            pos2 = torch.zeros((b, 3))
            pos2[:, 0] = pos1[:, 0] + 0.05   # slightly offset from bin1 along the x-axis 
            pos2[:, 1] = pos1[:, 1] + 0.2       # fixed distance from bin1 along the y-axis 
            pos2[:, 2] = self.bin_base_half_height  # on the table
            bin_pose2 = Pose.create_from_pq(p=pos2, q=q)
            self.bin2.set_pose(bin_pose2)

    def evaluate(self):
        pos_obj = self.obj.pose.p
        pos_bin = self.bin1.pose.p
        offset = pos_obj - pos_bin
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.radius - self.bin_base_half_len) <= 0.005
        is_obj_on_bin = torch.logical_and(xy_flag, z_flag)
        is_obj_grasped = self.agent.is_grasping(self.obj)
        success = is_obj_on_bin
        return {
            "is_obj_grasped": is_obj_grasped,
            "is_obj_on_bin": is_obj_on_bin,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            # is_grasped=info["is_obj_grasped"],
            is_grasped=self.agent.is_grasping(self.obj),
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bin_pos=self.bin1.pose.p
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        reward_components = {}
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        obj_pos = self.obj.pose.p
        obj_to_tcp_dist = torch.linalg.norm(tcp_pose - obj_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))

        # grasp and place reward
        obj_pos = self.obj.pose.p
        bin_pos = self.bin1.pose.p
        bin_top_pos = self.bin1.pose.p.clone()
        bin_top_pos[:, 2] = bin_top_pos[:, 2] + self.bin_base_half_height + self.radius
        obj_to_bin_top_dist = torch.linalg.norm(bin_top_pos - obj_pos, axis=1)
        place_reward = (1 - torch.tanh(5.0 * obj_to_bin_top_dist))

        reward[info["is_obj_grasped"]] = (4 + place_reward)[info["is_obj_grasped"]]

        # success reward
        reward[info["success"]] = 13
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 13.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    
    def get_fitness_score(self):
        # get the fitness score of the current episode
        # returns a tensore of shape (batch_size, )
        # currently, the fitness score is the distance between the sphere and the target bin
        # fitness score always the higher the better so we return the negative distance
        pos_obj = self.obj.pose.p
        pos_bin = self.bin1.pose.p
        return -torch.linalg.norm(pos_obj - pos_bin, axis=1)