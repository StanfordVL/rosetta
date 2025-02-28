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
@register_env("PlaceSphere-v2", max_episode_steps=50)
class PlaceSphereEnv_v2(BaseEnv):
    """
    Task Description
    ----------------
    Place the sphere into the shallow bin.

    Randomizations
    --------------
    The position of the bin and the sphere are randomized: The bin is inited in [0, 0.1]x[-0.1, 0.1], and the sphere is inited in [-0.1, -0.05]x[-0.1, 0.1]
    
    Success Conditions
    ------------------
    The sphere is place on the top of the bin. The robot remains static and the gripper is not closed at the end state
    """

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    # set some commonly used values
    radius = 0.02 # radius of the sphere
    inner_side_half_len = 0.02 # side length of the bin's inner square
    short_side_half_size = 0.0025 # length of the shortest edge of the block
    block_half_size = [short_side_half_size, 2*short_side_half_size+inner_side_half_len, 2*short_side_half_size+inner_side_half_len] # The bottom block of the bin, which is larger: The list represents the half length of the block along the [x, y, z] axis respectively.
    edge_block_half_size = [short_side_half_size, 2*short_side_half_size+inner_side_half_len, 2*short_side_half_size] # The edge block of the bin, which is smaller. The representations are similar to the above one
        
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_cfg=GPUMemoryConfig(
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
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
        
    def _build_bin(self, radius,name):
        builder = self.scene.create_actor_builder()
        
        # init the locations of the basic blocks
        dx = self.block_half_size[1] - self.block_half_size[0] 
        dy = self.block_half_size[1] - self.block_half_size[0] 
        dz = self.edge_block_half_size[2] + self.block_half_size[0]
        
        # build the bin bottom and edge blocks
        poses = [
            sapien.Pose([0, 0, 0]),
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
        ]
        half_sizes = [
            [self.block_half_size[1], self.block_half_size[2], self.block_half_size[0]],
            self.edge_block_half_size,
            self.edge_block_half_size,
            [self.edge_block_half_size[1], self.edge_block_half_size[0], self.edge_block_half_size[2]],
            [self.edge_block_half_size[1], self.edge_block_half_size[0], self.edge_block_half_size[2]],
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
        self.bin1 = self._build_bin(self.radius,'bin1')
        self.bin2 = self._build_bin(2*self.radius,'bin2')

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
            pos1 = torch.zeros((b, 3))
            pos1[:, 0] = torch.rand((b, 1))[..., 0] * 0.05 + 0.05  # ensuring it's within [0.05, 0.1]
            pos1[:, 1] = torch.rand((b, 1))[..., 0] * 0.2 - 0.1  # spanning all possible ys
            pos1[:, 2] = self.block_half_size[0]  # on the table
            bin_pose1 = Pose.create_from_pq(p=pos1, q=q)
            self.bin1.set_pose(bin_pose1)

            # init bin2 in the last 1/2 zone along the x-axis and make sure it does not collide with bin1
            pos2 = torch.zeros((b, 3))
            pos2[:, 0] = pos1[:, 0] + 0.1  # fixed distance from bin1 along the x-axis
            pos2[:, 1] = torch.rand((b, 1))[..., 0] * 0.2 - 0.1  # spanning all possible ys
            pos2[:, 2] = self.block_half_size[0]  # on the table
            bin_pose2 = Pose.create_from_pq(p=pos2, q=q)
            self.bin2.set_pose(bin_pose2)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_grasped=info["is_obj_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bin_pos=self.bin1.pose.p
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def evaluate(self: PlaceSphereEnv_v2) -> Dict[str, bool]:
        """
        Evaluate the current state of the environment and return a dictionary with relevant information.
        """
        info = {}
        
        # Check if the sphere is grasped by the robot
        info["is_obj_grasped"] = self.agent.is_grasping(self.obj)
        
        # Check if the sphere is within the bin
        sphere_pos = self.obj.pose.p
        bin_pos = self.bin1.pose.p
        bin_half_size = self.block_half_size

        # Check if the sphere is within the bin boundaries
        in_bin_x = (bin_pos[0] - bin_half_size[1] <= sphere_pos[0] <= bin_pos[0] + bin_half_size[1])
        in_bin_y = (bin_pos[1] - bin_half_size[2] <= sphere_pos[1] <= bin_pos[1] + bin_half_size[2])
        in_bin_z = (bin_pos[2] <= sphere_pos[2] <= bin_pos[2] + 2 * bin_half_size[0])

        info["in_bin"] = in_bin_x and in_bin_y and in_bin_z
        
        # Check if the task is successful
        info["success"] = info["in_bin"]
        
        return info

    def compute_dense_reward(self: PlaceSphereEnv_v2, obs: Any, action: torch.Tensor, info: Dict[str, bool]) -> float:
        """
        Compute the dense reward based on the current observation, action, and evaluation info.
        """
        reward = 0.0
        
        # Reward for grasping
        if action[0] > action[1]:
            if info["is_obj_grasped"]:
                reward += 1.0  # Positive reward for successfully grasping the sphere
            else:
                reward -= 0.5  # Negative reward for failing to grasp the sphere
        
        # Reward for placing
        if action[0] < action[1]:
            if info["is_obj_grasped"]:
                reward -= 0.1  # Small penalty for holding the object while placing
            if info["in_bin"]:
                reward += 2.0  # Positive reward for successfully placing the sphere in the bin
            else:
                reward -= 0.5  # Negative reward for failing to place the sphere in the bin
        
        # Additional reward for task success
        if info["success"]:
            reward = 3.625  # Large reward for completing the task successfully
        
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 3.625
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward